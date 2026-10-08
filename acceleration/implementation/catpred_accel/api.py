"""Explicit, serialized, reversible integration of the accepted S_STREAM path."""
from contextlib import contextmanager, ExitStack
from dataclasses import asdict, replace
import importlib
import inspect
import json
from pathlib import Path
import threading

from .config import RuntimeConfig
from .capabilities import Capability, resolve, UnsupportedCapability
from .cache_keys import digest_record
from .source import SourceMismatch
from . import _identity, _vendor

_LOCK = threading.RLock()
_OWNER = None
_ABSENT = object()


class RuntimeStateError(RuntimeError):
    pass


class RestorationConflict(RuntimeStateError):
    pass


class _Snapshot:
    """Restore caller state; remove only hooks and attributes owned by this scope."""
    def __init__(self, components, models):
        self.components, self.models = components, models
        data = importlib.import_module('catpred.data.data')
        feat = importlib.import_module('catpred.features.featurization')
        model = importlib.import_module('catpred.models.model')
        mpn = importlib.import_module('catpred.models.mpn')
        predict = importlib.import_module('catpred.train.predict')
        unc = importlib.import_module('catpred.uncertainty.uncertainty_predictor')
        self.targets = [(feat.BatchMolGraph,'__init__'),(data.MoleculeDataset,'batch_graph'),
            (model.MoleculeModel,'forward'),(mpn.MPNEncoder,'forward'),
            (predict,'predict'),(unc,'predict'),(unc.MVEPredictor,'calculate_predictions')]
        self.original = [(obj,name,getattr(obj,name)) for obj,name in self.targets]
        self.instances = []
        self.hooks = []
        for m in models:
            for obj,name in ((m.rotary_embedder,'rotate_queries_or_keys'),(m.multihead_attn,'forward')):
                self.instances.append((obj,name,obj.__dict__.get(name,_ABSENT)))
            self.hooks.append((m,set(m._forward_pre_hooks),set(m._forward_hooks)))
        names = {
            'packing':['_COUNTS','_ARM','_INSTRUMENT','_RUST_PACK'],
            'reuse':['_CONFIG','_INSPECTION','_MODEL_IDS','_PATCHED_MODEL','_PATCHED_MPN'],
            'probes':['_CONFIG','_MODELS','_T1_APPROVED','_ORIGINAL_PREDICT','_COUNTED_PREDICT',
                '_ORIGINAL_UNCERTAINTY_PREDICT','_PREDICT_SOURCE_SHA256','_ORIGINAL_PREDICT_MODULE','_INSTALLED'],
            'streaming':['_CONFIG','_HOST_AGGREGATE','_INSPECTION','_SOURCE_PROOF','_HOOKED_MODELS'],
        }
        self.globals = []
        for module,names in names.items():
            for name in names:
                value=getattr(components[module],name)
                saved=value.copy() if isinstance(value,(dict,set)) else value
                self.globals.append((components[module],name,value,saved))
        self.installed = None
        self.installed_instances = None
        self.owned_handles = []

    def installed_now(self):
        self.installed = [(obj,name,getattr(obj,name)) for obj,name in self.targets]
        self.installed_instances = [(obj,name,obj.__dict__.get(name,_ABSENT)) for obj,name,_ in self.instances]
        for module,key in ((self.components['probes'],'_MODELS'),(self.components['streaming'],'_HOOKED_MODELS')):
            old = next(saved for mod,name,original,saved in self.globals if mod is module and name==key)
            for ident,value in getattr(module,key).items():
                if ident not in old:
                    self.owned_handles.extend(value['hooks'] if key=='_MODELS' else value)

    def restore(self):
        # Also handles a partial inspect/installation failure.
        partial_installation=self.installed is None
        if partial_installation:
            self.installed_now()
        conflicts=[]
        for handle in self.owned_handles:
            handle.remove()
        if partial_installation:
            # A hook's handle is committed to the helper registry only after
            # both registrations return. A failure in the second registration
            # can leave the first hook without a published handle. No caller
            # body has been entered yet; remove only entries created since the
            # installation snapshot, including PyTorch's companion registries.
            for model,preexisting,postexisting in self.hooks:
                for key in set(model._forward_pre_hooks)-preexisting:
                    model._forward_pre_hooks.pop(key,None)
                    getattr(model,'_forward_pre_hooks_with_kwargs',{}).pop(key,None)
                for key in set(model._forward_hooks)-postexisting:
                    model._forward_hooks.pop(key,None)
                    getattr(model,'_forward_hooks_with_kwargs',{}).pop(key,None)
                    getattr(model,'_forward_hooks_always_called',{}).pop(key,None)
        for (obj,name,old),(_,_,installed) in zip(self.original,self.installed):
            if getattr(obj,name) is not installed:
                conflicts.append(type(obj).__name__+'.'+name)
                continue
            setattr(obj,name,old)
        for (obj,name,old),(_,_,installed) in zip(self.instances,self.installed_instances):
            if obj.__dict__.get(name,_ABSENT) is not installed:
                conflicts.append(type(obj).__name__+'.'+name)
                continue
            if old is _ABSENT:
                obj.__dict__.pop(name,None)
            else:
                setattr(obj,name,old)
        for module,name,original,saved in self.globals:
            if isinstance(original,(dict,set)):
                original.clear(); original.update(saved)
            setattr(module,name,original)
        if conflicts:
            raise RestorationConflict('Caller replaced adapter-owned entries: '+', '.join(conflicts))


class RequestReceipt:
    def __init__(self, runtime, diagnostic):
        self.runtime, self.diagnostic = runtime, bool(diagnostic)
        self.stream = self.probes = None
        self.closed = False
        self.failure = None
        self._summary = None

    def summary(self):
        if self._summary is None:
            raise RuntimeStateError('Request receipt is finalized on context exit')
        return self._summary


class Runtime:
    def __init__(self, config=None, *, components=None, certifications=None):
        self.config = config or RuntimeConfig()
        if not isinstance(self.config,RuntimeConfig):
            raise TypeError('Expected RuntimeConfig')
        self._provided = components
        self.components = None
        self.certifications = certifications
        self._private_names = []
        self._active = self._request_active = False
        self._manifest = {'requested':asdict(self.config),'selected_backend':'not_selected','engaged':False,
                          'validation_status':'not_run','reason':'Activation has not occurred'}
        self._snapshot = None
        self._models = self._scalers = None
        self._live = None

    def manifest(self):
        return dict(self._manifest,active=self._active)

    def _matching_certificate(self,key):
        records=self.certifications
        if records is None:
            path=Path(__file__).resolve().parent/'validation/certifications.json'
            records=json.loads(path.read_text()) if path.exists() else []
        for record in records:
            if (record.get('context_key')==key and record.get('passed') is True
                    and record.get('exact_bits') is True and record.get('lifecycle_passed') is True
                    and isinstance(record.get('evidence_sha256'),str) and len(record['evidence_sha256'])==64):
                return record
        return None

    def _prepare(self,models,scalers):
        import torch
        self._models,self._scalers=list(models),list(scalers)
        self._torch=torch
        if not self._models:
            raise RuntimeStateError('Activation requires resident models')
        device=next(self._models[0].parameters()).device
        software=_identity.software(torch)
        if device.type=='cuda':
            prop=torch.cuda.get_device_properties(device)
            hardware=dict(device_type='cuda',name=prop.name,compute_capability=[prop.major,prop.minor],
                          total_memory=prop.total_memory,multiprocessors=prop.multi_processor_count)
        else:
            hardware=dict(device_type=device.type,name='CPU' if device.type=='cpu' else str(device),compute_capability=None)
        errors=[]
        try:
            scientific=_identity.verify_scientific_sources()
            model_info=_identity.model_signature(self._models,self._scalers,torch)
        except (SourceMismatch,AttributeError) as error:
            scientific,model_info={},[]
            errors.append(str(error))
        try:
            native=importlib.import_module('catpred_rust_packing')
            native_ok=native.BUILD_PROFILE=='release'
            import hashlib
            native_file=Path(native.__file__).resolve()
            extension_paths=([native_file] if native_file.suffix in ('.so','.pyd') else
                             sorted(native_file.parent.glob('catpred_rust_packing*.so')))
            native_identity={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in extension_paths}
            native_ok=native_ok and bool(native_identity)
        except (ImportError,AttributeError):
            native_ok=False;native_identity={}
        context=dict(schema=1,software=software,hardware=hardware,scientific_sources=scientific,
                     implementation_sources=_identity.package_sources(),native_extensions=native_identity,
                     models=model_info,memory=self.config.memory,input_budget_bytes=self.config.input_budget_bytes)
        key=digest_record(context)
        certificate=self._matching_certificate(key)
        capability=Capability(device_type=device.type,
            compute_capability=tuple(hardware['compute_capability']) if hardware['compute_capability'] else None,
            device_name=hardware['name'],source_compatible=not errors,rust_available=native_ok,
            dtype='float32' if all(p.dtype==torch.float32 for m in self._models for p in m.parameters()) else 'other',
            software_key=digest_record(software),validated_context=certificate is not None)
        decision=resolve(self.config,capability)
        self._manifest=dict(decision.as_dict(),requested=asdict(self.config),context_key=key,
                            software=software,hardware=hardware,source_hashes=context['implementation_sources'],
                            scientific_sources=scientific,native_extensions=native_identity,
                            model_config_sha256=digest_record(model_info),compatibility_errors=errors,
                            certificate=certificate,feature_backend='external_consumed_features',
                            predictor_input_contract='original_esm2_layer33_fp32',t1=False,t2=False,
                            input_budget_scope='retained device inputs, not total device memory')
        if decision.engaged:
            # The byte-preserved scheduler accepts these production budgets.
            if self.config.input_budget_bytes not in (1<<30,2<<30,4<<30):
                raise UnsupportedCapability('Accepted scheduler supports only 1/2/4 GiB input budgets')
            self._live=_identity.live_signature(self._models,self._scalers)
            self._precision=(_identity.runtime_settings(torch),torch.is_autocast_enabled())
        return decision.engaged

    @contextmanager
    def activate(self,models,scalers):
        global _OWNER
        with _LOCK:
            if _OWNER is not None or self._active:
                raise RuntimeStateError('Concurrent/nested global adapter activations are unsupported')
            _OWNER=self
            try:
                engaged=self._prepare(models,scalers)
                if self._provided is not None:
                    self.components=dict(self._provided)
                    _identity.verify_components(self.components)
                elif engaged:
                    self.components,self._private_names=_vendor.load_private()
                if engaged:
                    c=self.components
                    if (c['reuse']._STATE.get() is not None or c['probes']._ACTIVE.get() is not None
                            or c['streaming']._ACTIVE.get() is not None):
                        raise RuntimeStateError('Existing accepted request context is active')
                    if (c['streaming']._CONFIG['enabled'] or any(c['reuse']._CONFIG.values())
                            or any(c['probes']._CONFIG.values())):
                        raise RuntimeStateError('Injected components must first be configured to original A/off')
                    self._snapshot=_Snapshot(c,self._models)
                    c['reuse'].inspect_models(self._models,self._scalers)
                    c['probes'].inspect_models(self._models,self._scalers)
                    c['streaming'].inspect_models(self._models,self._scalers)
                    c['streaming'].configure(enabled=False,budget_bytes=self.config.input_budget_bytes)
                    c['packing'].install_arm('C',instrument=True)
                    c['reuse'].configure(r1=True,r2=True)
                    c['probes'].configure(r3=True,t1=False,t2=False)
                    c['streaming'].configure(enabled=True,budget_bytes=self.config.input_budget_bytes)
                    self._snapshot.installed_now()
                self._active=True
                yield self
            finally:
                try:
                    if self._snapshot is not None:
                        self._snapshot.restore()
                        self._manifest['restored']=True
                finally:
                    self._snapshot=None
                    self._active=False
                    self._models=self._scalers=self._live=None
                    if self._private_names:
                        _vendor.unload_private(self._private_names)
                        self._private_names=[]
                        self.components=None
                    _OWNER=None

    @contextmanager
    def request(self,diagnostic=False):
        if not self._active or self._request_active:
            raise RuntimeStateError('A request requires one active non-nested runtime')
        self._request_active=True
        receipt=RequestReceipt(self,diagnostic)
        c=self.components
        initial_empty=(c is None or (c['reuse']._STATE.get() is None and c['probes']._ACTIVE.get() is None
                                     and c['streaming']._ACTIVE.get() is None))
        if not initial_empty:
            self._request_active=False
            raise RuntimeStateError('Request caches must begin empty')
        if self._manifest['engaged'] and _identity.live_signature(self._models,self._scalers)!=self._live:
            self._request_active=False
            raise RuntimeStateError('Inspected member parameters/scalers changed after activation')
        if self._manifest['engaged']:
            torch=self._torch
            effective=(_identity.runtime_settings(torch),torch.is_autocast_enabled())
            if effective!=self._precision:
                self._request_active=False
                raise RuntimeStateError('Precision/thread settings changed after activation')
            for obj,name,installed in self._snapshot.installed:
                if getattr(obj,name) is not installed:
                    self._request_active=False
                    raise RuntimeStateError('An installed runtime entry was replaced before request')
        try:
            with ExitStack() as stack:
                if c is not None:
                    c['packing'].reset_counts()
                    if self._manifest['engaged']:
                        receipt.stream=stack.enter_context(c['streaming'].request_context(diagnostic=diagnostic))
                    receipt.probes=stack.enter_context(c['probes'].request_context(diagnostic=diagnostic))
                yield receipt
        except BaseException as error:
            receipt.failure=type(error).__name__+': '+str(error)
            raise
        finally:
            receipt.closed=True
            empty=(c is None or (c['reuse']._STATE.get() is None and c['probes']._ACTIVE.get() is None
                                 and c['streaming']._ACTIVE.get() is None))
            receipt._summary=dict(stream=receipt.stream.summary() if receipt.stream is not None else None,
                probes=receipt.probes.summary() if receipt.probes is not None else None,reuse=None,
                closed=True,request_cache_initial_empty=initial_empty,request_cache_empty_on_exit=empty,
                failure=receipt.failure,backend=self.manifest(),packing=c['packing'].counts() if c is not None else None)
            self._request_active=False
