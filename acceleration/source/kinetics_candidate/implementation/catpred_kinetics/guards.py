"""Lazy framework validation for the experimental discarded-head-mean adapter."""
from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from .source_proof import prove_model_callsite,prove_torch_attention


class UnsupportedKinetics(RuntimeError):pass

def require(value,message):
    if not value:raise UnsupportedKinetics(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_torch():return importlib.import_module('torch')
def own_sources():
    root=Path(__file__).parent
    return {p.name:sha(p) for p in sorted(root.iterdir()) if p.suffix in ('.py','.json')}

HOOK_FIELDS=('_forward_pre_hooks','_forward_hooks','_backward_pre_hooks','_backward_hooks',
             '_forward_pre_hooks_with_kwargs','_forward_hooks_with_kwargs','_forward_hooks_always_called')
TORCH_FILES={'functional.py':'06f330c5ce12f7a7b5be938fb4fd1b78dadcc4a345c49aac58dbb923b3a3a405','activation.py':'8b8076c9383f8faea0800e5312e6df32b0829a6fd9a149ae4dc9620d2d6885fd'}
BASELINE_HOOK_SHA='28fb7ca73d8e1e8b1965c68cedb2011ef54abf3c4efbe308904f786344c5b9c8'

def accepted_counter_hook(fn,index):
    code=getattr(fn,'__code__',None)
    return bool(code is not None and getattr(fn,'__qualname__','')=='Runtime.__init__.<locals>.load.<locals>.hook'
                and getattr(fn,'__defaults__',None)==(index,) and Path(code.co_filename).name=='gpu_request.py'
                and sha(code.co_filename)==BASELINE_HOOK_SHA)

def dispatch_clear(torch):
    dispatch=importlib.import_module('torch.utils._python_dispatch')
    require(not dispatch._get_current_dispatch_mode_stack(),'Torch-dispatch modes are unsupported')
    require(not torch._C._is_torch_function_mode_enabled(),'Torch-function modes are unsupported')

GLOBAL_HOOK_FIELDS=('_global_forward_pre_hooks','_global_forward_hooks','_global_backward_pre_hooks','_global_backward_hooks')

def hooks(module):return tuple((name,tuple(getattr(module,name,{}).items())) for name in HOOK_FIELDS)
def params(module):
    return tuple((id(p),p._version,p.data_ptr(),str(p.dtype),str(p.device),tuple(p.shape),tuple(p.stride())) for p in module.parameters())


@dataclass
class GuardSet:
    runtime: object
    models: tuple
    torch: object
    identity: object
    settings: dict
    source_identity: dict
    model_type: type
    mha_type: type
    mha_forward: object
    mha_code: object
    functional: object
    functional_code: object
    module_hooks: object
    hook_snapshots: tuple
    attention_parameters: tuple
    model_forward: object
    originals: tuple

    def validate_runtime(self):
        require(self.runtime._active and self.runtime._manifest.get('engaged') is True,'Accepted runtime is no longer active')
        require(self.identity.runtime_settings(self.torch)==self.settings,'Precision/thread settings changed during K1 activation')
        require(self.mha_type.forward is self.mha_forward and self.torch.nn.functional.multi_head_attention_forward is self.functional,'PyTorch attention implementation was replaced')
        require(self.mha_forward.__code__ is self.mha_code and self.functional.__code__ is self.functional_code,'Torch attention code was replaced in place')
        dispatch_clear(self.torch)
        require(self.model_type.forward is self.model_forward,'Accepted model forward was replaced')
        require(not any(getattr(self.module_hooks,name,{}) for name in GLOBAL_HOOK_FIELDS),'Global module hooks are unsupported')
        config=self.runtime.components['probes']._CONFIG
        require(config.get('r3') is True and config.get('t1') is False and config.get('t2') is False,'Accepted probe configuration changed')

    def validate_model_call(self,index):
        self.validate_runtime();torch=self.torch;model=self.models[index]
        require(self.runtime._request_active,'K1 forward requires the accepted request context')
        require(not torch.is_grad_enabled(),'K1 requires no_grad at the actual forward')
        require(not torch.is_autocast_enabled() and not torch.is_autocast_enabled('cpu'),'K1 rejects autocast')
        require(not model.training and not model.multihead_attn.training,'K1 requires evaluation mode')
        for module,old in self.hook_snapshots[index]:
            require(hooks(module)==old,'A module hook changed after K1 activation')
            require(not module.training,'A child module entered training mode')
        require(params(model.multihead_attn)==self.attention_parameters[index],'Attention parameters changed after K1 activation')

    def validate_attention_call(self,index,args,kwargs):
        # The frozen model call has exactly three positional arguments. All other
        # public MHA uses remain outside this scope and pass through unchanged.
        require(len(args)==3 and not kwargs,'K1 accepts only the frozen q/k/value callsite')
        torch=self.torch;model=self.models[index];mha=model.multihead_attn
        require(not torch.is_grad_enabled() and not torch.is_autocast_enabled() and not torch.is_autocast_enabled('cpu'),'K1 attention requires FP32 no_grad without autocast')
        require(not model.training and not mha.training,'K1 attention entered training mode')
        q,k,v=args
        require(q is not k and q is not v and k is not v,'K1 requires original distinct rotary q/k and value tensors')
        require(all(type(t) is torch.Tensor for t in args),'Tensor subclasses are unsupported')
        require(all(t.ndim==3 and t.dtype==torch.float32 and t.device.type=='cuda' for t in args),'K1 requires three CUDA FP32 batch-first tensors')
        require(tuple(q.shape)==tuple(k.shape)==tuple(v.shape) and q.shape[-1]==mha.embed_dim,'K1 q/k/value shapes differ')
        require(0<q.shape[0]<=50 and q.shape[1]>0,'K1 requires intact original batches of at most 50 rows')
        require(q.device==k.device==v.device==mha.in_proj_weight.device,'K1 tensor and parameter devices differ')
        require(all(all(s>0 for s in t.stride()) for t in args),'Unsupported tensor strides')
        require(not any(getattr(mha,n,{}) for n in HOOK_FIELDS),'MHA hooks would observe the changed discarded weight result')
        require(not any(getattr(self.module_hooks,n,{}) for n in GLOBAL_HOOK_FIELDS),'Global hooks would observe the changed MHA result')
        relevant=args+tuple(mha.parameters())
        require(not torch.overrides.has_torch_function(relevant),'Torch-function override would change attention dispatch')
        dispatch_clear(torch)
        require(self.mha_type.forward is self.mha_forward and self.mha_forward.__code__ is self.mha_code and self.torch.nn.functional.multi_head_attention_forward is self.functional and self.functional.__code__ is self.functional_code,'Attention implementation changed during forward')
        require(params(mha)==self.attention_parameters[index],'Attention parameters changed during forward')


def validate_production_config(model,mha):
    # model.args.batch_size is the checkpoint's TRAINING batch (32 in the
    # production checkpoints). Inference keeps PredictArgs batch=50 through
    # the frozen streaming helper and validates actual tensor B at the call.
    require(not model.is_atom_bond_targets and model.loss_function=='mve','K1 requires production kcat MVE models')
    require(mha.batch_first and mha._qkv_same_embed_dim and mha.bias_k is None and mha.bias_v is None and not mha.add_zero_attn,'Unsupported original attention configuration')
    require(mha.embed_dim==model.args.seq_embed_dim and mha.num_heads==model.args.seq_self_attn_nheads,'MHA and original model dimensions differ')


def build_guards(runtime,models):
    torch=load_torch();models=tuple(models)
    require(type(runtime).__module__=='catpred_accel.api' and type(runtime).__name__=='Runtime','K1 requires the accepted Runtime object')
    require(runtime._active and not runtime._request_active,'Install K1 inside Runtime.activate and before Runtime.request')
    require(len(models)==10 and tuple(runtime._models)==models,'K1 requires the same ten runtime members')
    manifest=runtime.manifest()
    require(manifest.get('engaged') is True and manifest.get('selected_backend')=='accepted_stream','K1 requires the accepted streaming path')
    require(manifest.get('validation_status')=='verified','Base runtime must have a matching accepted certificate')
    require(manifest.get('requested',{}).get('numeric')=='exact' and manifest['requested'].get('input_budget_bytes')==2<<30,'K1 supports the measured exact default 2 GiB runtime')
    pinned=json.loads((Path(__file__).parent/'accepted_runtime.json').read_text())
    require(manifest.get('source_hashes')==pinned['python_sources'],'Runtime manifest differs from accepted source identity')
    package_root=Path(inspect.getfile(type(runtime))).resolve().parent
    actual={str(p.relative_to(package_root)):sha(p) for p in sorted(package_root.rglob('*.py'))}
    require(actual==pinned['python_sources'],'Runtime Python files differ from accepted source identity')
    require(manifest.get('scientific_sources')==pinned['scientific_sources'],'Runtime scientific source identity differs')
    require(str(torch.__version__)=='2.11.0+cu130' and torch.version.cuda=='13.0','K1 has only a torch 2.11.0+cu130 source-proof plan')
    identity=importlib.import_module('catpred_accel._identity')
    require(identity.verify_scientific_sources()==pinned['scientific_sources'],'Scientific source files changed')
    components=runtime.components
    identity.verify_components(components)
    require(components['streaming']._CONFIG.get('enabled') is True and components['reuse']._CONFIG=={'r1':True,'r2':True},'Accepted reuse/streaming configuration differs')
    config=components['probes']._CONFIG
    require(config.get('r3') is True and config.get('t1') is False and config.get('t2') is False,'K1 requires unchanged R3 with T1/T2 disabled')
    model_module=importlib.import_module('catpred.models.model');model_type=model_module.MoleculeModel
    mha_type=torch.nn.MultiheadAttention
    require(mha_type.__module__=='torch.nn.modules.activation','Unexpected MHA class')
    module_hooks=importlib.import_module('torch.nn.modules.module')
    require(not any(getattr(module_hooks,name,{}) for name in GLOBAL_HOOK_FIELDS),'Global module hooks are unsupported')
    callsite=prove_model_callsite(Path(model_module.__file__).read_text())
    mha_forward=mha_type.forward;functional=torch.nn.functional.multi_head_attention_forward
    require(mha_forward.__module__=='torch.nn.modules.activation' and functional.__module__=='torch.nn.functional','Torch attention function origin changed')
    torch_files={Path(inspect.getfile(fn)).name:sha(inspect.getfile(fn)) for fn in (mha_forward,functional)}
    require(torch_files==TORCH_FILES,'Actual torch source files differ from the reviewed official 2.11.0 files')
    dispatch_clear(torch)
    source_proof=prove_torch_attention(inspect.getsource(mha_forward),inspect.getsource(functional))
    hook_snapshots=[];originals=[];attention_parameters=[]
    require(model_type.forward is components['reuse']._PATCHED_MODEL,'Accepted R2 model forward is not installed')
    snapshots={id(obj):installed for obj,name,installed in runtime._snapshot.installed_instances if name=='forward'}
    for index,model in enumerate(models):
        require(type(model) is model_type and 'forward' not in model.__dict__,'External model forward or subclass is unsupported')
        mha=model.multihead_attn
        require(type(mha) is mha_type and mha.__dict__.get('forward') is snapshots.get(id(mha)),'External attention forward is unsupported')
        require(not model.training and not mha.training,'Set all ten original models to eval before K1 activation')
        validate_production_config(model,mha)
        require(all(p.dtype==torch.float32 and p.device.type=='cuda' for p in model.parameters()),'All member parameters must remain CUDA FP32')
        entry=components['probes']._MODELS[id(model)]
        require(entry['attention'] is mha and entry['original_attention'].__func__ is mha_forward,'Accepted attention wrapper has a different origin')
        allowed={h.id for h in entry['hooks']}|{h.id for h in components['streaming']._HOOKED_MODELS[id(model)]}
        extras=(set(model._forward_pre_hooks)|set(model._forward_hooks))-allowed
        require(all(key in model._forward_pre_hooks and accepted_counter_hook(model._forward_pre_hooks[key],index) for key in extras),'External model hooks are unsupported')
        require(allowed.issubset(set(model._forward_pre_hooks)|set(model._forward_hooks)),'Accepted runtime model hooks are missing')
        require(not model._backward_hooks and not getattr(model,'_backward_pre_hooks',{}),'External backward hooks are unsupported')
        members=[]
        for child in model.modules():
            if child is not model:require(not any(getattr(child,name,{}) for name in HOOK_FIELDS),'External child module hooks are unsupported')
            require(not child.training,'All child modules must be in evaluation mode')
            members.append((child,hooks(child)))
        originals.append((model.forward,mha.forward));hook_snapshots.append(tuple(members));attention_parameters.append(params(mha))
    return GuardSet(runtime,models,torch,identity,identity.runtime_settings(torch),
        dict(candidate_sources=own_sources(),accepted_runtime_context=manifest['context_key'],accepted_runtime_sources=actual,
             model_callsite=callsite,torch_source_proof=source_proof,torch_files=torch_files,torch_version=str(torch.__version__),
             hardware=manifest['hardware'],software=manifest['software'],numerical_validation='not_run'),
        model_type,mha_type,mha_forward,mha_forward.__code__,functional,functional.__code__,module_hooks,tuple(hook_snapshots),tuple(attention_parameters),model_type.forward,tuple(originals))
