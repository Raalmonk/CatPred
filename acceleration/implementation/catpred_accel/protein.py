"""Opt-in ESM2-650M FP32 features; native arithmetic, unused outputs omitted.

Forward arithmetic is adapted from fair-esm 2.0.0 (Meta Platforms, MIT).
The stock model remains the fallback. No model is created at import time.
"""
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import threading
import time
import gc
import os
from functools import lru_cache

from .cache_keys import digest_record

FAIR_SOURCES = {
 'esm.model.esm2':'aadad74d5c2a5b85786390df4071466acc57e028279fe499c97c473125e0d9f4',
 'esm.modules':'6a82388514f8a9a6e67a5636277df5e7a5864421f392b45f3da80717500c57d9',
 'esm.multihead_attention':'81bfc52ff93431d2384562c657cb0578dee8677fb9663babeec1239155b2cbc0',
 'esm.rotary_embedding':'fcdf9be00183786bf77e9db4767ea8baac2470407e29ef8a6a2840ac1e03bcdf',
 'esm.data':'73898ebaa4e7792a98122fef81a9a4087b9aa87eb7ee729a7c270a25a46da230',
 'esm.pretrained':'2104beefc2eec254fb01143b14b89856d03cecf1fb698d1951e0c5c0cbf4a59f',
}
CALLER_SHA='80fa2786ca403bc49c442115c20e9971fa32f9d61c480c0a38028aeb7877f5c4'
WEIGHTS = {'esm2_t33_650M_UR50D.pt':'ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c',
 'esm2_t33_650M_UR50D-contact-regression.pt':'8ffe6edbd4173dc8d45c2cd5cb27d43aad77ec26b4c768200c58ae1f96693575'}
MODES=frozenset(('off','head_only','weights_only','representations'))
_LOCK=threading.RLock()
_ACTIVE=False


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def _source_guard():
    if importlib.metadata.version('fair-esm')!='2.0.0':return 'Unsupported fair-esm distribution'
    for name,expected in FAIR_SOURCES.items():
        if _sha(importlib.import_module(name).__file__)!=expected:return 'ESM source identity differs: '+name
    return None


def _effective(torch):
    return dict(default_dtype=str(torch.get_default_dtype()),default_device=str(torch.get_default_device()),
        threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),cudnn_tf32=bool(torch.backends.cudnn.allow_tf32),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def _context(torch):
    from ._identity import package_sources,software
    gpu=None
    if torch.cuda.is_available():
        p=torch.cuda.get_device_properties(0)
        gpu=dict(name=p.name,capability=[p.major,p.minor],total_memory=p.total_memory,
                 multiprocessors=p.multi_processor_count)
    return dict(kind='esm_features',software=software(torch),hardware=gpu,effective_settings=_effective(torch),
                fair_esm_sources=FAIR_SOURCES,implementation_sources=package_sources(),
                model='esm2_t33_650M_UR50D',layer=33,dtype='float32',numeric='exact')


def _certified(context_key,mode):
    path=Path(__file__).parent/'validation/protein_certifications.json'
    if not path.exists():return False
    return any(r.get('kind')=='esm_features' and r.get('context_key')==context_key and r.get('mode')==mode
               and r.get('passed') is True and r.get('exact_bits') is True and r.get('warm_nonregression') is True
               and isinstance(r.get('evidence_sha256'),str) and len(r['evidence_sha256'])==64
               for r in json.loads(path.read_text()))


@lru_cache(maxsize=64)
def _native_source(method,cls):
    return hasattr(method,'__code__') and Path(method.__code__.co_filename).resolve()==Path(inspect.getfile(cls)).resolve()

def _native_method(obj,name,cls):
    value=getattr(obj,name)
    method=getattr(value,'__func__',None)
    return method is getattr(cls,name) and _native_source(method,cls)


class FeaturesAdapter:
    def __init__(self,model,mode='off',allow_unvalidated=False):
        if mode not in MODES:raise ValueError('Unknown ESM feature mode')
        import torch
        self.torch,self.model,self.mode=torch,model,mode
        if type(allow_unvalidated) is not bool:raise TypeError('allow_unvalidated must be bool')
        self.allow_unvalidated=allow_unvalidated;self.closed=False
        self.context=_context(torch);self.context_key=digest_record(self.context)
        self.source_reason=None if mode=='off' else _source_guard()
        self.certified=_certified(self.context_key,mode)
        self.reset_diagnostics()

    def reset_diagnostics(self):
        self._decision=dict(requested_mode=self.mode,used_mode=None,engaged=False,fallback_reason=None,
            context_key=self.context_key,certified=self.certified,allow_unvalidated=self.allow_unvalidated,
            counters=dict(lm_head_calls=None,attention_calls=None))

    def diagnostics(self):
        return json.loads(json.dumps(self._decision))

    def _reason(self,tokens):
        torch=self.torch;model=self.model
        from esm.model.esm2 import ESM2
        from esm.modules import TransformerLayer
        from esm.multihead_attention import MultiheadAttention
        from esm.rotary_embedding import RotaryEmbedding
        if self.source_reason:return self.source_reason
        if type(model) is not ESM2 or not _native_method(model,'forward',ESM2):return 'Model is not an unchanged native ESM2'
        if len(model.layers)!=33 or (model.num_layers,model.embed_dim,model.attention_heads,model.token_dropout)!=(33,1280,20,True):return 'Unsupported ESM2 configuration'
        if tokens.ndim!=2 or tokens.dtype not in (torch.int32,torch.int64) or tokens.device.type!='cuda':return 'CUDA integer token matrix required'
        if tuple(torch.cuda.get_device_capability(tokens.device)) not in ((9,0),(12,0)):return 'GPU architecture has no candidate contract'
        if torch.__version__!='2.11.0+cu130' or torch.version.cuda!='13.0':return 'GPU software stack has no candidate contract'
        if _effective(torch)!=self.context['effective_settings']:return 'Numeric/runtime settings changed after adapter construction'
        if torch.get_default_dtype()!=torch.float32 or str(torch.get_default_device())!='cpu':return 'Default dtype/device differs from the exact contract'
        if not self.allow_unvalidated and getattr(model,'_catpred_esm_checkpoint_sha256',None)!=WEIGHTS:return 'Loaded checkpoint provenance is not certified'
        if torch.is_grad_enabled():return 'Gradient tracking is enabled'
        if torch.is_autocast_enabled('cuda'):return 'CUDA autocast is enabled'
        if torch.backends.cuda.matmul.allow_tf32 or not torch.backends.cudnn.allow_tf32:return 'Precision settings differ from the exact contract'
        modules=torch.nn.modules.module
        if modules._global_forward_hooks or modules._global_forward_pre_hooks:return 'Global forward hooks are registered'
        for module in model.modules():
            if module.training:return 'Model is not wholly in evaluation mode'
            if module._forward_hooks or module._forward_pre_hooks:return 'Model or layer forward hooks are registered'
        for tensor in tuple(model.parameters())+tuple(model.buffers()):
            if tensor.device!=tokens.device or tensor.is_meta:return 'Model and tokens differ in device'
            if tensor.is_floating_point() and tensor.dtype!=torch.float32:return 'Non-FP32 model state'
        for layer in model.layers:
            if type(layer) is not TransformerLayer or not _native_method(layer,'forward',TransformerLayer):return 'Layer forward was replaced'
            attention=layer.self_attn
            if type(attention) is not MultiheadAttention or not _native_method(attention,'forward',MultiheadAttention):return 'Attention forward was replaced'
            if type(attention.rot_emb) is not RotaryEmbedding or not _native_method(attention.rot_emb,'forward',RotaryEmbedding):return 'Native rotary attention required'
            if attention.onnx_trace:return 'ONNX tracing enabled'
        if not self.allow_unvalidated and not self.certified:return 'Exact feature backend is not certified for this context'
        return None

    def forward(self,tokens):
        if self.closed:raise RuntimeError('FeaturesAdapter is closed')
        self.reset_diagnostics()
        reason=None if self.mode=='off' else self._reason(tokens)
        if self.mode=='off' or reason:
            self._decision.update(used_mode='off',fallback_reason=reason)
            return self.model(tokens,repr_layers=[33])['representations'][33]
        counts=dict(lm_head_calls=0,attention_calls=0)
        value=_representations(self.model,tokens,self.mode,counts)
        self._decision.update(used_mode=self.mode,engaged=True,counters=counts)
        return value

    def close(self):
        self.closed=True
        self.model=None


def _layer_without_attention_output(layer,x,padding_mask,counts):
    from esm.modules import gelu
    residual=x
    x=layer.self_attn_layer_norm(x)
    counts['attention_calls']+=1
    x,_=layer.self_attn(query=x,key=x,value=x,key_padding_mask=padding_mask,
                      need_weights=False,need_head_weights=False,attn_mask=None)
    x=residual+x
    residual=x
    x=layer.final_layer_norm(x)
    x=gelu(layer.fc1(x))
    x=layer.fc2(x)
    return residual+x


def _representations(model,tokens,mode,counts):
    # Exact fair-esm operation order: only dead head/returned attention mean omitted.
    skip_weights=mode in ('representations','weights_only')
    skip_head=mode in ('representations','head_only')
    padding_mask=tokens.eq(model.padding_idx)
    x=model.embed_scale*model.embed_tokens(tokens)
    if model.token_dropout:
        x.masked_fill_((tokens==model.mask_idx).unsqueeze(-1),0.0)
        mask_ratio_train=0.15*0.8
        src_lengths=(~padding_mask).sum(-1)
        mask_ratio_observed=(tokens==model.mask_idx).sum(-1).to(x.dtype)/src_lengths
        x=x*(1-mask_ratio_train)/(1-mask_ratio_observed)[:,None,None]
    if padding_mask is not None:
        x=x*(1-padding_mask.unsqueeze(-1).type_as(x))
    x=x.transpose(0,1)
    if not padding_mask.any():padding_mask=None
    for layer in model.layers:
        if skip_weights:x=_layer_without_attention_output(layer,x,padding_mask,counts)
        else:
            counts['attention_calls']+=1
            x,_=layer(x,self_attn_padding_mask=padding_mask,need_head_weights=False)
    x=model.emb_layer_norm_after(x)
    x=x.transpose(0,1)
    representation=x
    if not skip_head:
        counts['lm_head_calls']+=1
        model.lm_head(x)
    return representation


def preflight(checkpoint_path,regression_path):
    from .protein_loading import preflight as verify
    return verify(checkpoint_path,regression_path)


def load_esm2(checkpoint_path,regression_path,mode='off',allow_unvalidated=False,**kwargs):
    from .protein_loading import load_esm2 as load
    return load(checkpoint_path,regression_path,mode=mode,allow_unvalidated=allow_unvalidated,**kwargs)


def recommended_modes(checkpoint_path,regression_path):
    """Choose only a measured, downstream-accepted beneficial combination."""
    import torch
    from .protein_loading import preflight as verify,_certificate
    result=dict(mode='off',loader_mode='off',reason='No certified beneficial ESM profile',feature_context_key=None,loader_context_key=None)
    try:
        prepared=verify(checkpoint_path,regression_path)
        if prepared['support_reason'] is not None:
            result['reason']=prepared['support_reason'];return result
        feature_key=digest_record(_context(torch));loader_key=prepared['context_key']
        result.update(feature_context_key=feature_key,loader_context_key=loader_key)
        path=Path(__file__).parent/'validation/protein_certifications.json'
        records=json.loads(path.read_text()) if path.exists() else []
        for record in records:
            feature=record.get('feature_mode');loader=record.get('loader_mode')
            if not (record.get('kind')=='esm_profile' and record.get('feature_context_key')==feature_key
                    and record.get('loader_context_key')==loader_key and feature in MODES and loader in ('off','meta')):continue
            if not all(record.get(k) is True for k in ('passed','exact_bits','warm_nonregression','downstream_exact','beneficial')):continue
            if not isinstance(record.get('evidence_sha256'),str) or len(record['evidence_sha256'])!=64:continue
            if feature!='off' and not _certified(feature_key,feature):continue
            if loader!='off' and _certificate(loader_key) is None:continue
            return dict(result,mode=feature,loader_mode=loader,reason='Certified beneficial same-context profile',profile=record)
    except (FileNotFoundError,ValueError,ImportError) as error:
        result['reason']=type(error).__name__+': '+str(error)
    return result


class _FeatureContext:
    def __init__(self,mode,loader_mode):
        self.record=dict(requested_mode=mode,requested_loader_mode=loader_mode,engaged=False,
                         closed=False,forward_batches=[],loader_receipts=[],cache_namespace=None)
    def summary(self):return json.loads(json.dumps(self.record))
    def identity(self):return json.loads(json.dumps(self.record.get('identity',{'mode':'off','loader_mode':'off'})))
    def release_model(self):
        release=getattr(self,'_release',None)
        if release is not None:release()


@contextmanager
def features_context(esm_utils,*,sequences,mode='off',loader_mode='off',allow_unvalidated=False,cache_root=None,batch_observer=None):
    """Reversible original get_many/B4/cache integration; preflight performs no model load.

    Enter before the measured request to perform the same source/weight identity
    checks as the reference. Model loading, original tokenization, forwards and
    CPU result storage occur only when the original get_many requests an ESM batch.
    """
    global _ACTIVE
    if mode not in MODES or loader_mode not in ('off','meta'):raise ValueError('Unknown ESM mode')
    if type(allow_unvalidated) is not bool:raise TypeError('allow_unvalidated must be bool')
    handle=_FeatureContext(mode,loader_mode)
    if mode=='off' and loader_mode=='off':
        try:yield handle
        finally:handle.record['closed']=True
        return
    import torch
    from .protein_loading import preflight
    with _LOCK:
        if _ACTIVE:raise RuntimeError('Nested ESM feature contexts are unsupported')
        assert _sha(esm_utils.__file__)==CALLER_SHA,'Original ESM caller changed'
        assert esm_utils.DEFAULT_ESM_BATCH_SIZE==4 and esm_utils.ESM_MAX_LENGTH==2048
        assert not esm_utils.PROTEIN_EMBED_USE_CPU,'GPU feature candidate requires original CUDA caller'
        assert _source_guard() is None,'ESM source identity differs'
        ordered=list(dict.fromkeys(sequences));assert ordered and all(isinstance(s,str) and s for s in ordered)
        checkpoints=Path(torch.hub.get_dir())/'checkpoints'
        model_path=checkpoints/'esm2_t33_650M_UR50D.pt';regression_path=checkpoints/'esm2_t33_650M_UR50D-contact-regression.pt'
        preflight_receipt=preflight(model_path,regression_path)
        context=_context(torch)
        namespace=digest_record(dict(schema=1,kind='esm_features_actual_cache',context=context,
            weights=WEIGHTS,caller=CALLER_SHA,mode=mode,loader_mode=loader_mode,
            ordered_sequence_hashes=[hashlib.sha256(s.encode()).hexdigest() for s in ordered],
            batch_size=4,max_tokens=2048,layer=33,dtype='float32',padding='original ordered group of four'))
        old=(esm_utils._run_esm_batch,esm_utils.init_esm,esm_utils.ESM_CACHE_PATH,esm_utils.GLOBAL_VARIABLES['model'])
        old_many=esm_utils.get_many_esm_reprs
        old_config=esm_utils.PROTEIN_REPR_CONFIG['esm']['batch_fn']
        state={'adapter':None,'model':None,'alphabet':None}
        expected=tuple(ordered)
        handle.record['identity']=dict(mode=mode,loader_mode=loader_mode,cache_namespace=namespace,context=context,weights=WEIGHTS,caller_sha256=CALLER_SHA,ordered_request_sha256=digest_record(ordered),batch_size=4,max_tokens=2048)
        handle.record.update(cache_namespace=namespace,preflight=preflight_receipt,context=context,
                             ordered_request_sha256=digest_record(ordered),application_cache_reused=False)
        def initialize():
            if state['model'] is not None:return
            model,alphabet,receipt=load_esm2(model_path,regression_path,mode=loader_mode,allow_unvalidated=allow_unvalidated)
            model.eval();model=model.cuda()
            state.update(model=model,alphabet=alphabet,adapter=FeaturesAdapter(model,mode=mode,allow_unvalidated=allow_unvalidated))
            esm_utils.GLOBAL_VARIABLES['model']=(model,alphabet.get_batch_converter())
            handle.record['loader_receipts'].append(receipt)
        def batch(proteins):
            initialize();model,converter=esm_utils.GLOBAL_VARIABLES['model']
            _,_,tokens=converter([(f'protein_{i}',s) for i,s in enumerate(proteins)])
            tokens=tokens[:,:esm_utils.ESM_MAX_LENGTH].to(next(model.parameters()).device)
            with torch.no_grad():value=state['adapter'].forward(tokens)
            decision=state['adapter'].diagnostics()
            floating=[t for t in tuple(model.parameters())+tuple(model.buffers()) if t.is_floating_point()]
            assert not model.training and floating and all(t.device.type=='cuda' and t.dtype==torch.float32 for t in floating)
            event=dict(sequences=[hashlib.sha256(s.encode()).hexdigest() for s in proteins],device='cuda',dtype='torch.float32',eval=True,all_floating_parameters_buffers_cuda_fp32=True,floating_parameter_buffer_count=len(floating),decision=decision)
            handle.record['forward_batches'].append(event)
            if batch_observer is not None:batch_observer(json.loads(json.dumps(event)))
            handle.record['engaged'] |= decision['engaged'] or any(r.get('engaged',False) for r in handle.record['loader_receipts'])
            return [value[i][1:min(len(s),esm_utils.ESM_MAX_LENGTH-1)+1].detach() for i,s in enumerate(proteins)]
        def many(proteins,device='cpu',batch_size=None):
            # The namespace is conservative: an exact whole ordered request, not a shape-free sequence cache.
            seqs=esm_utils.tensor_to_aa_str(proteins) if isinstance(proteins,torch.Tensor) else list(proteins)
            assert tuple(dict.fromkeys(seqs))==expected,'ESM cache request/padding context differs'
            assert int(batch_size or esm_utils.DEFAULT_ESM_BATCH_SIZE)==4,'Original ESM batch size changed'
            # Complete namespace transactions prevent a partial hit set from
            # regrouping uncached sequences and changing their padding context.
            from catpred.data import cache_utils
            directory=cache_utils.CACHE_PATH/esm_utils.ESM_CACHE_PATH
            directory.mkdir(parents=True,exist_ok=True)
            marker=directory/'complete.json'
            paths={hashlib.md5(s.encode()).hexdigest()+'.pt' for s in ordered}
            complete=False
            if marker.exists():
                saved=json.loads(marker.read_text())
                complete=(saved.get('namespace')==namespace and set(saved.get('entries',{}))==paths
                          and all((directory/name).is_file() and [(directory/name).stat().st_size,(directory/name).stat().st_mtime_ns]==stamp for name,stamp in saved['entries'].items()))
            if not complete and (marker.exists() or any(directory.glob('*.pt'))):
                quarantine=directory.with_name(directory.name+'.incomplete_'+str(time.time_ns()))
                directory.replace(quarantine);directory.mkdir()
                handle.record.setdefault('incomplete_namespaces_preserved',[]).append(str(quarantine))
            if complete:handle.record['application_cache_reused']=True
            result=old_many(seqs,device=device,batch_size=4)
            if not complete:
                # OOM splitting produces different batches: refuse to publish a
                # reusable complete namespace instead of treating it as exact.
                batch_events=handle.record['forward_batches'][-((len(ordered)+3)//4):]
                assert [e['sequences'] for e in batch_events]==[[hashlib.sha256(s.encode()).hexdigest() for s in ordered[i:i+4]] for i in range(0,len(ordered),4)],'Original ESM groups changed'
                entries={name:[(directory/name).stat().st_size,(directory/name).stat().st_mtime_ns] for name in paths}
                tmp=directory/'complete.json.tmp';tmp.write_text(json.dumps(dict(namespace=namespace,entries=entries),sort_keys=True));tmp.replace(marker)
            return result
        def release_model():
            if state.get('adapter') is not None:state['adapter'].close()
            state.update(adapter=None,model=None,alphabet=None)
            esm_utils.GLOBAL_VARIABLES['model']=None
            gc.collect();torch.cuda.empty_cache()
            handle.record['model_released']=True
        handle._release=release_model
        try:
            _ACTIVE=True
            esm_utils.GLOBAL_VARIABLES['model']=None
            esm_utils.ESM_CACHE_PATH=str(Path(cache_root or old[2])/'catpred_exact'/namespace)
            esm_utils.init_esm=initialize;esm_utils._run_esm_batch=batch
            esm_utils.get_many_esm_reprs=many;esm_utils.PROTEIN_REPR_CONFIG['esm']['batch_fn']=many
            yield handle
        finally:
            esm_utils._run_esm_batch,esm_utils.init_esm,esm_utils.ESM_CACHE_PATH,esm_utils.GLOBAL_VARIABLES['model']=old
            esm_utils.get_many_esm_reprs=old_many;esm_utils.PROTEIN_REPR_CONFIG['esm']['batch_fn']=old_config
            if state.get('adapter') is not None:state['adapter'].close()
            state.clear();handle._release=None;handle.record['closed']=True;_ACTIVE=False


# MIT License (fair-esm-derived forward operations above)
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
