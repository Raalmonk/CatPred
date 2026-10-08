"""Explicit local ESM2 loading; experimental meta construction preserves FP32.

The optimized path reuses fair-esm's own version-pinned architecture and key
upgrade, assigns complete CPU checkpoint tensors, then restores the native tied
embedding object. No GPU transfer or inference occurs in this module.
"""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import time

MODEL_NAME='esm2_t33_650M_UR50D'
CHECKPOINT_SHA256={
    MODEL_NAME+'.pt':'ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c',
    MODEL_NAME+'-contact-regression.pt':'8ffe6edbd4173dc8d45c2cd5cb27d43aad77ec26b4c768200c58ae1f96693575',
}
FAIR_ESM_SOURCES={
    'esm.model.esm2':'aadad74d5c2a5b85786390df4071466acc57e028279fe499c97c473125e0d9f4',
    'esm.modules':'6a82388514f8a9a6e67a5636277df5e7a5864421f392b45f3da80717500c57d9',
    'esm.multihead_attention':'81bfc52ff93431d2384562c657cb0578dee8677fb9663babeec1239155b2cbc0',
    'esm.rotary_embedding':'fcdf9be00183786bf77e9db4767ea8baac2470407e29ef8a6a2840ac1e03bcdf',
    'esm.data':'73898ebaa4e7792a98122fef81a9a4087b9aa87eb7ee729a7c270a25a46da230',
    'esm.pretrained':'2104beefc2eec254fb01143b14b89856d03cecf1fb698d1951e0c5c0cbf4a59f',
}
_FILE_HASH_CACHE={}
_HASH_COUNTS=dict(bytes_read=0,files_hashed=0,cache_hits=0)


def _sha(path):
    path=Path(path).resolve();stat=path.stat()
    stamp=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    old=_FILE_HASH_CACHE.get(str(path))
    if old is not None and old[0]==stamp:
        _HASH_COUNTS['cache_hits']+=1
        return old[1]
    h=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(8<<20),b''):
            h.update(block);_HASH_COUNTS['bytes_read']+=len(block)
    _HASH_COUNTS['files_hashed']+=1
    # A writer must not change the file while its content identity is established.
    after=path.stat()
    assert stamp==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
    result=h.hexdigest();_FILE_HASH_CACHE[str(path)]=(stamp,result)
    return result


def preflight(checkpoint_path,regression_path):
    """Verify/cache immutable file identities before the application request timer.

    Only bytes and metadata are read. No checkpoint tensors/model are loaded.
    Each later loader invocation rechecks file metadata; changed files are hashed
    again before use. This is symmetric with the benchmark's stock weight setup.
    """
    from ._identity import software,package_sources
    from .cache_keys import digest_record
    import torch
    import importlib.metadata
    hash_start=dict(_HASH_COUNTS)
    paths=[Path(checkpoint_path).resolve(),Path(regression_path).resolve()]
    weights={}
    for path,name in zip(paths,CHECKPOINT_SHA256):
        if not path.is_file():raise FileNotFoundError(str(path))
        actual=_sha(path)
        if actual!=CHECKPOINT_SHA256[name]:raise ValueError('Official ESM2 checkpoint identity mismatch: '+name)
        weights[name]=actual
    actual_sources={}
    for name in FAIR_ESM_SOURCES:
        module=importlib.import_module(name)
        actual_sources[name]=_sha(module.__file__)
    hardware=dict(device_type='cpu',name='CPU',compute_capability=None)
    if torch.cuda.is_available():
        props=torch.cuda.get_device_properties(torch.cuda.current_device())
        hardware=dict(device_type='cuda',name=props.name,compute_capability=[props.major,props.minor],
                      total_memory=props.total_memory,multiprocessors=props.multi_processor_count)
    context=dict(schema=1,kind='esm_loader',mode='meta',software=software(torch),hardware=hardware,
                 fair_esm_version=importlib.metadata.version('fair-esm'),fair_esm_sources=actual_sources,
                 weights=weights,implementation_sources=package_sources(),output_device='cpu',dtype='torch.float32')
    reason=None
    if context['fair_esm_version']!='2.0.0':reason='Unsupported fair-esm version'
    elif actual_sources!=FAIR_ESM_SOURCES:reason='Pinned fair-esm source identity differs'
    elif torch.get_default_dtype()!=torch.float32:reason='Default dtype is not unchanged FP32'
    elif str(torch.get_default_device())!='cpu':reason='Default tensor device is not CPU'
    else:
        import inspect
        if 'assign' not in inspect.signature(torch.nn.Module.load_state_dict).parameters:reason='load_state_dict assign is unavailable'
    return dict(context_key=digest_record(context),context=context,support_reason=reason,
                checkpoint_path=str(paths[0]),regression_path=str(paths[1]),weight_metadata_checked=True,
                file_validation={key:_HASH_COUNTS[key]-hash_start[key] for key in _HASH_COUNTS})


def _certificate(context_key,certifications=None):
    if certifications is None:
        path=Path(__file__).resolve().parent/'validation/protein_certifications.json'
        certifications=json.loads(path.read_text()) if path.is_file() else []
    for record in certifications:
        if (record.get('kind')=='esm_loader' and record.get('mode')=='meta'
                and record.get('context_key')==context_key and record.get('passed') is True
                and record.get('exact_bits') is True and record.get('warm_nonregression') is True
                and isinstance(record.get('evidence_sha256'),str) and len(record['evidence_sha256'])==64):
            return deepcopy(record)
    return None


def _rng_sha(torch):
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()


def _meta_from_payloads(model_data,regression_data):
    import esm
    import torch
    cfg=model_data['cfg']['model']
    if (cfg.encoder_layers!=33 or cfg.encoder_embed_dim!=1280 or cfg.encoder_attention_heads!=20
            or cfg.token_dropout is not True):
        raise ValueError('Checkpoint is not the production ESM2 t33 650M architecture')
    combined=dict(model_data);combined['model']=dict(model_data['model'])
    combined['model'].update(regression_data['model'])
    with torch.device('meta'):
        model,alphabet,state=esm.pretrained._load_model_and_alphabet_core_v2(combined)
    expected=model.state_dict()
    missing,unexpected=set(expected)-set(state),set(state)-set(expected)
    if missing or unexpected:raise ValueError('Incomplete checkpoint keys: missing='+repr(sorted(missing))+' unexpected='+repr(sorted(unexpected)))
    for name,target in expected.items():
        source=state[name]
        if not torch.is_tensor(source) or source.device.type!='cpu' or source.layout!=torch.strided:
            raise ValueError('Checkpoint value is not a strided CPU tensor: '+name)
        if source.shape!=target.shape or source.dtype!=target.dtype or source.stride()!=target.stride():
            raise ValueError('Checkpoint shape/dtype/stride differs from the native model: '+name)
        if source.is_floating_point() and source.dtype!=torch.float32:raise ValueError('Non-FP32 checkpoint tensor: '+name)
    embed,head=state['embed_tokens.weight'],state['lm_head.weight']
    if not torch.equal(embed.contiguous().view(torch.uint8),head.contiguous().view(torch.uint8)):
        raise ValueError('Tied embedding and LM head checkpoint bits differ')
    model.load_state_dict(state,strict=True,assign=True)
    model.lm_head.weight=model.embed_tokens.weight
    for name,tensor in list(model.named_parameters())+list(model.named_buffers()):
        if tensor.is_meta or tensor.device.type!='cpu':raise ValueError('Unmaterialized/non-CPU tensor: '+name)
        if tensor.is_floating_point() and tensor.dtype!=torch.float32:raise ValueError('Non-FP32 assigned tensor: '+name)
    for index,layer in enumerate(model.layers):
        rotary=layer.self_attn.rot_emb
        if rotary.inv_freq.is_meta or rotary.inv_freq.device.type!='cpu':raise ValueError('Unmaterialized rotary buffer '+str(index))
        if any(getattr(rotary,n) is not None for n in ('_seq_len_cached','_cos_cached','_sin_cached')):
            raise ValueError('Nonempty initial rotary cache '+str(index))
    if model.lm_head.weight is not model.embed_tokens.weight:raise ValueError('Embedding tie was not restored')
    proof=dict(complete_state_key_count=len(expected),missing_keys=[],unexpected_keys=[],all_shapes_dtypes_strides_native=True,
               all_parameters_buffers_cpu_fp32=True,no_meta_tensors=True,embedding_lm_head_same_object=True,
               rotary_layer_count=len(model.layers),rotary_initial_caches_empty=True,
               checkpoint_state_assignment='strict=True, assign=True; native embedding object tie restored')
    return model,alphabet,proof


def load_esm2(checkpoint_path,regression_path,*,mode='off',allow_unvalidated=False,certifications=None):
    """Return a CPU FP32 model, native alphabet and explicit selection receipt."""
    if mode not in ('off','meta'):raise ValueError('Unknown ESM2 loader mode: '+str(mode))
    if not isinstance(allow_unvalidated,bool):raise TypeError('allow_unvalidated must be bool')
    import torch
    import esm
    if torch.get_default_dtype()!=torch.float32 or str(torch.get_default_device())!='cpu':
        raise ValueError('Local ESM2 loader requires the original CPU/FP32 construction defaults')
    start=time.perf_counter();pre=preflight(checkpoint_path,regression_path)
    certificate=_certificate(pre['context_key'],certifications)
    reason=pre['support_reason']
    if mode=='meta' and reason is None and certificate is None and not allow_unvalidated:
        reason='Loader context has no matching exact and warm-nonregression certificate'
    used='meta' if mode=='meta' and reason is None else 'off'
    receipt=dict(requested_mode=mode,used_mode=used,engaged=used=='meta',fallback_reason=reason if mode=='meta' else None,
                 validation_status='verified' if used=='meta' and certificate is not None else 'experimental_unvalidated' if used=='meta' else 'not_run',
                 context_key=pre['context_key'],context=pre['context'],certificate=certificate,
                 output_device='cpu',dtype='torch.float32',hidden_gpu_transfers=0,
                 weights_only=False,preflight_seconds=time.perf_counter()-start,file_validation=pre['file_validation'])
    rng_before=_rng_sha(torch);payload_start=time.perf_counter()
    model_data=torch.load(pre['checkpoint_path'],map_location='cpu',weights_only=False)
    regression_data=torch.load(pre['regression_path'],map_location='cpu',weights_only=False)
    receipt['checkpoint_deserialization_seconds']=time.perf_counter()-payload_start
    construct_start=time.perf_counter()
    if used=='meta':
        model,alphabet,proof=_meta_from_payloads(model_data,regression_data)
        receipt['state_proof']=proof
    else:
        model,alphabet=esm.pretrained.load_model_and_alphabet_core(MODEL_NAME,model_data,regression_data)
    receipt['construction_assignment_seconds']=time.perf_counter()-construct_start
    receipt['cpu_rng_before_sha256']=rng_before;receipt['cpu_rng_after_sha256']=_rng_sha(torch)
    receipt['cpu_rng_unchanged']=receipt['cpu_rng_before_sha256']==receipt['cpu_rng_after_sha256']
    receipt['rng_contract']='Meta construction skips disposable initialization; RNG identity with stock is not claimed'
    model._catpred_esm_checkpoint_sha256=dict(pre['context']['weights'])
    receipt['load_seconds']=time.perf_counter()-start
    # Do not change eval/training state; the original caller owns model.eval().
    return model,alphabet,receipt
