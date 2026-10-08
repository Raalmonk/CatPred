"""Inference context identity and strict source/configuration inspection."""
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig
from .cache_keys import digest_record
from .source import SourceMismatch, verify_files

SCIENTIFIC_FILES = {
    "models/model.py":"8cbc6ee75ff2c6526031e856612bf4384d81d3673be49902a8ac3cb708f39676",
    "models/mpn.py":"d1309fdf0c6daf2409a780ed5d85da399778f7067142e6b9099cbbf2f8143313",
    "uncertainty/uncertainty_predictor.py":"87692aabf08a30d8bbc52290026abc72afad2c6bd25339671273e2875509616c",
    "train/predict.py":"0dad566c983ad95a4fc14d912da6ad713fe3ed19007ba8c719b05a83c86e9c2c",
    "data/data.py":"aab48495d5df9ddf8c00535ed05e4a48b8a0ba8ed7dafa34fa9424a7a1cfaaaa",
    "features/featurization.py":"4b37628e1fc59e786371a780f979400e0e33faa4440c50004d6551e25a4d3006",
}


def package_sources():
    root = Path(__file__).resolve().parent
    return {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py"))}


def runtime_settings(torch):
    """Mutable numerical/runtime settings; no distribution or source-file reads."""
    return dict(matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                cudnn_tf32=bool(torch.backends.cudnn.allow_tf32),
                default_dtype=str(torch.get_default_dtype()),default_device=str(torch.get_default_device()),
                matmul_precision=torch.get_float32_matmul_precision(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                cudnn_benchmark=bool(torch.backends.cudnn.benchmark),threads=torch.get_num_threads(),
                interop_threads=torch.get_num_interop_threads())


def software(torch):
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
    return dict(python=platform.python_version(),python_abi=sysconfig.get_config_var("SOABI"),
                torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                numpy=version("numpy"),pandas=version("pandas"),rdkit=version("rdkit"),
                fair_esm=version("fair-esm"),rotary_embedding=version("rotary-embedding-torch"),
                **runtime_settings(torch))


def verify_scientific_sources():
    module=importlib.import_module("catpred.models.model")
    root=Path(module.__file__).resolve().parents[1]
    return verify_files(root,SCIENTIFIC_FILES)


def verify_components(components):
    root=Path(__file__).resolve().parent
    expected=json.loads((root/"accepted_sources.json").read_text())
    if set(components)!={"packing","reuse","probes","streaming"}:
        raise SourceMismatch("Supply exactly packing/reuse/probes/streaming modules")
    for name,module in components.items():
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=expected[name+".py"]:
            raise SourceMismatch("Accepted helper identity mismatch: "+name)
    return expected


def freeze(value):
    import numpy as np
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    if isinstance(value,np.ndarray):
        arr=np.ascontiguousarray(value)
        return dict(dtype=arr.dtype.str,shape=list(arr.shape),sha256=hashlib.sha256(arr.tobytes()).hexdigest())
    if isinstance(value,dict):
        return {str(k):freeze(v) for k,v in sorted(value.items(),key=lambda x:str(x[0]))}
    if isinstance(value,(tuple,list)):
        return [freeze(v) for v in value]
    return repr(value)


def model_signature(models,scalers,torch):
    if len(models)!=10 or len(scalers)!=10:
        raise SourceMismatch("Production kcat requires all ten members and scaler lists")
    flags_names=("skip_protein","add_esm_feats","add_pretrained_egnn_feats","atom_messages",
                 "number_of_molecules","atom_descriptors","bond_descriptors","overwrite_default_atom_features",
                 "overwrite_default_bond_features","reaction","reaction_solvent")
    if torch.is_autocast_enabled():
        raise SourceMismatch("Autocast is outside the unchanged FP32 contract")
    signatures=[]
    for model,member_scalers in zip(models,scalers):
        args=model.args
        if len(member_scalers)!=5 or any(s is not None for s in member_scalers[1:]):
            raise SourceMismatch("Accelerated sharing rejects non-null input/atomic scalers")
        if (model.is_atom_bond_targets or model.classification or model.multiclass or model.loss_function!="mve"
                or args.skip_protein or not args.add_esm_feats or args.add_pretrained_egnn_feats or args.atom_messages
                or args.number_of_molecules!=1 or args.atom_descriptors or args.bond_descriptors):
            raise SourceMismatch("Unsupported kcat model configuration")
        params=list(model.named_parameters())
        if not params or {p.dtype for _,p in params}!={torch.float32}:
            raise SourceMismatch("Unchanged FP32 model parameters required")
        if {str(p.device) for _,p in params}!={str(model.device)}:
            # torch.device('cuda') and actual cuda:0 refer to the same effective
            # device; compare normalized parameter device against model index.
            declared=torch.device(model.device)
            devices={str(p.device) for _,p in params}
            if not (declared.type=="cuda" and declared.index is None and len(devices)==1 and next(iter(devices)).startswith("cuda:")):
                raise SourceMismatch("Model parameters and declared device differ")
        signatures.append(dict(flags={name:freeze(getattr(args,name,None)) for name in flags_names},
            parameters=[dict(name=n,shape=list(p.shape),stride=list(p.stride()),dtype=str(p.dtype)) for n,p in params],
            scalers=[None if s is None else dict(type=type(s).__module__+"."+type(s).__name__,state=freeze(vars(s))) for s in member_scalers]))
    if len({digest_record(s["flags"]) for s in signatures})!=1:
        raise SourceMismatch("Member input flags differ")
    if len({str(next(m.parameters()).device) for m in models})!=1:
        raise SourceMismatch("Members must share one device")
    return signatures


def live_signature(models,scalers):
    parameters=[]
    for model in models:
        parameters.append([(id(p),p._version,p.data_ptr(),str(p.device),str(p.dtype),tuple(p.shape),tuple(p.stride())) for p in model.parameters()])
    flags=[{name:freeze(getattr(m.args,name,None)) for name in (
        "skip_protein","add_esm_feats","add_pretrained_egnn_feats","atom_messages","number_of_molecules",
        "atom_descriptors","bond_descriptors","overwrite_default_atom_features","overwrite_default_bond_features",
        "reaction","reaction_solvent")} for m in models]
    return (parameters,flags,freeze([[None if s is None else vars(s) for s in member] for member in scalers]))
