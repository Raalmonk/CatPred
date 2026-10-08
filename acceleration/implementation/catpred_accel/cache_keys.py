"""Deterministic namespaces; no filesystem cache or tensor evaluation."""
import hashlib
import json


def digest_record(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(name + " must be lowercase SHA-256")
    return value


def feature_key(sequence, *, checkpoint_sha256, feature_code_sha256, tokenizer_sha256,
                backend, software, batch_context, layer=33, dtype="float32", numeric="exact"):
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("Sequence must be a nonempty string")
    if numeric not in ("off", "exact", "fast"):
        raise ValueError("Unknown numerical contract")
    if batch_context is None:
        raise ValueError("Explicit original padding/batch context is required")
    if not backend or not software:
        raise ValueError("Backend and software identities are required")
    record = dict(schema=1, kind="esm2_feature", sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
                  checkpoint_sha256=_sha(checkpoint_sha256,"checkpoint"),
                  feature_code_sha256=_sha(feature_code_sha256,"feature code"),
                  tokenizer_sha256=_sha(tokenizer_sha256,"tokenizer"), layer=layer,
                  dtype=dtype, numeric=numeric, backend=backend, software=software,
                  batch_context=batch_context)
    return digest_record(record)


def compile_key(*, operation, source_sha256, gpu_arch, software, compiler_abi, cpu_features=()):
    if not operation or not compiler_abi or not software:
        raise ValueError("Operation, software and compiler ABI are required")
    return digest_record(dict(schema=1, kind="compiled_operation", operation=operation,
                              source_sha256=_sha(source_sha256,"operation source"),
                              gpu_arch=gpu_arch, software=software, compiler_abi=compiler_abi,
                              cpu_features=sorted(set(cpu_features))))
