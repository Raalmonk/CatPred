"""Colab-only, request-scoped R3/T1/T2 probes for the pinned CatPred kcat path.

No learned values survive a model forward (T1) or a member prediction (R3).
inspect_models() composes with reuse.py without replacing MoleculeModel.forward.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import textwrap
from types import MethodType

import numpy as np
import torch
from tqdm import tqdm


class UnsupportedProbe(RuntimeError):
    """Reject a configuration outside this experimentally checked path."""


# A pending list and its optional contiguous concatenation jointly fit this cap.
OUTPUT_DEVICE_STORAGE_CAP_BYTES = 1 << 20
_PENDING_CAP_BYTES = OUTPUT_DEVICE_STORAGE_CAP_BYTES // 2
_CONFIG = {"r3": False, "t1": False, "t2": False}
_ACTIVE = ContextVar("catpred_phase2_probe_request", default=None)
_MODELS = {}
_T1_APPROVED = set()
_ORIGINAL_PREDICT = None
_COUNTED_PREDICT = None
_ORIGINAL_UNCERTAINTY_PREDICT = None
_PREDICT_SOURCE_SHA256 = None
_ORIGINAL_PREDICT_MODULE = None
_INSTALLED = False


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:  # inference-mode tensors intentionally lack counters
        return None


def _nbytes(tensor):
    return tensor.numel() * tensor.element_size()


def _simple(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, (list, tuple)) and all(
        v is None or isinstance(v, (str, bool, int, float)) for v in value
    ):
        return list(value)
    return None


@dataclass
class RequestState:
    config: dict
    diagnostic: bool = False
    counters: Counter = field(default_factory=Counter)
    member_counters: dict = field(default_factory=dict)
    pending_rotary: dict = field(default_factory=dict)
    forward_slots: dict = field(default_factory=dict)
    raw_diagnostic_checks: list = field(default_factory=list)
    rotary_diagnostic_checks: list = field(default_factory=list)
    failures: list = field(default_factory=list)

    def increment(self, name, value=1, member=None):
        self.counters[name] += int(value)
        if member is not None:
            self.member_counters.setdefault(str(member), Counter())[name] += int(value)

    def maximum(self, name, value):
        self.counters[name] = max(self.counters[name], int(value))

    def summary(self):
        return {
            "configuration": dict(self.config),
            "diagnostic": self.diagnostic,
            "counters": dict(self.counters),
            "member_counters": {k: dict(v) for k, v in self.member_counters.items()},
            "raw_deferred_output_checks": list(self.raw_diagnostic_checks),
            "rotary_checks": list(self.rotary_diagnostic_checks),
            "failures": list(self.failures),
            "output_device_storage_cap_bytes": OUTPUT_DEVICE_STORAGE_CAP_BYTES,
            "output_transfer_boundary": "Actual CUDA raw-output .cpu() invocations and tensor bytes; diagnostic verification copies counted separately.",
            "attention_backend": "Selected backend is determined by recorded profiler operators, not inferred from need_weights.",
            "request_cache_empty_on_entry": True,
            "no_learned_state_retained_after_forward_or_member": True,
        }


@contextmanager
def request_context(diagnostic=False):
    if _ACTIVE.get() is not None:
        raise RuntimeError("Nested probe request contexts are unsupported")
    state = RequestState(dict(_CONFIG), bool(diagnostic))
    token = _ACTIVE.set(state)
    try:
        yield state
    except BaseException as error:
        state.failures.append({"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        state.pending_rotary.clear()
        state.forward_slots.clear()
        _ACTIVE.reset(token)


def _cpu_numpy(tensor, diagnostic=False, member=None):
    """The original .data.cpu().numpy() expression with exact transfer counts."""
    state = _ACTIVE.get()
    raw = tensor.data
    result = raw.cpu().numpy()
    if state is not None:
        kind = "diagnostic_output_d2h" if diagnostic else "output_d2h"
        state.increment(kind + "_calls", int(raw.device.type == "cuda"), member)
        state.increment(kind + "_bytes", _nbytes(raw) if raw.device.type == "cuda" else 0, member)
        state.increment("raw_output_numpy_views", 1, member)
    return result


def _install_predict_dispatch():
    global _INSTALLED, _ORIGINAL_PREDICT, _COUNTED_PREDICT
    global _ORIGINAL_UNCERTAINTY_PREDICT, _PREDICT_SOURCE_SHA256, _ORIGINAL_PREDICT_MODULE
    if _INSTALLED:
        return
    module = importlib.import_module("catpred.train.predict")
    uncertainty = importlib.import_module("catpred.uncertainty.uncertainty_predictor")
    _ORIGINAL_PREDICT_MODULE = module
    _ORIGINAL_PREDICT = module.predict
    _ORIGINAL_UNCERTAINTY_PREDICT = uncertainty.predict
    if _ORIGINAL_UNCERTAINTY_PREDICT is not _ORIGINAL_PREDICT:
        raise UnsupportedProbe("predict aliases already differ; refusing to overwrite an unknown predictor")
    source = textwrap.dedent(inspect.getsource(_ORIGINAL_PREDICT))
    _PREDICT_SOURCE_SHA256 = hashlib.sha256(source.encode()).hexdigest()
    needle = "            batch_preds = batch_preds.data.cpu().numpy()"
    if source.count(needle) != 1:
        raise UnsupportedProbe("Pinned non-atomic prediction transfer site did not match")
    # Everything else, including dtype promotion/scalers/NumPy ordering, remains
    # exactly the original predictor source in the disabled-R3 control.
    counted_source = source.replace(
        needle,
        "            batch_preds = _phase2_cpu_numpy(batch_preds, member=_phase2_member_index(model))",
    )
    namespace = dict(_ORIGINAL_PREDICT.__globals__)
    namespace.update(_phase2_cpu_numpy=_cpu_numpy, _phase2_member_index=_member_index)
    exec(compile(counted_source, "<phase2_counted_original_predict>", "exec"), namespace)
    _COUNTED_PREDICT = namespace["predict"]
    module.predict = _predict_dispatch
    uncertainty.predict = _predict_dispatch
    _INSTALLED = True


def configure(r3=False, t1=False, t2=False):
    if _ACTIVE.get() is not None:
        raise RuntimeError("Cannot reconfigure a running request")
    _install_predict_dispatch()
    _CONFIG.update(r3=bool(r3), t1=bool(t1), t2=bool(t2))
    return dict(_CONFIG)


def _member_index(model):
    entry = _MODELS.get(id(model))
    return entry["index"] if entry is not None else "unregistered"


def _predict_dispatch(model, data_loader, disable_progress_bar=False, scaler=None,
                      atom_bond_scaler=None, return_unc_parameters=False, dropout_prob=0.0):
    state = _ACTIVE.get()
    kwargs = dict(model=model, data_loader=data_loader,
                  disable_progress_bar=disable_progress_bar, scaler=scaler,
                  atom_bond_scaler=atom_bond_scaler,
                  return_unc_parameters=return_unc_parameters, dropout_prob=dropout_prob)
    if state is None:
        return _ORIGINAL_PREDICT(**kwargs)
    member = _member_index(model)
    state.increment("member_predict_calls", 1, member)
    if not state.config["r3"]:
        return _COUNTED_PREDICT(**kwargs)
    return _deferred_predict(state=state, member=member, **kwargs)


def _deferred_predict(model, data_loader, disable_progress_bar=False, scaler=None,
                      atom_bond_scaler=None, return_unc_parameters=False,
                      dropout_prob=0.0, *, state, member):
    if id(model) not in _MODELS:
        raise UnsupportedProbe("Call inspect_models with the resident ensemble before R3")
    if (model.is_atom_bond_targets or model.loss_function != "mve"
            or model.classification or model.multiclass or dropout_prob != 0
            or atom_bond_scaler is not None):
        raise UnsupportedProbe("R3 supports non-atomic regression MVE with dropout disabled and no atom/bond target scaler")
    model.eval()
    preds, var = [], []
    pending, pending_sizes, expected_raw = [], [], []
    pending_bytes = 0

    def drain():
        nonlocal pending_bytes
        if not pending:
            return
        # Concatenation copies only FP32 values; it performs no arithmetic.
        joined = torch.cat(pending, dim=0) if len(pending) > 1 else pending[0]
        owned = pending_bytes + (_nbytes(joined) if len(pending) > 1 else 0)
        if owned > OUTPUT_DEVICE_STORAGE_CAP_BYTES:
            raise UnsupportedProbe("R3 output buffer budget exceeded")
        state.maximum("r3_peak_owned_output_bytes", owned)
        host = _cpu_numpy(joined, member=member)
        offset = 0
        for batch_index, size in enumerate(pending_sizes):
            batch_preds = host[offset:offset + size]
            offset += size
            if state.diagnostic:
                expected = expected_raw[batch_index]
                exact = (batch_preds.shape == expected.shape
                         and batch_preds.dtype == expected.dtype
                         and np.array_equal(batch_preds, expected))
                state.raw_diagnostic_checks.append({
                    "member": str(member), "rows": size, "shape": list(batch_preds.shape),
                    "dtype": str(batch_preds.dtype), "exact": bool(exact),
                })
                if not exact:
                    raise AssertionError("R3 raw deferred copy differs from the original per-batch copy")
            # These operations are the original per-batch MVE CPU path, in the
            # original order. In particular StandardScaler casts means to float64
            # and variance multiplication follows its original NumPy promotion.
            batch_preds, batch_var = np.split(batch_preds, 2, axis=1)
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)
                batch_var = batch_var * scaler.stds ** 2
            preds.extend(batch_preds.tolist())
            var.extend(batch_var.tolist())
            state.increment("r3_original_numpy_batches", 1, member)
        state.increment("r3_host_flushes", 1, member)
        pending.clear()
        pending_sizes.clear()
        expected_raw.clear()
        pending_bytes = 0

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        # Keep original batch construction and per-member feature access.
        mol_batch = batch.batch_graph()
        features_batch = batch.features()
        atom_descriptors_batch = batch.atom_descriptors()
        atom_features_batch = batch.atom_features()
        bond_descriptors_batch = batch.bond_descriptors()
        bond_features_batch = batch.bond_features()
        constraints_batch = batch.constraints()
        with _ORIGINAL_PREDICT_MODULE._inference_context():
            raw = model(mol_batch, features_batch, atom_descriptors_batch,
                        atom_features_batch, bond_descriptors_batch, bond_features_batch,
                        constraints_batch, None)
        if (not torch.is_tensor(raw) or raw.ndim != 2 or raw.shape[1] != 2
                or raw.dtype != torch.float32 or raw.device.type != "cuda"):
            raise UnsupportedProbe("R3 requires CUDA FP32 [batch_rows,2] raw single-target MVE output")
        raw = raw.data
        size_bytes = _nbytes(raw)
        if size_bytes > _PENDING_CAP_BYTES:
            raise UnsupportedProbe("One raw batch exceeds R3 bounded storage")
        if pending_bytes + size_bytes > _PENDING_CAP_BYTES:
            drain()
        pending.append(raw)
        pending_sizes.append(raw.shape[0])
        pending_bytes += size_bytes
        state.maximum("r3_peak_pending_output_bytes", pending_bytes)
        state.maximum("r3_peak_pending_batches", len(pending))
        state.increment("r3_device_output_batches", 1, member)
        if state.diagnostic:
            expected_raw.append(_cpu_numpy(raw, diagnostic=True, member=member).copy())
    drain()  # All required device work/copies finish before NumPy aggregation.
    return (preds, var) if return_unc_parameters else preds


def _tensor_key(tensor):
    return (id(tensor), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
            str(tensor.dtype), str(tensor.device), _version(tensor))


def _argument_key(args, kwargs):
    def key(value):
        if torch.is_tensor(value):
            return ("tensor", _tensor_key(value))
        if value is None or isinstance(value, (bool, int, float, str)):
            return ("value", value)
        return ("unsupported", type(value).__qualname__, id(value))
    return tuple(key(x) for x in args), tuple(sorted((k, key(v)) for k, v in kwargs.items()))


def _geometry_key(model_id, tensor, args, kwargs):
    return (model_id, tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype),
            str(tensor.device), repr(_argument_key(args, kwargs)))


def _rotary_snapshot(module):
    tensors = {}
    for prefix, values in (("parameter", module.named_parameters()), ("buffer", module.named_buffers())):
        for name, value in values:
            tensors[(prefix, name)] = (_tensor_key(value), value.detach().clone())
    scalar_values = {name: _simple(value) for name, value in vars(module).items()
                     if value is None or isinstance(value, (str, bool, int, float, torch.dtype, torch.device))}
    return tensors, scalar_values


def _assert_rotary_unchanged(module, snapshot):
    old_tensors, old_scalars = snapshot
    current = {}
    for prefix, values in (("parameter", module.named_parameters()), ("buffer", module.named_buffers())):
        for name, value in values:
            current[(prefix, name)] = value
    if set(old_tensors) != set(current):
        raise AssertionError("T1 second rotary call changed parameter/buffer topology")
    for name, (key, values) in old_tensors.items():
        tensor = current[name]
        if _tensor_key(tensor) != key or not torch.equal(tensor, values):
            raise AssertionError("T1 second rotary call mutated a parameter or buffer: " + str(name))
    scalar_values = {name: _simple(value) for name, value in vars(module).items()
                     if value is None or isinstance(value, (str, bool, int, float, torch.dtype, torch.device))}
    if scalar_values != old_scalars:
        raise AssertionError("T1 second rotary call changed module scalar state")


def _install_attention_wrappers(model, index):
    model_id = id(model)
    rotary, attention = model.rotary_embedder, model.multihead_attn
    original_rotate = rotary.rotate_queries_or_keys
    original_attention = attention.forward
    rotate_source = inspect.getsource(original_rotate)
    rotate_sha = hashlib.sha256(rotate_source.encode()).hexdigest()

    def begin_forward(module, args):
        state = _ACTIVE.get()
        if state is None:
            return
        state.increment("model_forward_calls", 1, index)
        if state.config["t1"]:
            state.pending_rotary.pop(model_id, None)
            state.forward_slots[model_id] = {"calls": 0, "paired": False}

    def end_forward(module, args, output):
        state = _ACTIVE.get()
        if state is None:
            return
        slot = state.forward_slots.pop(model_id, None)
        state.pending_rotary.pop(model_id, None)
        if state.config["t1"] and output is not None and (slot is None or slot["calls"] != 2 or not slot["paired"]):
            raise UnsupportedProbe("T1 expected exactly one immediate identical rotary q/k pair in each model forward")

    def rotate_wrapper(_module, tensor, *args, **kwargs):
        state = _ACTIVE.get()
        if state is None:
            return original_rotate(tensor, *args, **kwargs)
        state.increment("rotary_api_calls", 1, index)
        if not state.config["t1"]:
            state.increment("rotary_computations", 1, index)
            return original_rotate(tensor, *args, **kwargs)
        slot = state.forward_slots.get(model_id)
        if slot is None:
            raise UnsupportedProbe("T1 rotary call occurred outside a registered model forward")
        slot["calls"] += 1
        if slot["calls"] == 1:
            result = original_rotate(tensor, *args, **kwargs)
            state.increment("rotary_computations", 1, index)
            state.pending_rotary[model_id] = {
                "input": tensor, "input_key": _tensor_key(tensor), "arguments": _argument_key(args, kwargs),
                "output": result,
            }
            return result
        previous = state.pending_rotary.pop(model_id, None)
        if (slot["calls"] != 2 or previous is None or previous["input"] is not tensor
                or previous["input_key"] != _tensor_key(tensor)
                or previous["arguments"] != _argument_key(args, kwargs)):
            raise UnsupportedProbe("T1 rotary q/k inputs or arguments are not identical and adjacent")
        geometry = _geometry_key(model_id, tensor, args, kwargs)
        if state.diagnostic:
            snapshot = _rotary_snapshot(rotary)
            input_copy = tensor.detach().clone()
            reference = original_rotate(tensor, *args, **kwargs)
            state.increment("diagnostic_rotary_computations", 1, index)
            _assert_rotary_unchanged(rotary, snapshot)
            exact = (reference.dtype == previous["output"].dtype
                     and reference.shape == previous["output"].shape
                     and torch.equal(reference, previous["output"])
                     and torch.equal(tensor, input_copy))
            if not exact:
                raise AssertionError("T1 second rotary call differs or mutates its input")
            _T1_APPROVED.add(geometry)
            state.rotary_diagnostic_checks.append({
                "member": index, "input_shape": list(tensor.shape),
                "dtype": str(tensor.dtype), "arguments": repr(_argument_key(args, kwargs)),
                "outputs_exact": True, "input_unchanged": True,
                "parameters_buffers_scalars_unchanged": True, "rotate_source_sha256": rotate_sha,
            })
        elif geometry not in _T1_APPROVED:
            raise UnsupportedProbe("T1 geometry lacks a diagnostic equivalence/side-effect check; run diagnostic=True first")
        slot["paired"] = True
        state.increment("rotary_reused_second_calls", 1, index)
        return previous["output"]

    def attention_wrapper(_module, *args, **kwargs):
        state = _ACTIVE.get()
        if state is not None:
            state.increment("mha_forward_calls", 1, index)
            if state.config["t2"]:
                args = list(args)
                kwargs = dict(kwargs)
                if len(args) >= 5:
                    args[4] = False
                    kwargs.pop("need_weights", None)
                else:
                    kwargs["need_weights"] = False
                state.increment("mha_need_weights_false_calls", 1, index)
        return original_attention(*args, **kwargs)

    rotary.rotate_queries_or_keys = MethodType(rotate_wrapper, rotary)
    attention.forward = MethodType(attention_wrapper, attention)
    pre_handle = model.register_forward_pre_hook(begin_forward)
    post_handle = model.register_forward_hook(end_forward, always_call=True)
    return {"index": index, "model": model, "rotary": rotary, "attention": attention,
            "original_rotate": original_rotate, "original_attention": original_attention,
            "hooks": [pre_handle, post_handle], "rotary_source_sha256": rotate_sha,
            "rotary_source": rotate_source}


def _array_description(value):
    if value is None:
        return None
    array = np.asarray(value)
    if array.dtype.hasobject:
        encoded = repr(value).encode()
    else:
        encoded = np.ascontiguousarray(array).tobytes()
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "values": array.tolist() if array.size <= 32 else None}


def inspect_models(models, scalers=None):
    """Inspect all scalers and install composable per-instance attention hooks.

    Call once after all ten resident models are loaded and before any probe.
    Repeated calls for exactly the same models preserve their installed hooks.
    """
    _install_predict_dispatch()
    models = list(models)
    if len(models) != 10:
        raise UnsupportedProbe("The probes require all ten production kcat members")
    if scalers is not None:
        scalers = list(scalers)
        if len(scalers) != len(models):
            raise UnsupportedProbe("Scaler/member counts disagree")
    report = {"models": [], "scalers": [], "prediction_source_sha256": _PREDICT_SOURCE_SHA256,
              "r3_device_storage_cap_bytes": OUTPUT_DEVICE_STORAGE_CAP_BYTES,
              "rotary_sources": {}}
    for index, model in enumerate(models):
        if (model.is_atom_bond_targets or model.classification or model.multiclass
                or model.loss_function != "mve" or getattr(model.args, "skip_protein", False)
                or not getattr(model.args, "add_esm_feats", False)
                or getattr(model.args, "add_pretrained_egnn_feats", False)):
            raise UnsupportedProbe("Probes require the pinned protein+ESM, non-atomic kcat MVE path")
        if {p.dtype for p in model.parameters()} != {torch.float32}:
            raise UnsupportedProbe("Model parameters must all retain FP32")
        forward_file = Path(inspect.getfile(type(model)))
        source = forward_file.read_text()
        fragments = ["q = self.rotary_embedder.rotate_queries_or_keys(seq_outs,",
                     "k = self.rotary_embedder.rotate_queries_or_keys(seq_outs,",
                     "seq_outs, _ = self.multihead_attn(q, k, seq_outs)"]
        if not all(fragment in source for fragment in fragments):
            raise UnsupportedProbe("Pinned attention sites did not match inspected model source")
        existing = _MODELS.get(id(model))
        if existing is not None and existing["index"] != index:
            raise UnsupportedProbe("Ensemble model order changed")
        if existing is None:
            _MODELS[id(model)] = _install_attention_wrappers(model, index)
        entry = _MODELS[id(model)]
        report["rotary_sources"][entry["rotary_source_sha256"]] = entry["rotary_source"]
        report["models"].append({
            "index": index, "class": type(model).__qualname__, "dtype": "torch.float32",
            "device": str(next(model.parameters()).device), "loss_function": model.loss_function,
            "model_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "rotary_source_sha256": entry["rotary_source_sha256"],
            "attention_class": type(model.multihead_attn).__qualname__,
            "separate_learned_qkv_projections_preserved": True,
        })
        if scalers is not None:
            scaler_list = list(scalers[index])
            if len(scaler_list) != 5:
                raise UnsupportedProbe("Expected five scaler slots per production member")
            names = ["target", "features", "atom_descriptor", "bond_descriptor", "atom_bond_target"]
            item = {"member": index, "slots": {}}
            for name, scaler in zip(names, scaler_list):
                item["slots"][name] = None if scaler is None else {
                    "class": type(scaler).__qualname__, "means": _array_description(getattr(scaler, "means", None)),
                    "stds": _array_description(getattr(scaler, "stds", None)),
                    "replace_nan_token": _simple(getattr(scaler, "replace_nan_token", None)),
                    "inverse_transform_source_sha256": hashlib.sha256(inspect.getsource(type(scaler).inverse_transform).encode()).hexdigest(),
                }
            report["scalers"].append(item)
    return report
