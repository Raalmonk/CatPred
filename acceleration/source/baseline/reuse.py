"""Request-local, bounded input reuse for the accepted CatPred Rust path.

No model output or learned state is cached. Original member/batch execution order
and all operations following deterministic input preparation are retained.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import importlib
import inspect
import json
import os
import textwrap

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from catpred.data.data import MoleculeDataset
from catpred.features import featurization

model_module = importlib.import_module("catpred.models.model")
mpn_module = importlib.import_module("catpred.models.mpn")
ORIGINAL_BATCH_GRAPH = MoleculeDataset.batch_graph
ORIGINAL_MODEL_FORWARD = model_module.MoleculeModel.forward
ORIGINAL_MPN_FORWARD = mpn_module.MPNEncoder.forward
_STATE = ContextVar("catpred_request_input_reuse", default=None)
_CONFIG = {"r1": False, "r2": False}
_INSPECTION = None
_MODEL_IDS = set()
_PATCHED_MODEL = None
_PATCHED_MPN = None
_GRAPH_TENSORS = ("f_atoms", "f_bonds", "a2b", "b2a", "b2revb")
_GRAPH_META = (
    "n_atoms", "n_bonds", "max_num_bonds", "a_scope", "b_scope",
    "atom_fdim", "bond_fdim", "overwrite_default_atom_features",
    "overwrite_default_bond_features", "is_reaction",
)
_LETTERS = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
            'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
            'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
            'N': 2, 'Y': 18, 'M': 12}


class ReuseUnsupported(RuntimeError):
    pass


class ReuseBudgetExceeded(ReuseUnsupported):
    pass


def _freeze(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return (str(array.dtype), array.shape, hashlib.sha256(array.tobytes()).hexdigest())
    if isinstance(value, dict):
        return tuple((str(key), _freeze(val)) for key, val in sorted(value.items(), key=lambda x: str(x[0])))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(val) for val in value)
    return repr(value)


def _scaler_info(scaler):
    if scaler is None:
        return None
    fields = dict(vars(scaler))
    content = _freeze(fields)
    return {
        "class": type(scaler).__module__ + "." + type(scaler).__name__,
        "fields": {key: _freeze(value) for key, value in fields.items()},
        "state_sha256": hashlib.sha256(repr(content).encode()).hexdigest(),
    }


def inspect_models(models, scalers):
    """Inspect every scaler; target scaling remains in original predict()."""
    global _INSPECTION, _MODEL_IDS
    if len(models) != len(scalers):
        raise ReuseUnsupported("Model/scaler counts differ")
    members = []
    names = ("target", "features", "atom_descriptor", "bond_descriptor", "atom_bond")
    for index, (model, member_scalers) in enumerate(zip(models, scalers)):
        if len(member_scalers) != 5:
            raise ReuseUnsupported("Expected all five scaler entries for every member")
        scaler_info = {name: _scaler_info(value) for name, value in zip(names, member_scalers)}
        members.append({
            "index": index, "model_id": id(model), "device": str(model.device),
            "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
            "scalers": scaler_info,
            "input_scaler_partition": tuple(None if scaler_info[name] is None else scaler_info[name]["state_sha256"] for name in names[1:4]),
            "flags": {name: getattr(model.args, name, None) for name in (
                "skip_protein", "add_esm_feats", "add_pretrained_egnn_feats", "atom_messages",
                "atom_descriptors", "bond_descriptors", "overwrite_default_atom_features",
                "overwrite_default_bond_features", "is_atom_bond_targets", "number_of_molecules")},
        })
    _MODEL_IDS = {id(model) for model in models}
    _INSPECTION = {
        "members": members, "ensemble_size": len(models),
        "input_scaler_partitions": len({member["input_scaler_partition"] for member in members}),
        "r1_compatible": True,
        "r2_compatible": all(not model.is_atom_bond_targets for model in models),
        "key_policy": "Ordered request datapoint identities, actual normalized atom/bond feature content, overwrite flags, protein/embedding identity and full featurization state",
        "target_scaling": "Original member-specific inverse target and MVE variance scaling untouched",
    }
    return _INSPECTION


def _storage_bytes(tensors):
    seen, total = set(), 0
    for tensor in tensors:
        if tensor is None:
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total


def _version_signature(tensor):
    return (tensor._version, tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype), str(tensor.device))


def _protein_source_key(records, use_esm):
    values = []
    for record in records:
        if use_esm:
            tensor = record["esm2_feats"]
            try:
                version = tensor._version
            except RuntimeError:
                version = None
            identity = (id(tensor), tensor.data_ptr(), version, tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype), str(tensor.device))
        else:
            identity = None
        values.append((record["seq"], identity))
    return tuple(values)


class RequestState:
    def __init__(self, diagnostic=False):
        self.diagnostic = bool(diagnostic)
        self.config = dict(_CONFIG)
        self.counts = Counter()
        self.graphs = {}
        self.graph_snapshots = {}
        self.graph_device = {}
        self.protein_device = {}
        self.tokens = {}
        self.device_versions = []
        self.cpu_bytes = self.device_bytes = 0
        self.cpu_peak = self.device_peak = 0
        ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        self.cpu_limit = min(2 * 1024 ** 3, ram // 10)
        self.device_limit = 0
        self.closed = False
        self.failure = None
        self.graph_entries = self.protein_entries = self.device_graph_entries = 0

    def reserve_cpu(self, byte_count):
        if self.cpu_bytes + byte_count > self.cpu_limit:
            raise ReuseBudgetExceeded(f"CPU input cache would exceed {self.cpu_limit} bytes")
        self.cpu_bytes += byte_count
        self.cpu_peak = max(self.cpu_peak, self.cpu_bytes)

    def reserve_device(self, byte_count, device):
        device = torch.device(device)
        if device.type != "cuda":
            raise ReuseUnsupported("R2 requires the measured CUDA inference device")
        limit = min(16 * 1024 ** 3, torch.cuda.get_device_properties(device).total_memory // 4)
        self.device_limit = limit if not self.device_limit else min(self.device_limit, limit)
        if self.device_bytes + byte_count > self.device_limit:
            raise ReuseBudgetExceeded(
                f"R2 full-request cache needs more than {self.device_limit} bytes; "
                f"retained={self.device_bytes}, next={byte_count}. Candidate excluded; no silent fallback."
            )
        self.device_bytes += byte_count
        self.device_peak = max(self.device_peak, self.device_bytes)

    def remember_versions(self, tensors):
        if self.diagnostic:
            self.device_versions.extend((tensor, _version_signature(tensor)) for tensor in tensors if tensor is not None)

    def check_immutable(self):
        if not self.diagnostic:
            return
        for graph, snapshot in self.graph_snapshots.values():
            for name, expected in snapshot.items():
                actual = getattr(graph, name)
                if torch.is_tensor(expected):
                    assert torch.equal(actual, expected), "Cached graph mutation: " + name
                else:
                    assert _freeze(actual) == expected, "Cached graph metadata mutation: " + name
            self.counts["graph_immutability_checks"] += 1
        for tensor, expected in self.device_versions:
            assert _version_signature(tensor) == expected, "Cached device input mutated"
            self.counts["device_input_immutability_checks"] += 1

    def summary(self):
        return {
            "r1": self.config["r1"], "r2": self.config["r2"], "diagnostic": self.diagnostic,
            "counts": dict(self.counts), "cpu_cache_limit_bytes": self.cpu_limit,
            "device_cache_limit_bytes": self.device_limit, "cpu_cache_peak_bytes": self.cpu_peak,
            "device_cache_peak_bytes": self.device_peak, "cpu_cache_retained_bytes": self.cpu_bytes,
            "device_cache_retained_bytes": self.device_bytes, "cpu_graph_entries": self.graph_entries,
            "device_graph_entries": self.device_graph_entries, "protein_input_entries": self.protein_entries,
            "closed": self.closed, "failure": self.failure,
            "exact_graph_checks_passed": self.diagnostic and self.counts["exact_graph_checks"] > 0 and self.failure is None,
            "exact_protein_input_checks_passed": self.diagnostic and self.counts["exact_protein_input_checks"] > 0 and self.failure is None,
            "policy": "Empty at request start; population inside request; original member-outer order; bounded full-request CPU/device inputs; no cross-request reuse",
        }


@contextmanager
def request_context(diagnostic=False):
    if _STATE.get() is not None:
        raise ReuseUnsupported("Nested reuse requests are unsupported")
    state = RequestState(diagnostic)
    token = _STATE.set(state)
    try:
        yield state
        state.check_immutable()
    except BaseException as error:
        state.failure = type(error).__name__ + ": " + str(error)
        raise
    finally:
        state.graph_entries = len(state.graphs)
        state.device_graph_entries = len(state.graph_device)
        state.protein_entries = len(state.protein_device)
        state.graphs.clear()
        state.graph_snapshots.clear()
        state.graph_device.clear()
        state.protein_device.clear()
        state.tokens.clear()
        state.device_versions.clear()
        state.cpu_bytes = state.device_bytes = 0
        state.closed = True
        _STATE.reset(token)


def _batch_key(dataset):
    points = tuple((
        id(point), tuple(point.smiles),
        _freeze(point.atom_features), _freeze(point.bond_features),
        point.overwrite_default_atom_features, point.overwrite_default_bond_features,
        id(point.protein_record), _freeze(point.ec_features), _freeze(point.tax_features),
    ) for point in dataset._data)
    return (_freeze(vars(featurization.PARAMS)), points)


def _assert_graph_equal(left, right):
    for name in _GRAPH_TENSORS:
        x, y = getattr(left, name), getattr(right, name)
        assert x.shape == y.shape and x.dtype == y.dtype and torch.equal(x, y), "Graph mismatch: " + name
    for name in _GRAPH_META:
        assert getattr(left, name) == getattr(right, name), "Graph metadata mismatch: " + name


def _batch_graph(dataset):
    state = _STATE.get()
    if state is None or not state.config["r1"]:
        return ORIGINAL_BATCH_GRAPH(dataset)
    if dataset._batch_graph is not None:
        return dataset._batch_graph
    key = _batch_key(dataset)
    cached = state.graphs.get(key)
    if cached is not None:
        state.counts["cpu_graph_batch_hits"] += 1
        state.counts["graph_constructions_avoided"] += len(cached)
        if state.diagnostic:
            fresh = ORIGINAL_BATCH_GRAPH(MoleculeDataset(dataset._data))
            for current, expected in zip(cached, fresh):
                _assert_graph_equal(current, expected)
                state.counts["exact_graph_checks"] += 1
            state.counts["diagnostic_reference_constructions"] += len(fresh)
        dataset._batch_graph = cached
        return cached
    packed = ORIGINAL_BATCH_GRAPH(dataset)
    byte_count = sum(_storage_bytes([getattr(graph, name) for name in _GRAPH_TENSORS]) for graph in packed)
    state.reserve_cpu(byte_count)
    state.graphs[key] = packed
    state.counts["cpu_graph_batch_misses"] += 1
    state.counts["graph_constructions"] += len(packed)
    if state.diagnostic:
        reference = ORIGINAL_BATCH_GRAPH(MoleculeDataset(dataset._data))
        state.reserve_cpu(byte_count)
        for graph, expected in zip(packed, reference):
            _assert_graph_equal(graph, expected)
            state.counts["exact_graph_checks"] += 1
            state.graph_snapshots[id(graph)] = (graph, {
                **{name: getattr(graph, name).clone() for name in _GRAPH_TENSORS},
                **{name: _freeze(getattr(graph, name)) for name in _GRAPH_META},
            })
        state.counts["diagnostic_reference_constructions"] += len(reference)
        state.counts["diagnostic_snapshot_bytes"] += byte_count
    return packed


def _graph_components(encoder, graph):
    state = _STATE.get()
    components = graph.get_components(atom_messages=encoder.atom_messages)
    if state is None or not state.config["r2"]:
        return tuple(t.to(encoder.device) for t in components[:5]) + components[5:]
    key = (id(graph), bool(encoder.atom_messages), str(encoder.device))
    if key in state.graph_device:
        state.counts["graph_device_hits"] += 1
        state.counts["graph_h2d_calls_avoided"] += 5
        return state.graph_device[key]
    expected_bytes = sum(t.numel() * t.element_size() for t in components[:5])
    state.reserve_device(expected_bytes, encoder.device)
    # Ordinary tensors track their mutation version even when later consumed
    # under predict()'s inference_mode. This changes no learned operation.
    with torch.inference_mode(False):
        device_tensors = tuple(t.to(encoder.device) for t in components[:5])
    result = device_tensors + components[5:]
    state.graph_device[key] = result
    state.counts["graph_device_misses"] += 1
    state.counts["graph_h2d_calls"] += 5
    state.counts["graph_h2d_bytes"] += expected_bytes
    state.remember_versions(device_tensors)
    if state.diagnostic:
        for expected, actual in zip(components[:5], device_tensors):
            assert expected.dtype == actual.dtype and expected.shape == actual.shape and torch.equal(expected, actual.cpu())
        state.counts["exact_graph_device_checks"] += 1
    return result


def _original_protein_inputs(model, records):
    sequences = [torch.as_tensor([_LETTERS[letter] for letter in record["seq"]], device=model.device, dtype=torch.long) for record in records]
    sequences = pad_sequence(sequences, batch_first=True, padding_value=20).to(model.device)
    esm = None
    if model.args.add_esm_feats:
        esm = pad_sequence([record["esm2_feats"] for record in records], batch_first=True).to(model.device)
        if sequences.shape[1] != esm.shape[1]:
            common_len = min(sequences.shape[1], esm.shape[1])
            sequences, esm = sequences[:, :common_len], esm[:, :common_len]
    return sequences, esm


def _protein_inputs(model, graph, records):
    state = _STATE.get()
    if state is None or not state.config["r2"]:
        return _original_protein_inputs(model, records)
    if _MODEL_IDS and id(model) not in _MODEL_IDS:
        raise ReuseUnsupported("R2 model was not included in scaler/model inspection")
    key = (id(graph), str(model.device), bool(model.args.add_esm_feats), _protein_source_key(records, model.args.add_esm_feats))
    if key in state.protein_device:
        state.counts["protein_device_hits"] += 1
        state.counts["sequence_preparations_avoided"] += len(records)
        state.counts["esm_padding_calls_avoided"] += int(model.args.add_esm_feats)
        state.counts["protein_h2d_calls_avoided"] += 1 + int(model.args.add_esm_feats)
        return state.protein_device[key]
    sequences = []
    for record in records:
        sequence = record["seq"]
        token = state.tokens.get(sequence)
        if token is None:
            token = torch.as_tensor([_LETTERS[letter] for letter in sequence], dtype=torch.long, device="cpu")
            state.reserve_cpu(token.numel() * token.element_size())
            state.tokens[sequence] = token
            state.counts["tokenized_unique_sequences"] += 1
            state.counts["tokenized_residues"] += len(sequence)
        else:
            state.counts["token_cpu_hits"] += 1
        sequences.append(token)
    cpu_sequences = pad_sequence(sequences, batch_first=True, padding_value=20)
    state.counts["sequence_padding_calls"] += 1
    cpu_esm = None
    if model.args.add_esm_feats:
        cpu_esm = pad_sequence([record["esm2_feats"] for record in records], batch_first=True)
        state.counts["esm_padding_calls"] += 1
    inputs = [cpu_sequences] + ([cpu_esm] if cpu_esm is not None else [])
    expected_bytes = sum(tensor.numel() * tensor.element_size() for tensor in inputs)
    state.reserve_device(expected_bytes, model.device)
    with torch.inference_mode(False):
        seq_arr = cpu_sequences.to(model.device)
        esm_arr = cpu_esm.to(model.device) if cpu_esm is not None else None
    if esm_arr is not None and seq_arr.shape[1] != esm_arr.shape[1]:
        common_len = min(seq_arr.shape[1], esm_arr.shape[1])
        seq_arr, esm_arr = seq_arr[:, :common_len], esm_arr[:, :common_len]
    result = (seq_arr, esm_arr)
    state.protein_device[key] = result
    state.counts["protein_device_misses"] += 1
    state.counts["protein_h2d_calls"] += len(inputs)
    state.counts["protein_h2d_bytes"] += expected_bytes
    state.remember_versions(result)
    if state.diagnostic:
        reference = _original_protein_inputs(model, records)
        for expected, actual in zip(reference, result):
            if expected is None:
                assert actual is None
            else:
                assert expected.dtype == actual.dtype and expected.shape == actual.shape and expected.stride() == actual.stride() and torch.equal(expected, actual), "Prepared protein input differs from original"
        state.counts["exact_protein_input_checks"] += 1
        state.counts["diagnostic_reference_protein_preparations"] += 1
    return result


def _compile_patches():
    global _PATCHED_MODEL, _PATCHED_MPN
    if _PATCHED_MODEL is not None:
        return
    source = textwrap.dedent(inspect.getsource(ORIGINAL_MODEL_FORWARD))
    begin = "            seq_arr = [seq_to_tensor(each['seq']) for each in protein_records]"
    end = "            # project sequence to embed dim"
    assert source.count(begin) == source.count(end) == 1, "Pinned protein preparation block changed"
    left, right = source.index(begin), source.index(end)
    replacement = "            seq_arr, esm_feature_arr = _reuse_protein_inputs(self, batch[-1], protein_records)\n\n"
    source = source[:left] + replacement + source[right:]
    namespace = dict(vars(model_module))
    namespace["_reuse_protein_inputs"] = _protein_inputs
    exec(compile(source, "<phase2-reuse-model-forward>", "exec"), namespace)
    _PATCHED_MODEL = namespace["forward"]
    source = textwrap.dedent(inspect.getsource(ORIGINAL_MPN_FORWARD))
    begin = "    f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(atom_messages=self.atom_messages)"
    end = "    f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(self.device), a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device)"
    assert source.count(begin) == source.count(end) == 1, "Pinned MPN preparation block changed"
    left, right = source.index(begin), source.index(end) + len(end)
    source = source[:left] + "    f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = _reuse_graph_components(self, mol_graph)" + source[right:]
    namespace = dict(vars(mpn_module))
    namespace["_reuse_graph_components"] = _graph_components
    exec(compile(source, "<phase2-reuse-mpn-forward>", "exec"), namespace)
    _PATCHED_MPN = namespace["forward"]


def configure(r1=False, r2=False):
    if _STATE.get() is not None:
        raise ReuseUnsupported("Cannot change arm inside a request")
    if r2 and not r1:
        raise ReuseUnsupported("R2 is incremental on R1; configure r1=True, r2=True")
    if r2:
        if _INSPECTION is None:
            raise ReuseUnsupported("Inspect all resident models and scalers before R2")
        if not _INSPECTION["r2_compatible"]:
            raise ReuseUnsupported("R2 experiment excludes atom/bond target models")
        _compile_patches()
    _CONFIG.update(r1=bool(r1), r2=bool(r2))
    MoleculeDataset.batch_graph = _batch_graph if r1 else ORIGINAL_BATCH_GRAPH
    model_module.MoleculeModel.forward = _PATCHED_MODEL if r2 else ORIGINAL_MODEL_FORWARD
    mpn_module.MPNEncoder.forward = _PATCHED_MPN if r2 else ORIGINAL_MPN_FORWARD
    return dict(_CONFIG)
