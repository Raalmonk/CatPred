"""Bounded, exact-input chunk scheduling for the pinned ten-member kcat MVE path.

Only original whole batches are grouped. The accepted Phase 2 input preparation
and per-batch NumPy postprocessing are reused; learned computation is unchanged.
"""
from __future__ import annotations

import ast
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import importlib
import inspect
import os
from pathlib import Path
import textwrap
import weakref

import torch

import probes
import reuse

GIB = 1 << 30
_CONFIG = {"enabled": False, "budget_bytes": 2 * GIB}
_ACTIVE = ContextVar("phase3_stream_request", default=None)
_HOST_RESULTS = ContextVar("phase3_stream_host_outputs", default=None)
_UNCERTAINTY = importlib.import_module("catpred.uncertainty.uncertainty_predictor")
_ORIGINAL_CALCULATE = _UNCERTAINTY.MVEPredictor.calculate_predictions
_HOST_AGGREGATE = None
_INSPECTION = None
_SOURCE_PROOF = None
_HOOKED_MODELS = {}
_PINNED_FILES = {
    "reuse": "594f14ac87387b0b8fc22df1692c4c25c7510f24cce905f748c54168c98763c0",
    "probes": "2588f38d4fc381f80b887f3c6fb231e25757d36944dd78aed1baccc8e879e088",
}
_PINNED_AST = {
    "model_forward": "c1eca4ea8d690f472d3f6322cd8d748a0ed442b146075ea83b0b88e301a7b541",
    "mpn_forward": "7070c4ea64792fd8f426a12e956e1f774d8c26e43163159594b98bd62a979d03",
    "mve_calculate": "292c8e621548d943ac82b1f0291c8a3e0aa56a38e8b9598e935de33ea812ed1a",
}


class StreamUnsupported(reuse.ReuseUnsupported):
    pass


class StreamBudgetExceeded(reuse.ReuseBudgetExceeded):
    pass


def _unique_storage_bytes(tensors):
    """Count complete underlying allocations once, including retained views."""
    seen = set()
    total = 0
    for tensor in tensors:
        if tensor is None:
            continue
        storage = tensor.untyped_storage()
        identity = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if identity not in seen:
            total += storage.nbytes()
            seen.add(identity)
    return total


def _host_member_prediction(model, **kwargs):
    results = _HOST_RESULTS.get()
    if results is None or id(model) not in results:
        raise StreamUnsupported("Missing complete per-member CPU outputs")
    if not kwargs.get("return_unc_parameters"):
        raise StreamUnsupported("Expected original MVE mean/variance aggregation")
    return results[id(model)]


def verify_sources():
    """Verify legacy text overlays before they are enabled, and AST-edit one call."""
    global _SOURCE_PROOF, _HOST_AGGREGATE
    if _SOURCE_PROOF is not None:
        return dict(_SOURCE_PROOF)
    proof = {"files": {}, "ast": {}, "aggregation_predict_calls_replaced": 0}
    for name, module in (("reuse", reuse), ("probes", probes)):
        actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if actual != _PINNED_FILES[name]:
            raise StreamUnsupported("Accepted Phase 2 source changed: " + name)
        proof["files"][name] = actual
    functions = {
        "model_forward": reuse.ORIGINAL_MODEL_FORWARD,
        "mpn_forward": reuse.ORIGINAL_MPN_FORWARD,
        "mve_calculate": _ORIGINAL_CALCULATE,
    }
    trees = {}
    for name, function in functions.items():
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        node = tree.body[0]
        digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        if digest != _PINNED_AST[name]:
            raise StreamUnsupported("Pinned function AST changed: " + name)
        proof["ast"][name] = digest
        trees[name] = tree

    class ReplacePredict(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "predict":
                node.func = ast.copy_location(ast.Name(id="_stream_host_member_prediction", ctx=ast.Load()), node.func)
                proof["aggregation_predict_calls_replaced"] += 1
            return node

    tree = ReplacePredict().visit(trees["mve_calculate"])
    if proof["aggregation_predict_calls_replaced"] != 1:
        raise StreamUnsupported("Expected exactly one original MVE predict call")
    ast.fix_missing_locations(tree)
    namespace = dict(_ORIGINAL_CALCULATE.__globals__)
    namespace["_stream_host_member_prediction"] = _host_member_prediction
    exec(compile(tree, "<phase3-original-mve-aggregation>", "exec"), namespace)
    _HOST_AGGREGATE = namespace["calculate_predictions"]
    proof["aggregation_policy"] = "Original AST with only predict() replaced by complete CPU per-member outputs"
    _SOURCE_PROOF = proof
    return dict(proof)


def inspect_models(models, scalers):
    """Fail closed on input normalization or feature configurations not tested."""
    global _INSPECTION
    verify_sources()
    models, scalers = list(models), list(scalers)
    if len(models) != 10 or len(scalers) != 10:
        raise StreamUnsupported("All ten production members and scaler lists are required")
    members = []
    signatures = set()
    for index, (model, member_scalers) in enumerate(zip(models, scalers)):
        if len(member_scalers) != 5 or any(s is not None for s in member_scalers[1:]):
            raise StreamUnsupported("S requires verified null input and atom/bond target scalers; no unsafe shared normalization")
        args = model.args
        if (model.is_atom_bond_targets or model.classification or model.multiclass
                or model.loss_function != "mve" or args.skip_protein
                or not args.add_esm_feats or args.add_pretrained_egnn_feats
                or args.atom_messages or args.number_of_molecules != 1
                or args.atom_descriptors or args.bond_descriptors):
            raise StreamUnsupported("S supports the inspected production single-molecule FP32 kcat MVE configuration")
        if {p.dtype for p in model.parameters()} != {torch.float32}:
            raise StreamUnsupported("S requires unchanged FP32 parameters")
        flags = {name: getattr(args, name, None) for name in (
            "atom_messages", "number_of_molecules", "add_esm_feats", "skip_protein",
            "overwrite_default_atom_features", "overwrite_default_bond_features",
            "atom_descriptors", "bond_descriptors", "reaction", "reaction_solvent")}
        signature = (str(model.device), repr(sorted(flags.items())))
        signatures.add(signature)
        members.append({"index": index, "model_id": id(model), "device": str(model.device),
                        "flags": flags, "scalers": [reuse._scaler_info(s) for s in member_scalers]})
        if id(model) not in _HOOKED_MODELS:
            def count_forward(module, inputs, member=index):
                active = _ACTIVE.get()
                if active is not None and active.config["enabled"]:
                    graphs = inputs[0]
                    if not isinstance(graphs, list) or len(graphs) != 1 or id(graphs[0]) not in active.current_graph_rows:
                        raise StreamUnsupported("Forward received a graph outside the current original chunk")
                    row_info = active.current_graph_rows[id(graphs[0])]
                    if row_info["batch_index"] != active.forward_counts[member]:
                        raise StreamUnsupported("A member's original batch order changed")
                    active.member_batch_order[member].append(dict(row_info))
                    active.forward_counts[member] += 1

            def check_raw_storage(module, inputs, output):
                active = _ACTIVE.get()
                if active is None or not active.config["enabled"] or output is None:
                    return
                if (not torch.is_tensor(output) or output.ndim != 2 or output.shape[1] != 2
                        or output.dtype != torch.float32 or not output.is_contiguous()
                        or output.storage_offset() != 0
                        or output.untyped_storage().nbytes() != output.numel() * output.element_size()
                        or output.untyped_storage().nbytes() > probes._PENDING_CAP_BYTES):
                    raise StreamUnsupported("Raw MVE output storage violates the accepted bounded R3 contract")
                active.counts["actual_raw_storage_checks"] += 1
            _HOOKED_MODELS[id(model)] = [model.register_forward_pre_hook(count_forward),
                                        model.register_forward_hook(check_raw_storage, always_call=True)]
    if len(signatures) != 1:
        raise StreamUnsupported("Member input configurations differ; unpartitioned sharing rejected")
    _INSPECTION = {"members": members, "source_proof": verify_sources(),
                   "input_scaler_policy": "All five scaler slots inspected per member; target scaler preserved, other slots must be null"}
    return _INSPECTION


def configure(enabled=False, budget_bytes=2 * GIB):
    if _ACTIVE.get() is not None:
        raise StreamUnsupported("Cannot change S configuration inside a request")
    if enabled:
        if _INSPECTION is None:
            raise StreamUnsupported("Call streaming.inspect_models before enabling S")
        if budget_bytes not in (GIB, 2 * GIB, 4 * GIB):
            raise StreamUnsupported("Only the prespecified 1/2/4 GiB input budgets are supported")
        verify_sources()
    _CONFIG.update(enabled=bool(enabled), budget_bytes=int(budget_bytes))
    _UNCERTAINTY.MVEPredictor.calculate_predictions = _calculate_stream if enabled else _ORIGINAL_CALCULATE
    return dict(_CONFIG)


class StreamState:
    def __init__(self, diagnostic=False):
        self.config = dict(_CONFIG)
        self.diagnostic = bool(diagnostic)
        self.counts = Counter()
        self.chunks = []
        self.device_peak = self.cpu_peak = 0
        self.forward_counts = [0] * 10
        self.closed = False
        self.failure = None
        self.active_chunk = None
        self.retained = 0
        self.final_cpu_output_bytes = 0
        self.output_peak_owned_bytes = 0
        self.current_graph_rows = {}
        self.member_batch_order = [[] for _ in range(10)]
        self.cpu_padding_transient_peak = 0
        self.diagnostic_budget_rejection = None

    def summary(self):
        return {"enabled": self.config["enabled"], "budget_bytes": self.config["budget_bytes"],
                "diagnostic": self.diagnostic, "chunk_count": len(self.chunks), "chunks": self.chunks,
                "device_cache_peak_bytes": self.device_peak, "cpu_cache_peak_bytes": self.cpu_peak,
                "device_cache_retained_bytes": self.active_chunk.device_bytes if self.active_chunk is not None else 0,
                "cpu_cache_retained_bytes": self.active_chunk.cpu_bytes if self.active_chunk is not None else 0,
                "output_peak_owned_bytes": self.output_peak_owned_bytes,
                "output_device_storage_cap_bytes": probes.OUTPUT_DEVICE_STORAGE_CAP_BYTES,
                "cpu_cache_limit_bytes": min(2 * GIB, os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // 10),
                "retained_after_request_bytes": self.retained, "closed": self.closed, "failure": self.failure,
                "forward_counts": self.forward_counts, "counts": dict(self.counts),
                "member_batch_order": self.member_batch_order,
                "cpu_padded_transient_upper_bound_bytes": self.cpu_padding_transient_peak,
                "diagnostic_single_batch_rejection": self.diagnostic_budget_rejection,
                "final_cpu_scaled_output_payload_bytes": self.final_cpu_output_bytes,
                "fallback_count": 0, "source_proof": verify_sources(),
                "policy": "Whole original batches; chunk outer/member inner; no learned outputs cached; input cache released after each chunk"}


@contextmanager
def request_context(diagnostic=False):
    if _ACTIVE.get() is not None or reuse._STATE.get() is not None:
        raise StreamUnsupported("S owns its chunk input states; do not nest a reuse request")
    state = StreamState(diagnostic)
    token = _ACTIVE.set(state)
    try:
        yield state
    except BaseException as error:
        state.failure = type(error).__name__ + ": " + str(error)
        raise
    finally:
        if state.active_chunk is not None:
            state.active_chunk.clear()
            state.active_chunk = None
        state.retained = 0
        state.current_graph_rows.clear()
        state.closed = True
        _ACTIVE.reset(token)


class _ChunkState(reuse.RequestState):
    def __init__(self, stream, lookahead_cpu_bytes=0):
        super().__init__(stream.diagnostic)
        self.config = {"r1": True, "r2": True}
        self.device_limit = stream.config["budget_bytes"]
        self.reserve_cpu(lookahead_cpu_bytes)

    def reserve_device(self, byte_count, device):
        if torch.device(device).type != "cuda":
            raise StreamUnsupported("S requires the measured CUDA device")
        if self.device_bytes + byte_count > self.device_limit:
            raise StreamBudgetExceeded("S input storage allocation rejected before copy: retained=%d next=%d cap=%d" %
                                       (self.device_bytes, byte_count, self.device_limit))
        self.device_bytes += byte_count
        self.device_peak = max(self.device_peak, self.device_bytes)

    def register_batch(self, batch):
        graphs = batch._batch_graph
        key = reuse._batch_key(batch)
        self.graphs[key] = graphs
        self.reserve_cpu(_unique_storage_bytes(t for graph in graphs for t in _graph_tensors(graph)))
        self.counts["cpu_graph_batch_misses"] += 1
        self.counts["graph_constructions"] += len(graphs)
        if self.diagnostic:
            reference = reuse.ORIGINAL_BATCH_GRAPH(type(batch)(batch._data))
            for graph, expected in zip(graphs, reference):
                reuse._assert_graph_equal(graph, expected)
                self.counts["exact_graph_checks"] += 1
                tensors = {name: getattr(graph, name).clone() for name in reuse._GRAPH_TENSORS}
                self.reserve_cpu(_unique_storage_bytes(tensors.values()))
                self.graph_snapshots[id(graph)] = (graph, {**tensors, **{
                    name: reuse._freeze(getattr(graph, name)) for name in reuse._GRAPH_META}})
            self.counts["diagnostic_reference_constructions"] += len(reference)

    def actual_device_bytes(self):
        tensors = [t for values in self.graph_device.values() for t in values[:5]]
        tensors.extend(t for values in self.protein_device.values() for t in values if t is not None)
        actual = _unique_storage_bytes(tensors)
        if actual != self.device_bytes or actual > self.device_limit:
            raise StreamBudgetExceeded("Actual unique retained input storage disagrees with preallocation accounting")
        return actual

    def clear(self):
        self.graph_entries = len(self.graphs)
        self.device_graph_entries = len(self.graph_device)
        self.protein_entries = len(self.protein_device)
        self.graphs.clear()
        self.graph_snapshots.clear()
        self.graph_device.clear()
        self.protein_device.clear()
        self.tokens.clear()
        self.device_versions.clear()
        self.cpu_bytes = self.device_bytes = 0
        self.closed = True


def _graph_tensors(graph):
    return tuple(getattr(graph, name) for name in reuse._GRAPH_TENSORS)


def _protein_plan(records):
    rows = len(records)
    if not rows:
        raise StreamUnsupported("Empty original batch")
    sequence_length = max(len(record["seq"]) for record in records)
    features = [record["esm2_feats"] for record in records]
    if any(t.device.type != "cpu" or t.dtype != torch.float32 or t.ndim != 2 or t.shape[1] != 1280 for t in features):
        raise StreamUnsupported("Expected shared CPU FP32 ESM features with width 1280")
    esm_length = max(t.shape[0] for t in features)
    token_bytes = rows * sequence_length * 8
    esm_bytes = rows * esm_length * 1280 * 4
    return {"rows": rows, "token_padded_shape": [rows, sequence_length],
            "esm_padded_shape": [rows, esm_length, 1280],
            "common_len": min(sequence_length, esm_length),
            "token_storage_bytes": token_bytes, "esm_storage_bytes": esm_bytes,
            "protein_storage_bytes": token_bytes + esm_bytes}


def _batch_plan(batch):
    graphs = batch._batch_graph
    if not graphs or len(graphs) != 1:
        raise StreamUnsupported("Expected exactly one already packed original batch graph")
    tensors = [t for graph in graphs for t in _graph_tensors(graph)]
    if any(t.device.type != "cpu" or not t.is_contiguous() or t.storage_offset() != 0
           or t.untyped_storage().nbytes() != t.numel() * t.element_size() for t in tensors):
        raise StreamUnsupported("Graph source storage does not match the checked contiguous native packing contract")
    plan = _protein_plan(graphs[-1].protein_record_list)
    plan["graph_storage_bytes"] = _unique_storage_bytes(tensors)
    plan["device_input_storage_bytes"] = plan["protein_storage_bytes"] + plan["graph_storage_bytes"]
    plan["unique_token_cpu_bytes"] = sum(len(seq) * 8 for seq in {r["seq"] for r in graphs[-1].protein_record_list})
    plan["batch_identity_sha256"] = hashlib.sha256(repr(reuse._batch_key(batch)).encode()).hexdigest()
    return plan


def _assert_batch_budget(plan, budget_bytes, batch_index):
    if plan["device_input_storage_bytes"] > budget_bytes:
        raise StreamBudgetExceeded("Original batch %d needs %d input bytes, exceeds S cap %d before GPU allocation" %
                                   (batch_index, plan["device_input_storage_bytes"], budget_bytes))


def budget_rejection_and_cleanup_checks():
    """Allocation-free exception-path diagnostics; actual-batch probe also runs in S diagnostics.

    The accepted 1/2/4 GiB configuration is not changed. The rejection is tested
    by asking the same preallocation guard to reserve cap+1 bytes, without a copy.
    """
    if _ACTIVE.get() is not None or reuse._STATE.get() is not None:
        raise StreamUnsupported("Cleanup diagnostic requires an idle request boundary")
    before = dict(_CONFIG)
    stream = StreamState(diagnostic=True)
    chunk = _ChunkState(stream)
    rejected = False
    try:
        chunk.reserve_device(chunk.device_limit + 1, "cuda:0")
    except StreamBudgetExceeded:
        rejected = True
    if not rejected or chunk.device_bytes != 0 or chunk.graph_device or chunk.protein_device:
        raise AssertionError("Over-budget preallocation did not fail without retaining inputs")
    chunk.clear()

    class IntentionalCleanupProbe(Exception):
        pass

    failed_state = None
    try:
        with request_context(diagnostic=True) as failed_state:
            failed_state.active_chunk = _ChunkState(failed_state)
            raise IntentionalCleanupProbe("Intentional allocation-free request failure")
    except IntentionalCleanupProbe:
        pass
    with request_context(diagnostic=True) as next_state:
        assert _ACTIVE.get() is next_state and reuse._STATE.get() is None
    if (_ACTIVE.get() is not None or reuse._STATE.get() is not None
            or not failed_state.closed or not next_state.closed or _CONFIG != before):
        raise AssertionError("Request exception cleanup or subsequent context failed")
    return {"passed": True, "allocation_free": True, "configured_budget_bytes": before["budget_bytes"],
            "over_budget_reserve_rejected_before_copy": rejected,
            "failed_request_closed": failed_state.closed, "next_request_closed": next_state.closed,
            "context_states_cleared": True, "configuration_unchanged": True,
            "real_batch_rejection": "Each diagnostic S request additionally tests its first actual batch against its measured requirement minus one byte"}


def plan_for_dataset(test_data, models=None, scalers=None, budget_bytes=2 * GIB, batch_size=50):
    """Metadata-only protein lower bound, suitable for L preflight rejection.

    Exact graph storage is measured while S packs each batch, before GPU copies.
    This function never materializes padded tensors or modifies a request cache.
    """
    if batch_size != 50:
        raise StreamUnsupported("Only original batch=50 is supported")
    points = test_data._data
    plans = []
    for start in range(0, len(points), batch_size):
        plan = _protein_plan([point.protein_record for point in points[start:start + batch_size]])
        plan.update(start_row=start, stop_row=min(start + batch_size, len(points)))
        plans.append(plan)
    total = sum(plan["protein_storage_bytes"] for plan in plans)
    oversized = [i for i, plan in enumerate(plans) if plan["protein_storage_bytes"] > budget_bytes]
    return {"rows": len(points), "original_batch_size": batch_size, "batches": plans,
            "whole_request_protein_storage_lower_bound_bytes": total,
            "budget_bytes": budget_bytes, "single_batch_protein_lower_bound_rejections": oversized,
            "whole_request_exceeds_budget_by_protein_alone": total > budget_bytes,
            "graph_storage": "Measured from actual unique native buffers per batch during request, before any GPU allocation"}


def _calculate_stream(predictor):
    state = _ACTIVE.get()
    if state is None or not state.config["enabled"]:
        raise StreamUnsupported("Enabled S prediction requires a fresh streaming request context")
    if reuse._STATE.get() is not None or not reuse._CONFIG["r2"]:
        raise StreamUnsupported("S requires configured reuse R1/R2 but no outer reuse cache")
    probe = probes._ACTIVE.get()
    if probe is None or not probe.config["r3"] or probe.config["t1"] or probe.config["t2"]:
        raise StreamUnsupported("S requires a fresh R3 probe context with T1/T2 disabled")
    models, scalers = list(predictor.models), list(predictor.scalers)
    predictor.models, predictor.scalers = models, scalers
    if [id(model) for model in models] != [m["model_id"] for m in _INSPECTION["members"]]:
        raise StreamUnsupported("Inspected model order changed")
    loader = predictor.test_data_loader
    if loader._batch_size != 50 or loader._shuffle or loader._class_balance or loader._num_workers != 0:
        raise StreamUnsupported("S requires original ordered batch=50, no worker prefetch")
    if list(loader._sampler) != list(range(len(predictor.test_data))):
        raise StreamUnsupported("Original sampler is not the required contiguous row order")
    host_outputs = {id(model): ([], []) for model in models}
    iterator = iter(loader)
    pending = None
    row_offset = 0
    batch_index = 0
    cpu_limit = min(2 * GIB, os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // 10)
    batches = []
    try:
        while True:
            batches, plans = [], []
            device_bytes = cpu_graph_bytes = cpu_token_upper = 0
            exhausted = False
            while True:
                if pending is not None:
                    batch, plan = pending
                    pending = None
                else:
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    plan = _batch_plan(batch)
                try:
                    _assert_batch_budget(plan, state.config["budget_bytes"], batch_index + len(batches))
                except StreamBudgetExceeded:
                    state.counts["single_batch_budget_rejections"] += 1
                    raise
                if state.diagnostic and state.diagnostic_budget_rejection is None:
                    diagnostic_cap = plan["device_input_storage_bytes"] - 1
                    try:
                        _assert_batch_budget(plan, diagnostic_cap, batch_index + len(batches))
                    except StreamBudgetExceeded as error:
                        state.diagnostic_budget_rejection = {"passed": True, "actual_batch_input_bytes": plan["device_input_storage_bytes"],
                            "diagnostic_guard_limit_bytes": diagnostic_cap, "error": str(error), "no_GPU_allocation": True,
                            "production_configuration_unchanged": True}
                    else:
                        raise AssertionError("Actual oversized batch preflight did not reject")
                graph_factor = 2 if state.diagnostic else 1
                next_cpu = cpu_graph_bytes + plan["graph_storage_bytes"] * graph_factor + cpu_token_upper + plan["unique_token_cpu_bytes"]
                # Reserve one lookahead graph in the same CPU cache accounting.
                if batches and (device_bytes + plan["device_input_storage_bytes"] > state.config["budget_bytes"]
                                or next_cpu + plan["graph_storage_bytes"] > cpu_limit):
                    pending = (batch, plan)
                    break
                if next_cpu > cpu_limit:
                    raise StreamBudgetExceeded("Single batch CPU input cache exceeds the configured host limit")
                batches.append(batch)
                plans.append(plan)
                device_bytes += plan["device_input_storage_bytes"]
                cpu_graph_bytes += plan["graph_storage_bytes"] * graph_factor
                cpu_token_upper += plan["unique_token_cpu_bytes"]
            if not batches:
                break
            lookahead_bytes = pending[1]["graph_storage_bytes"] if pending is not None else 0
            chunk = _ChunkState(state, lookahead_bytes)
            state.active_chunk = chunk
            token = reuse._STATE.set(chunk)
            chunk_rows = sum(plan["rows"] for plan in plans)
            try:
                offset = row_offset
                for local_index, (batch, plan) in enumerate(zip(batches, plans)):
                    chunk.register_batch(batch)
                    state.current_graph_rows[id(batch._batch_graph[0])] = {
                        "batch_index": batch_index + local_index, "start_row": offset,
                        "stop_row": offset + plan["rows"], "rows": plan["rows"]}
                    offset += plan["rows"]
                    state.cpu_padding_transient_peak = max(state.cpu_padding_transient_peak, plan["protein_storage_bytes"])
                for member, (model, scaler_list) in enumerate(zip(models, scalers)):
                    preds, variances = probes._predict_dispatch(
                        model=model, data_loader=batches, disable_progress_bar=True,
                        scaler=scaler_list[0], atom_bond_scaler=scaler_list[4], return_unc_parameters=True)
                    if len(preds) != chunk_rows or len(variances) != chunk_rows:
                        raise StreamUnsupported("Chunk member output row count differs")
                    host_outputs[id(model)][0].extend(preds)
                    host_outputs[id(model)][1].extend(variances)
                    if member == 0:
                        actual = chunk.actual_device_bytes()
                        if actual != device_bytes:
                            raise StreamUnsupported("Actual chunk allocation differs from the pre-GPU batch storage plan")
                    else:
                        chunk.counts["cpu_graph_batch_hits"] += len(batches)
                        chunk.counts["graph_constructions_avoided"] += len(batches)
                chunk.check_immutable()
                # Output drains are synchronous D2H; explicitly finish all input
                # dependencies before freeing the chunk for the next iteration.
                torch.cuda.synchronize(models[0].device)
                actual = chunk.actual_device_bytes()
                state.device_peak = max(state.device_peak, actual)
                state.cpu_peak = max(state.cpu_peak, chunk.cpu_peak)
                state.counts.update(chunk.counts)
                state.chunks.append({"chunk_index": len(state.chunks), "start_row": row_offset,
                    "stop_row": row_offset + chunk_rows, "first_batch_index": batch_index,
                    "batch_count": len(batches), "planned_device_storage_bytes": device_bytes,
                    "actual_unique_device_storage_bytes": actual, "cpu_cache_peak_bytes": chunk.cpu_peak,
                    "lookahead_graph_cpu_bytes": lookahead_bytes, "batches": plans,
                    "diagnostic": state.diagnostic, "exact_graph_checks": chunk.counts["exact_graph_checks"],
                    "exact_protein_input_checks": chunk.counts["exact_protein_input_checks"],
                    "device_input_immutability_checks": chunk.counts["device_input_immutability_checks"],
                    "released": False})
            finally:
                # Also synchronize failed forwards before clearing shared inputs.
                try:
                    torch.cuda.synchronize(models[0].device)
                finally:
                    state.device_peak = max(state.device_peak, chunk.device_peak)
                    state.cpu_peak = max(state.cpu_peak, chunk.cpu_peak)
                    cached_weakrefs = []
                    if state.diagnostic:
                        cached_weakrefs = [weakref.ref(t) for values in chunk.graph_device.values() for t in values[:5]]
                        cached_weakrefs.extend(weakref.ref(t) for values in chunk.protein_device.values() for t in values if t is not None)
                    chunk.clear()
                    reuse._STATE.reset(token)
                    state.active_chunk = None
                    for batch in batches:
                        batch._batch_graph = None
                    batches.clear()
                    state.current_graph_rows.clear()
                    if state.chunks and state.chunks[-1]["start_row"] == row_offset:
                        state.chunks[-1]["released"] = True
                        state.chunks[-1]["actual_device_cache_bytes_after_clear"] = chunk.actual_device_bytes()
                        state.chunks[-1]["device_tensor_references_alive_after_clear"] = sum(ref() is not None for ref in cached_weakrefs) if state.diagnostic else None
                    if state.diagnostic and any(ref() is not None for ref in cached_weakrefs):
                        raise AssertionError("A chunk device input tensor survived cache release")
            row_offset += chunk_rows
            batch_index += len(plans)
            if exhausted:
                break
        if row_offset != len(predictor.test_data):
            raise StreamUnsupported("S did not produce every original input row")
        state.final_cpu_output_bytes = 10 * row_offset * 2 * 8
        token = _HOST_RESULTS.set(host_outputs)
        try:
            _HOST_AGGREGATE(predictor)
        finally:
            _HOST_RESULTS.reset(token)
    finally:
        if pending is not None:
            pending[0]._batch_graph = None
        for batch in batches:
            batch._batch_graph = None
        batches.clear()
        host_outputs.clear()
        state.output_peak_owned_bytes = probe.counters["r3_peak_owned_output_bytes"]
        state.retained = 0
