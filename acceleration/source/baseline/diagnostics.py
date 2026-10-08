"""Diagnostic-only instrumentation, excluded from headline measurements."""
from contextlib import contextmanager
from collections import Counter
import gzip
import json
from pathlib import Path
import shutil


@contextmanager
def transfer_and_module_diagnostics(models):
    import torch
    import catpred.models.model as model_module
    import reuse
    counts = Counter()
    handles = []
    stacks = {}
    original_to, original_cpu = torch.Tensor.to, torch.Tensor.cpu
    original_as_tensor = torch.as_tensor
    original_pad = model_module.pad_sequence
    original_reuse_pad = reuse.pad_sequence

    def transfer(source, result, route):
        if source.device.type != result.device.type:
            direction = source.device.type + '_to_' + result.device.type
            counts[direction+'_calls'] += 1
            counts[direction+'_bytes'] += result.numel()*result.element_size()
            counts[route+'_actual_transfers'] += 1
        return result

    def to(tensor, *args, **kwargs):
        counts['tensor_to_calls'] += 1
        return transfer(tensor, original_to(tensor, *args, **kwargs), 'tensor_to')

    def cpu(tensor, *args, **kwargs):
        counts['tensor_cpu_calls'] += 1
        return transfer(tensor, original_cpu(tensor, *args, **kwargs), 'tensor_cpu')

    def as_tensor(data, *args, **kwargs):
        result = original_as_tensor(data, *args, **kwargs)
        if not torch.is_tensor(data) and result.device.type == 'cuda':
            counts['non_tensor_to_cuda_calls'] += 1
            counts['non_tensor_to_cuda_bytes'] += result.numel()*result.element_size()
        return result

    def pad(sequences, *args, **kwargs):
        label = 'token_padding' if sequences[0].dtype in (torch.int32, torch.int64) else 'esm_padding'
        counts[label+'_calls'] += 1
        counts[label+'_rows'] += len(sequences)
        result = original_pad(sequences, *args, **kwargs)
        counts[label+'_output_bytes'] += result.numel()*result.element_size()
        return result

    def attach(module, label):
        def begin(mod, args):
            marker = torch.profiler.record_function(label)
            marker.__enter__()
            stacks.setdefault(id(mod), []).append(marker)
            counts[label+'_calls'] += 1
        def end(mod, args, output):
            stacks[id(mod)].pop().__exit__(None, None, None)
        handles.extend([module.register_forward_pre_hook(begin), module.register_forward_hook(end)])

    for i, model in enumerate(models):
        for name in ('encoder', 'seq_embedder', 'multihead_attn', 'attentive_pooler', 'readout'):
            module = getattr(model, name, None)
            if module is not None:
                attach(module, 'CatPred/'+name)
    torch.Tensor.to, torch.Tensor.cpu, torch.as_tensor = to, cpu, as_tensor
    model_module.pad_sequence = pad
    reuse.pad_sequence = pad
    try:
        yield counts
    finally:
        torch.Tensor.to, torch.Tensor.cpu, torch.as_tensor = original_to, original_cpu, original_as_tensor
        model_module.pad_sequence = original_pad
        reuse.pad_sequence = original_reuse_pad
        for handle in handles:
            handle.remove()
        for stack in stacks.values():
            while stack:
                stack.pop().__exit__(None, None, None)


def save_torch_profile(profiler, output, stem):
    output = Path(output)
    raw, compressed = output/(stem+'.json'), output/(stem+'.json.gz')
    profiler.export_chrome_trace(str(raw))
    with raw.open('rb') as source, gzip.open(compressed, 'wb', compresslevel=6) as target:
        shutil.copyfileobj(source, target)
    raw.unlink()
    events = []
    for event in profiler.key_averages():
        events.append(dict(operator=event.key, count=event.count,
            self_cpu_time_us=event.self_cpu_time_total, cpu_time_us=event.cpu_time_total,
            self_device_time_us=getattr(event,'self_device_time_total',0),
            device_time_us=getattr(event,'device_time_total',0)))
    events.sort(key=lambda x:x['self_cpu_time_us'],reverse=True)
    (output/(stem+'_operators.json')).write_text(json.dumps(events,indent=2))
    (output/(stem+'_table.txt')).write_text(profiler.key_averages().table(sort_by='self_cpu_time_total',row_limit=60))
    return dict(trace_path=str(compressed), compressed_bytes=compressed.stat().st_size,
        attention_operators=[v for v in events if any(x in v['operator'].lower() for x in ('attention','scaled_dot','flash','efficient'))],
        copy_operators=[v for v in events if any(x in v['operator'].lower() for x in ('copy','memcpy','synchronize'))],
        time_units='microseconds; CPU and device durations overlap and are not summed')


@contextmanager
def original_graph_checks():
    """Compare each distinct original batch against Python, outside headline."""
    import torch
    import packing
    feat = packing.feat
    original = feat.BatchMolGraph.__init__
    seen = set()
    receipt = dict(distinct_batches_checked=0, exact=True, failures=[])
    def checked(graph,mol_graphs,embed_feature_list,protein_record_list):
        original(graph,mol_graphs,embed_feature_list,protein_record_list)
        key = tuple(id(value) for value in mol_graphs)
        if key in seen:
            return
        seen.add(key)
        reference = object.__new__(feat.BatchMolGraph)
        packing.ORIGINAL_INIT(reference,mol_graphs,embed_feature_list,protein_record_list)
        for name in ('f_atoms','f_bonds','a2b','b2a','b2revb'):
            actual,expected = getattr(graph,name),getattr(reference,name)
            assert actual.shape==expected.shape and actual.stride()==expected.stride()
            assert actual.dtype==expected.dtype and torch.equal(actual,expected), ('graph_mismatch',name)
        for name in ('a_scope','b_scope','n_atoms','n_bonds','max_num_bonds'):
            assert getattr(graph,name)==getattr(reference,name), ('graph_metadata',name)
        receipt['distinct_batches_checked'] += 1
    feat.BatchMolGraph.__init__ = checked
    try:
        yield receipt
    except BaseException as error:
        receipt['exact'] = False
        receipt['failures'].append(str(error))
        raise
    finally:
        feat.BatchMolGraph.__init__ = original
