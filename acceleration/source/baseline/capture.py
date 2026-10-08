"""Full-precision evidence from transfers and host results already required by inference.

No extra CUDA transfer is performed. Required host arrays are retained within the
request; packing/NPZ serialization and hashes occur after the request timer.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from collections import defaultdict
from pathlib import Path
import hashlib
import json

_ACTIVE = ContextVar('phase3_precision_capture', default=None)
_INSTALLED = False


class CaptureState:
    def __init__(self, row_ids):
        self.row_ids = list(row_ids)
        self.member_pieces = defaultdict(list)
        self.aggregate = None
        self.final_frame = None
        self.closed = False

    def save(self, directory):
        import numpy as np
        directory = Path(directory)
        n = len(self.row_ids)
        members = []
        for i in range(10):
            value = np.concatenate(self.member_pieces[i], axis=0)
            assert value.shape == (n, 2) and value.dtype == np.float32
            assert np.isfinite(value).all(), ('nonfinite_raw_member', i)
            members.append(value)
        raw = np.stack(members, axis=0)
        assert self.aggregate is not None and self.final_frame is not None
        columns = ['uncal_preds', 'uncal_vars', 'uncal_aleatoric_vars', 'uncal_epistemic_vars']
        aggregate = np.concatenate([np.asarray(self.aggregate[key]) for key in columns], axis=1)
        assert aggregate.shape == (n, 4) and aggregate.dtype == np.float64
        frame = self.final_frame
        numeric_columns = frame.select_dtypes(include='number').columns.tolist()
        metadata_columns = [key for key in frame.columns if key not in numeric_columns]
        numeric = frame[numeric_columns].to_numpy(copy=True)
        assert frame.row_id.tolist() == self.row_ids
        arrays = dict(member_raw=raw, numeric_raw=aggregate, numeric_processed=numeric)
        for key, value in arrays.items():
            assert np.isfinite(value).all(), ('nonfinite_precision_output', key)
        path = directory/'precision.npz'
        np.savez(path, **arrays)
        metadata = dict(
            capture_boundary='Original required output D2H; original MVE aggregation before CSV; final DataFrame immediately before CSV',
            row_ids=self.row_ids,
            stages={
                'member_raw':dict(dtype=raw.dtype.str, shape=list(raw.shape), columns=['raw_mean','raw_variance']),
                'numeric_raw':dict(dtype=aggregate.dtype.str,shape=list(aggregate.shape),columns=columns,
                    metadata_columns=['row_id'],metadata_rows=[[v] for v in self.row_ids],row_ids=self.row_ids),
                'numeric_processed':dict(dtype=numeric.dtype.str,shape=list(numeric.shape),columns=numeric_columns,
                    metadata_columns=metadata_columns,metadata_rows=frame[metadata_columns].values.tolist(),row_ids=self.row_ids),
            },
            retained_raw_member_bytes=int(raw.nbytes), aggregate_bytes=int(aggregate.nbytes),
            final_numeric_bytes=int(numeric.nbytes), extra_cuda_transfers=0,
        )
        manifest = directory/'precision_manifest.json'
        manifest.write_text(json.dumps(metadata,indent=2,allow_nan=False))
        def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
        return dict(precision_path=str(path),precision_sha256=sha(path),
            precision_manifest_path=str(manifest),precision_manifest_sha256=sha(manifest),
            precision_raw_member_bytes=int(raw.nbytes),precision_extra_cuda_transfers=0)


@contextmanager
def request_context(row_ids):
    if _ACTIVE.get() is not None:
        raise RuntimeError('Nested full precision capture')
    state = CaptureState(row_ids)
    token = _ACTIVE.set(state)
    try:
        yield state
    finally:
        state.closed = True
        _ACTIVE.reset(token)


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import probes
    import pandas as pd
    from catpred.uncertainty.uncertainty_predictor import UncertaintyPredictor, MVEPredictor
    assert probes._COUNTED_PREDICT is not None, 'Configure probes first'
    original_copy = probes._cpu_numpy
    original_init = UncertaintyPredictor.__init__
    original_csv = pd.DataFrame.to_csv

    def transfer(tensor, diagnostic=False, member=None):
        result = original_copy(tensor,diagnostic=diagnostic,member=member)
        state = _ACTIVE.get()
        if state is not None and not diagnostic:
            if member not in range(10):
                raise AssertionError(('unregistered_capture_member',member))
            state.member_pieces[member].append(result)
        return result

    def initialize(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        state = _ACTIVE.get()
        if state is not None and isinstance(self,MVEPredictor):
            assert state.aggregate is None, 'Multiple MVE aggregators in one request'
            state.aggregate = {key:getattr(self,key) for key in (
                'uncal_preds','uncal_vars','uncal_aleatoric_vars','uncal_epistemic_vars')}

    def write_csv(frame,*args,**kwargs):
        state = _ACTIVE.get()
        if state is not None and 'Prediction_log10' in frame:
            assert state.final_frame is None, 'Multiple final result frames'
            state.final_frame = frame
        return original_csv(frame,*args,**kwargs)

    probes._cpu_numpy = transfer
    probes._COUNTED_PREDICT.__globals__['_phase2_cpu_numpy'] = transfer
    UncertaintyPredictor.__init__ = initialize
    pd.DataFrame.to_csv = write_csv
    _INSTALLED = True


def compare_files(left,right):
    """Independent of the live model; preserves dtype, shape, order and metadata."""
    import numpy as np
    result = {}
    a_meta = json.loads(Path(left['precision_manifest_path']).read_text())
    b_meta = json.loads(Path(right['precision_manifest_path']).read_text())
    with np.load(left['precision_path'],allow_pickle=False) as aa, np.load(right['precision_path'],allow_pickle=False) as bb:
        for key in ('member_raw','numeric_raw','numeric_processed'):
            a,b = aa[key],bb[key]
            structure = a.shape == b.shape and a.dtype == b.dtype
            ma,mb = a_meta['stages'][key],b_meta['stages'][key]
            structure = structure and ma == mb and a_meta['row_ids']==b_meta['row_ids']
            finite = structure and bool(np.isfinite(a).all() and np.isfinite(b).all())
            if finite:
                delta = np.abs(a.astype(np.float64)-b.astype(np.float64))
                relative = np.divide(delta,np.abs(a),out=np.zeros_like(delta),where=a!=0)
                exact = bool(np.array_equal(a,b))
                close = bool(np.allclose(b,a,atol=1e-6,rtol=1e-5))
                bad = np.argwhere(a!=b)
                result[key] = dict(structure_equal=True,finite=True,exact=exact,allclose=close,
                    max_absolute_error=float(delta.max()),max_relative_error_nonzero_reference=float(relative.max()),
                    nonzero_error_at_zero_reference=int(((a==0)&(delta!=0)).sum()),
                    first_differing_indices=bad[:10].tolist(),mismatching_values=int(len(bad)))
            else:
                result[key] = dict(structure_equal=bool(structure),finite=False,exact=False,allclose=False)
    return dict(atol=1e-6,rtol=1e-5,arrays=result,
        exact=all(v['exact'] for v in result.values()),allclose=all(v['allclose'] for v in result.values()))
