"""Controlled BatchMolGraph constructor replacements; import in the pinned checkout.

All timed work begins with the original Python MolGraph objects.  No prepared
arrays or packed batches are cached.  Diagnostic counting is opt-in so headline
timings can install a constructor directly, without a timing/count wrapper.
"""

from __future__ import annotations

from collections import Counter
from functools import wraps
from importlib import import_module

import numpy as np
import torch

from catpred.features import featurization as feat


ORIGINAL_INIT = feat.BatchMolGraph.__init__
original = ORIGINAL_INIT
_COUNTS = Counter()
_ARM = "A"
_INSTRUMENT = False
_RUST_PACK = None


def _metadata(self, mol_graphs, embed_feature_list, protein_record_list):
    # Keep reference identity and the original first-graph configuration rules.
    self.mol_graphs = mol_graphs
    self.protein_record_list = protein_record_list
    self.embed_feature_list = embed_feature_list
    self.overwrite_default_atom_features = mol_graphs[0].overwrite_default_atom_features
    self.overwrite_default_bond_features = mol_graphs[0].overwrite_default_bond_features
    self.is_reaction = mol_graphs[0].is_reaction
    self.atom_fdim = feat.get_atom_fdim(
        overwrite_default_atom=self.overwrite_default_atom_features,
        is_reaction=self.is_reaction,
    )
    self.bond_fdim = feat.get_bond_fdim(
        overwrite_default_bond=self.overwrite_default_bond_features,
        overwrite_default_atom=self.overwrite_default_atom_features,
        is_reaction=self.is_reaction,
    )


def _finish(self, arrays):
    f_atoms, f_bonds, a2b, b2a, b2revb, self.a_scope, self.b_scope = arrays
    self.n_atoms = f_atoms.shape[0]
    self.n_bonds = f_bonds.shape[0]
    self.max_num_bonds = a2b.shape[1]
    # Each tensor holds a reference to its NumPy owner.  These calls share the
    # writable, C-contiguous buffers; no list-to-tensor conversion follows them.
    self.f_atoms = torch.from_numpy(f_atoms)
    self.f_bonds = torch.from_numpy(f_bonds)
    self.a2b = torch.from_numpy(a2b)
    self.b2a = torch.from_numpy(b2a)
    self.b2revb = torch.from_numpy(b2revb)
    self.b2b = None
    self.a2a = None
    self.b2br = None


def pack_numpy(mol_graphs, atom_fdim, bond_fdim):
    """Preallocate final typed buffers, copy features, and offset graph indices."""
    n_atoms = 1 + sum(graph.n_atoms for graph in mol_graphs)
    n_bonds = 1 + sum(graph.n_bonds for graph in mol_graphs)
    max_num_bonds = max(1, max(
        (len(row) for graph in mol_graphs for row in graph.a2b), default=0
    ))
    f_atoms = np.zeros((n_atoms, atom_fdim), dtype=np.float32)
    f_bonds = np.zeros((n_bonds, bond_fdim), dtype=np.float32)
    a2b = np.zeros((n_atoms, max_num_bonds), dtype=np.int64)
    b2a = np.zeros(n_bonds, dtype=np.int64)
    b2revb = np.zeros(n_bonds, dtype=np.int64)
    a_scope, b_scope = [], []
    atom_offset = bond_offset = 1
    for graph in mol_graphs:
        na, nb = graph.n_atoms, graph.n_bonds
        if na:
            # NumPy converts Python numeric feature rows directly to FP32 on
            # assignment.  Any temporary conversion/copy is inside this call.
            f_atoms[atom_offset:atom_offset + na] = graph.f_atoms
        if nb:
            f_bonds[bond_offset:bond_offset + nb] = graph.f_bonds
            b2a[bond_offset:bond_offset + nb] = graph.b2a
            b2a[bond_offset:bond_offset + nb] += atom_offset
            b2revb[bond_offset:bond_offset + nb] = graph.b2revb
            b2revb[bond_offset:bond_offset + nb] += bond_offset
        for atom_idx in range(na):
            incoming = graph.a2b[atom_idx]
            count = len(incoming)
            if count:
                # Degrees are small: offset while traversing the input list
                # instead of launching a NumPy ufunc for every atom.
                a2b[atom_offset + atom_idx, :count] = [
                    bond + bond_offset for bond in incoming
                ]
        a_scope.append((atom_offset, na))
        b_scope.append((bond_offset, nb))
        atom_offset += na
        bond_offset += nb
    return f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope


def numpy_init(self, mol_graphs, embed_feature_list, protein_record_list):
    _metadata(self, mol_graphs, embed_feature_list, protein_record_list)
    _finish(self, pack_numpy(mol_graphs, self.atom_fdim, self.bond_fdim))


def rust_init(self, mol_graphs, embed_feature_list, protein_record_list):
    _metadata(self, mol_graphs, embed_feature_list, protein_record_list)
    _finish(self, _RUST_PACK(mol_graphs, self.atom_fdim, self.bond_fdim))


def _instrumented(arm, constructor):
    @wraps(constructor)
    def measured(self, mol_graphs, embed_feature_list, protein_record_list):
        constructor(self, mol_graphs, embed_feature_list, protein_record_list)
        _COUNTS["constructor_calls"] += 1
        _COUNTS["molgraph_inputs"] += len(mol_graphs)
        _COUNTS["atoms_including_padding"] += self.n_atoms
        _COUNTS["bonds_including_padding"] += self.n_bonds
        _COUNTS["output_buffer_bytes"] += sum(
            value.numel() * value.element_size()
            for value in (self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb)
        )
        _COUNTS["float_feature_values_converted"] += (
            (self.n_atoms - 1) * self.atom_fdim + (self.n_bonds - 1) * self.bond_fdim
        )
        _COUNTS["tensor_from_list_calls"] += 5 if arm == "A" else 0
        _COUNTS["tensor_from_numpy_calls"] += 0 if arm == "A" else 5
        _COUNTS["rust_ffi_calls"] += int(arm == "C")
        _COUNTS["numpy_pack_calls"] += int(arm == "B")
    return measured


def install_arm(arm: str, instrument: bool = False):
    """Install A/original, B/NumPy, or C/Rust on the existing class.

    Arm installation/import is setup work, outside the constructor timing.
    With instrument=False this adds no counter wrapper to any constructor.
    """
    global _ARM, _INSTRUMENT, _RUST_PACK
    arm = arm.upper()
    constructors = {"A": ORIGINAL_INIT, "B": numpy_init, "C": rust_init}
    if arm not in constructors:
        raise ValueError(f"Unknown packing arm: {arm!r}")
    if arm == "C" and _RUST_PACK is None:
        _RUST_PACK = import_module("catpred_rust_packing").pack
    constructor = constructors[arm]
    feat.BatchMolGraph.__init__ = _instrumented(arm, constructor) if instrument else constructor
    _ARM, _INSTRUMENT = arm, bool(instrument)
    return constructor


def reset_counts():
    _COUNTS.clear()


def counts():
    return {"arm": _ARM, "instrumented": _INSTRUMENT, **dict(_COUNTS)}


reset_counters = reset_counts
get_counters = counts
snapshot_counts = counts
set_arm = install_arm
