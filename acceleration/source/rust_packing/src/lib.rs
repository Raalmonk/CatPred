//! BatchMolGraph packing only: graph extraction, numeric conversion, allocation,
//! offsetting and owned NumPy outputs are all inside this single FFI boundary.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList, PyListMethods};

type Packed<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Vec<(usize, usize)>,
    Vec<(usize, usize)>,
);

fn value_error(message: &str) -> PyErr {
    PyValueError::new_err(message.to_owned())
}

fn checked_product(left: usize, right: usize) -> PyResult<usize> {
    left.checked_mul(right).ok_or_else(|| value_error("Packed shape overflow"))
}

fn copy_features(
    rows: &Bound<'_, PyAny>,
    target: &mut [f32],
    row_count: usize,
    width: usize,
) -> PyResult<()> {
    let rows = rows.cast::<PyList>()?;
    if rows.len() != row_count {
        return Err(value_error("Feature row count disagrees with MolGraph"));
    }
    for (row_idx, row) in rows.iter().enumerate() {
        let row = row.cast::<PyList>()?;
        if row.len() != width {
            return Err(value_error("Feature width disagrees with BatchMolGraph"));
        }
        let output = &mut target[row_idx * width..(row_idx + 1) * width];
        for (column, value) in row.iter().enumerate() {
            // PyO3's f32 extraction reads Python's numeric value and converts
            // once to f32; identical final precision to NumPy float32/torch.float.
            output[column] = value.extract::<f32>()?;
        }
    }
    Ok(())
}

fn copy_indices(
    values: &Bound<'_, PyAny>,
    target: &mut [i64],
    offset: usize,
) -> PyResult<()> {
    let values = values.cast::<PyList>()?;
    if values.len() != target.len() {
        return Err(value_error("Index count disagrees with MolGraph"));
    }
    for (index, value) in values.iter().enumerate() {
        target[index] = value.extract::<i64>()? + offset as i64;
    }
    Ok(())
}

#[pyfunction]
fn pack<'py>(
    py: Python<'py>,
    mol_graphs: &Bound<'py, PyList>,
    atom_fdim: usize,
    bond_fdim: usize,
) -> PyResult<Packed<'py>> {
    let mut n_atoms = 1usize;
    let mut n_bonds = 1usize;
    let mut max_num_bonds = 1usize;
    let mut sizes = Vec::with_capacity(mol_graphs.len());
    for graph in mol_graphs.iter() {
        let na = graph.getattr("n_atoms")?.extract::<usize>()?;
        let nb = graph.getattr("n_bonds")?.extract::<usize>()?;
        n_atoms = n_atoms.checked_add(na).ok_or_else(|| value_error("Atom count overflow"))?;
        n_bonds = n_bonds.checked_add(nb).ok_or_else(|| value_error("Bond count overflow"))?;
        let adjacency = graph.getattr("a2b")?;
        for incoming in adjacency.cast::<PyList>()?.iter() {
            max_num_bonds = max_num_bonds.max(incoming.len()?);
        }
        sizes.push((na, nb));
    }

    // Same five zero-filled final buffers as pack_numpy. Padding is written
    // only here; graph copying never changes row zero or adjacency padding.
    let mut f_atoms = vec![0f32; checked_product(n_atoms, atom_fdim)?];
    let mut f_bonds = vec![0f32; checked_product(n_bonds, bond_fdim)?];
    let mut a2b = vec![0i64; checked_product(n_atoms, max_num_bonds)?];
    let mut b2a = vec![0i64; n_bonds];
    let mut b2revb = vec![0i64; n_bonds];
    let mut a_scope = Vec::with_capacity(mol_graphs.len());
    let mut b_scope = Vec::with_capacity(mol_graphs.len());
    let mut atom_offset = 1usize;
    let mut bond_offset = 1usize;

    for (graph, (na, nb)) in mol_graphs.iter().zip(sizes.into_iter()) {
        copy_features(
            &graph.getattr("f_atoms")?,
            &mut f_atoms[atom_offset * atom_fdim..(atom_offset + na) * atom_fdim],
            na,
            atom_fdim,
        )?;
        copy_features(
            &graph.getattr("f_bonds")?,
            &mut f_bonds[bond_offset * bond_fdim..(bond_offset + nb) * bond_fdim],
            nb,
            bond_fdim,
        )?;
        copy_indices(
            &graph.getattr("b2a")?,
            &mut b2a[bond_offset..bond_offset + nb],
            atom_offset,
        )?;
        copy_indices(
            &graph.getattr("b2revb")?,
            &mut b2revb[bond_offset..bond_offset + nb],
            bond_offset,
        )?;
        let incoming_rows_object = graph.getattr("a2b")?;
        let incoming_rows = incoming_rows_object.cast::<PyList>()?;
        if incoming_rows.len() != na {
            return Err(value_error("Adjacency row count disagrees with MolGraph"));
        }
        for (atom_idx, incoming) in incoming_rows.iter().enumerate() {
            let count = incoming.len()?;
            let start = (atom_offset + atom_idx) * max_num_bonds;
            copy_indices(&incoming, &mut a2b[start..start + count], bond_offset)?;
        }
        a_scope.push((atom_offset, na));
        b_scope.push((bond_offset, nb));
        atom_offset += na;
        bond_offset += nb;
    }

    let atoms = Array2::from_shape_vec((n_atoms, atom_fdim), f_atoms)
        .map_err(|error| value_error(&error.to_string()))?;
    let bonds = Array2::from_shape_vec((n_bonds, bond_fdim), f_bonds)
        .map_err(|error| value_error(&error.to_string()))?;
    let adjacency = Array2::from_shape_vec((n_atoms, max_num_bonds), a2b)
        .map_err(|error| value_error(&error.to_string()))?;
    // into_pyarray transfers each Rust allocation into a Python-owned backing
    // object; it does not copy to a new NumPy allocation or borrow stack data.
    Ok((
        atoms.into_pyarray(py),
        bonds.into_pyarray(py),
        adjacency.into_pyarray(py),
        b2a.into_pyarray(py),
        b2revb.into_pyarray(py),
        a_scope,
        b_scope,
    ))
}

#[pymodule]
fn catpred_rust_packing(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(pack, module)?)?;
    module.add("BUILD_PROFILE", if cfg!(debug_assertions) { "debug" } else { "release" })?;
    module.add("PYO3_VERSION", "0.27.0")?;
    module.add("RUST_NUMPY_VERSION", "0.27.0")?;
    Ok(())
}
