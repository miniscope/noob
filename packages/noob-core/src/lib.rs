use pyo3::prelude::*;

pub type FxIndexSet<T> = indexmap::IndexSet<T, rustc_hash::FxBuildHasher>;
pub type FxIndexMap<K, V> = indexmap::IndexMap<K, V, rustc_hash::FxBuildHasher>;

pub mod exceptions;
pub mod item;
pub mod toposort;
// not yet ported to the new design:
pub mod epoch;
// pub mod event;
// pub mod scheduler;

/// The python extension module. Empty for now: the sorter is private to
/// rust, and the scheduler that will be exposed here isn't ported yet.
#[pymodule]
fn _core(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
