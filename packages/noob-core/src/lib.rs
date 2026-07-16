use pyo3::prelude::*;

pub type FxIndexSet<T> = indexmap::IndexSet<T, rustc_hash::FxBuildHasher>;
pub type FxIndexMap<K, V> = indexmap::IndexMap<K, V, rustc_hash::FxBuildHasher>;

pub mod bridge;
pub mod epoch;
pub mod exceptions;
pub mod item;
pub mod scheduler;
pub mod sorter;
pub mod tube;

#[pymodule]
fn _core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<bridge::PyScheduler>()?;
    m.add_class::<epoch::Epoch>()?;
    Ok(())
}
