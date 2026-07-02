use pyo3::prelude::*;

pub mod epoch;
pub mod event;
pub mod exceptions;
pub mod item;
pub mod scheduler;
pub mod toposort;

#[pymodule]
fn _core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
  m.add_class::<scheduler::Scheduler>()?;
//   m.add_class::<pysorter::CoreTopoSorter>()?;
  Ok(())
}