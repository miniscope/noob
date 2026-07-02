se std::cmp::Ordering;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::bridge::epoch_to_py;

/// Native representation of `noob.types.Epoch`:
/// a sequence of `(node_id, epoch_number)` segments.
/// Root epochs are a single `("tube", n)` segment.
pub type EpochKey = Vec<(String, i64)>;

/// The integer of the root segment
pub fn root_int(epoch: &EpochKey) -> i64 {
    epoch[0].1
}

/// The immediate parent epoch, or None for root epochs
pub fn parent(epoch: &EpochKey) -> Option<EpochKey> {
    if epoch.len() > 1 {
        Some(epoch[..epoch.len() - 1].to_vec())
    } else {
        None
    }
}

/// All parent epochs, immediate parent first, root last.
/// Matches `Epoch.parents`.
pub fn parents(epoch: &EpochKey) -> Vec<EpochKey> {
    (1..epoch.len())
        .rev()
        .map(|i| epoch[..i].to_vec())
        .collect()
}

/// Sort key ordering used by `Scheduler.get_ready`:
/// compare node-id tuples, then epoch-int tuples.
pub fn get_ready_order(a: &EpochKey, b: &EpochKey) -> Ordering {
    a.iter()
        .map(|seg| seg.0.as_str())
        .cmp(b.iter().map(|seg| seg.0.as_str()))
        .then_with(|| a.iter().map(|seg| seg.1).cmp(b.iter().map(|seg| seg.1)))
}

/// Output-side conversion: wraps a native epoch key so returning it
/// constructs a real `noob.types.Epoch`
pub struct Ep(pub EpochKey);

impl<'py> IntoPyObject<'py> for Ep {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        epoch_to_py(py, &self.0)
    }
}

/// Input-side conversion for parameters that accept `int | Epoch`,
/// e.g. `add_epoch` / `end_epoch` / `__getitem__`
#[derive(Clone, Debug)]
pub enum EpochArg {
    Int(i64),
    Key(EpochKey),
}

impl EpochArg {
    pub fn into_key(self) -> EpochKey {
        match self {
            EpochArg::Int(number) => vec![("tube".to_owned(), number)],
            EpochArg::Key(key) => key,
        }
    }

    /// Matches the python `epoch == -1` check that means "the latest epoch":
    /// the int -1 or a root Epoch with number -1
    pub fn is_latest(&self) -> bool {
        match self {
            EpochArg::Int(number) => *number == -1,
            EpochArg::Key(key) => key.len() == 1 && key[0].1 == -1,
        }
    }
}

impl<'py> FromPyObject<'py> for EpochArg {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(number) = ob.extract::<i64>() {
            return Ok(EpochArg::Int(number));
        }
        if let Ok(key) = ob.extract::<EpochKey>() {
            return Ok(EpochArg::Key(key));
        }
        Err(PyTypeError::new_err(
            "Can only create an epoch from an epoch or integer",
        ))
    }
}

/// Match `Epoch.__repr__`: the bare int for root epochs,
/// a tuple of tuples for subepochs.
pub fn fmt_epoch(epoch: &EpochKey) -> String {
    if epoch.len() == 1 {
        format!("{}", epoch[0].1)
    } else {
        let segs: Vec<String> = epoch.iter().map(|(n, e)| format!("('{n}', {e})")).collect();
        format!("({})", segs.join(", "))
    }
}