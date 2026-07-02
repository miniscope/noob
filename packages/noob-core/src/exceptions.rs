use std::error::Error;
use std::fmt;

/// Errors raised by the pure-rust core.
///
/// These mirror `noob.exceptions`: the scheduler's python boundary layer is
/// responsible for translating them into the corresponding python exception
/// types. Keeping this enum free of pyo3 lets the sorter and its tests stay
/// pure rust.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoreError {
    /// `noob.exceptions.AlreadyDoneError`
    AlreadyDone(String),
    /// `noob.exceptions.NotAddedError`
    NotAdded(String),
    /// `ValueError`
    Value(String),
}

/// Shorthand for fallible core operations, like `PyResult<T>` is for pyo3.
pub type CoreResult<T> = Result<T, CoreError>;

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreError::AlreadyDone(msg) | CoreError::NotAdded(msg) | CoreError::Value(msg) => {
                write!(f, "{msg}")
            }
        }
    }
}

impl Error for CoreError {}
