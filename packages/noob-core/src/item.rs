use std::fmt;

use indexmap::IndexSet;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;


/// A graph item: either a node id, or a (node id, signal name) pair.
///
/// The native representation of `noob.types.NodeID | noob.types.NodeSignal`:
/// a `str` or a 2-tuple of `str` on the python side.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Item {
    Node(String),
    Signal(String, String),
}

impl Item {
    pub fn is_signal(&self) -> bool {
        matches!(self, Item::Signal(..))
    }

    /// The node id: itself for nodes, the node part for signals
    pub fn node_id(&self) -> &str {
        match self {
            Item::Node(n) => n,
            Item::Signal(n, _) => n,
        }
    }
}

impl fmt::Display for Item {
    /// Match the python repr: `'node'` for node ids (str repr),
    /// `('node', 'signal')` for signals (NodeSignal repr)
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Item::Node(n) => write!(f, "'{n}'"),
            Item::Signal(n, s) => write!(f, "('{n}', '{s}')"),
        }
    }
}

impl<'py> FromPyObject<'py> for Item {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(s) = ob.extract::<String>() {
            return Ok(Item::Node(s));
        }
        if let Ok((node, signal)) = ob.extract::<(String, String)>() {
            return Ok(Item::Signal(node, signal));
        }
        Err(PyTypeError::new_err(
            "graph items must be a node id string or a (node_id, signal) tuple",
        ))
    }
}

impl<'py> IntoPyObject<'py> for Item {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(match self {
            Item::Node(n) => n.into_pyobject(py)?.into_any(),
            // construct a real noob.types.NodeSignal when noob is importable,
            // a plain tuple (equal to it) when used standalone
            Item::Signal(n, s) => match crate::bridge::Bridge::get(py) {
                Ok(bridge) => bridge.node_signal.bind(py).call1((n, s))?,
                Err(_) => (n, s).into_pyobject(py)?.into_any(),
            },
        })
    }
}
