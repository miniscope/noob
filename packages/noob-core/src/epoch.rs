use crate::item::{ItemID, TUBE_NODE, interner, resolve_or_intern_node};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyType};
use std::fmt;
use std::iter::once;
use std::ops::{Add, Div, Sub};

/// A single layer of a subepoch
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct EpochSegment {
    /// The node that created this layer of the subepoch
    pub node: ItemID,
    /// The index of this subepoch within its layer
    pub epoch: u32,
}

impl From<(ItemID, u32)> for EpochSegment {
    fn from(segment: (ItemID, u32)) -> EpochSegment {
        EpochSegment {
            node: segment.0,
            epoch: segment.1,
        }
    }
}

/// Takes the interner arc lock - don't be string formatting while mutating interns
impl fmt::Display for EpochSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let interner = interner();
        let node = interner.resolve(self.node);
        write!(f, "('{}', {})", node.node_id(), self.epoch)
    }
}

/// The basic unit of event alignment in Noob:
/// Events emitted within the same epoch are passed to a node's slots together.
///
/// Epochs are hierarchical: by default, all events exist in an integer-valued root epoch.
/// However if a node like `Map` expands cardinality by emitting multiple events per event taken in,
/// Epochs grow "layers" of *subepochs* labeled with the ID of the node that emitted them.
#[pyclass(module = "noob_core._core", frozen, eq, ord, hash, str, from_py_object)]
#[derive(Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct Epoch {
    /// The common, tube-level epoch
    pub root: u32,
    /// Subepoch segments induced by cardinality expanding Maplike events.
    pub path: Vec<EpochSegment>,
}

impl Epoch {
    pub fn new(root: u32, path: impl IntoIterator<Item = impl Into<EpochSegment>>) -> Epoch {
        Epoch {
            root,
            path: path.into_iter().map(Into::into).collect(),
        }
    }

    pub fn is_root(&self) -> bool {
        self.path.is_empty()
    }

    /// 1 for root epochs, 1 + subepoch segments otherwise.
    pub fn n_segments(&self) -> usize {
        self.path.len() + 1
    }

    pub fn segments(&self) -> impl Iterator<Item = EpochSegment> + '_ {
        once(EpochSegment {
            node: TUBE_NODE,
            epoch: self.root,
        })
        .chain(self.path.iter().copied())
    }

    pub fn parent(&self) -> Option<Epoch> {
        self.path.split_last().map(|(_leaf, rest)| Epoch {
            root: self.root,
            path: rest.to_vec(),
        })
    }

    pub fn parents(&self) -> impl Iterator<Item = Epoch> + '_ {
        (0..self.path.len()).rev().map(|i| Epoch {
            root: self.root,
            path: self.path[..i].to_vec(),
        })
    }

    pub fn child(mut self, subep: EpochSegment) -> Epoch {
        self.path.push(subep);
        self
    }

    /// Create a collection of `n` sequential subepochs induced by events from the given node
    ///
    /// # Example
    ///
    /// ```
    /// Epoch::from(0).make_subepochs(6, 2)
    /// // vec![Epoch(0) / (6, 0), Epoch(0) / (6, 1)]
    /// ```
    pub fn make_subepochs(&self, node: ItemID, n: u32) -> Vec<Epoch> {
        (0..n)
            .map(|i| self.clone().child(EpochSegment { epoch: i, node }))
            .collect()
    }

    pub fn root(&self) -> u32 {
        self.root
    }

    /// The last available subepoch segment, if any.
    /// `None` for root epochs.
    pub fn leaf(&self) -> Option<&EpochSegment> {
        self.path.last()
    }

    fn checked_sub(mut self, rhs: u32) -> Option<Epoch> {
        let slot = match self.path.last_mut() {
            Some(seg) => &mut seg.epoch,
            None => &mut self.root,
        };
        *slot = slot.checked_sub(rhs)?;
        Some(self)
    }
}

#[pymethods]
impl Epoch {
    #[new]
    #[pyo3(signature = (root, path=Vec::new()))]
    fn py_new(root: u32, path: Vec<(String, u32)>) -> Self {
        Epoch {
            root,
            path: path
                .into_iter()
                .map(|(node_id, epoch)| EpochSegment {
                    node: resolve_or_intern_node(&node_id),
                    epoch,
                })
                .collect(),
        }
    }

    #[getter(root)]
    fn py_root(&self) -> u32 {
        self.root()
    }

    #[getter(parents)]
    fn py_parents(&self) -> Vec<Epoch> {
        self.parents().collect()
    }

    #[getter(parent)]
    fn py_parent(&self) -> Option<Epoch> {
        self.parent()
    }

    #[getter(root_epoch)]
    fn root_epoch(&self) -> Epoch {
        Epoch {
            root: self.root,
            path: Vec::new(),
        }
    }

    #[getter(is_root)]
    fn py_is_root(&self) -> bool {
        self.is_root()
    }

    #[getter(leaf)]
    fn py_leaf(&self) -> Option<(String, u32)> {
        let leaf = self.leaf()?;
        let interner = interner();
        Some((interner.resolve(leaf.node).node_id().to_owned(), leaf.epoch))
    }

    #[pyo3(name = "make_subepochs")]
    fn py_make_subepochs(&self, node: &str, n: u32) -> Vec<Epoch> {
        let id = resolve_or_intern_node(node);
        self.make_subepochs(id, n)
    }

    fn __repr__(&self) -> String {
        format!("{self}")
    }

    fn __truediv__(&self, rhs: (String, u32)) -> Epoch {
        self / (resolve_or_intern_node(&rhs.0), rhs.1)
    }

    fn __add__(&self, rhs: u32) -> Epoch {
        self + rhs
    }

    fn __sub__(&self, rhs: u32) -> PyResult<Epoch> {
        self.clone()
            .checked_sub(rhs)
            .ok_or_else(|| PyValueError::new_err("Negative epochs are invalid"))
    }

    fn __len__(&self) -> usize {
        self.n_segments()
    }

    fn __getitem__(&self, index: isize) -> PyResult<(String, u32)> {
        let len = self.n_segments();
        let idx = if index < 0 {
            index + len as isize
        } else {
            index
        };
        usize::try_from(idx)
            .ok()
            .and_then(|i| self.segments().nth(i))
            .map(|seg| (interner().resolve(seg.node).node_id().to_owned(), seg.epoch))
            .ok_or_else(|| {
                PyIndexError::new_err(format!(
                    "Index {index} out of range for epoch of length {len}"
                ))
            })
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> (Bound<'py, PyType>, (u32, PyEpochSegments)) {
        let interner = interner();
        (
            py.get_type::<Epoch>(),
            (
                self.root,
                self.path
                    .iter()
                    .map(|seg| (interner.resolve(seg.node).node_id().to_owned(), seg.epoch))
                    .collect(),
            ),
        )
    }

    fn to_wire(&self) -> WireEpoch {
        if self.is_root() {
            return WireEpoch::Root(self.root);
        }
        let interner = interner();
        WireEpoch::Path(
            once(WireItem::Root(self.root))
                .chain(self.path.iter().map(|seg| {
                    WireItem::Segment(interner.resolve(seg.node).node_id().to_owned(), seg.epoch)
                }))
                .collect(),
        )
    }

    #[staticmethod]
    fn from_wire(wire: WireEpoch) -> PyResult<Epoch> {
        match wire {
            WireEpoch::Root(root) => Ok(Epoch::from(root)),
            WireEpoch::Path(items) => {
                let mut items = items.into_iter();
                let Some(WireItem::Root(root)) = items.next() else {
                    return Err(PyValueError::new_err(
                        "wire epoch must start with an int root",
                    ));
                };
                let path = items
                    .map(|item| match item {
                        WireItem::Segment(name, epoch) => Ok(EpochSegment {
                            node: resolve_or_intern_node(&name),
                            epoch,
                        }),
                        WireItem::Root(n) => Err(PyValueError::new_err(format!(
                            "unexpected bare int {n} in path"
                        ))),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(Epoch { root, path })
            }
        }
    }

    #[classmethod]
    fn __get_pydantic_core_schema__<'py>(
        cls: &Bound<'py, PyType>,
        _source_type: &Bound<'py, PyAny>,
        _handler: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let core_schema = py.import("pydantic_core")?.getattr("core_schema")?;

        // handles to core_schema creators
        let union = core_schema.getattr("union_schema")?;
        let str = core_schema.getattr("str_schema")?;
        let int = core_schema.getattr("int_schema")?;
        let list = core_schema.getattr("list_schema")?;
        let tuple = core_schema.getattr("tuple_schema")?;

        let isinstance = core_schema.getattr("is_instance_schema")?.call1((cls,))?;

        // Coerce ints and lists of lists into ints or lists of tuples,
        // then call `from_wire`
        let validator = core_schema
            .getattr("no_info_after_validator_function")?
            .call1((
                cls.getattr("from_wire")?,
                union.call1((vec![
                    int.call0()?,
                    list.call1((union.call1((vec![
                        int.call0()?,
                        tuple.call1((vec![str.call0()?, int.call0()?],))?,
                    ],))?,))?,
                ],))?,
            ))?;
        let serialization = core_schema
            .getattr("plain_serializer_function_ser_schema")?
            .call(
                (cls.getattr("to_wire")?,),
                Some(&[("when_used", "json")].into_py_dict(py)?),
            )?;
        let kwargs = [("serialization", serialization)].into_py_dict(py)?;

        core_schema
            .getattr("union_schema")?
            .call((vec![isinstance, validator],), Some(&kwargs))
    }
}

type PyEpochSegments = Vec<(String, u32)>;

#[derive(FromPyObject, IntoPyObject)]
enum WireEpoch {
    Root(u32),
    Path(Vec<WireItem>),
}

#[derive(FromPyObject, IntoPyObject)]
enum WireItem {
    Root(u32),
    Segment(String, u32),
}

/// ```
/// Epoch::from(0)
/// ```
impl From<u32> for Epoch {
    fn from(number: u32) -> Self {
        Epoch {
            root: number,
            path: Vec::new(),
        }
    }
}

/// ```
/// Epoch::from(0) / (1, 2)
/// Epoch::from(0) / EpochSegment{ node: 1, epoch: 2 }
/// ```
impl<T: Into<EpochSegment>> Div<T> for Epoch {
    type Output = Epoch;
    fn div(self, rhs: T) -> Epoch {
        self.child(rhs.into())
    }
}

impl<T: Into<EpochSegment>> Div<T> for &Epoch {
    type Output = Epoch;
    fn div(self, rhs: T) -> Epoch {
        let epoch = self.clone();
        epoch.child(rhs.into())
    }
}

/// Increment the lowest layer of the epoch.
///
/// # Examples:
/// ```
/// Epoch::from(0) + 1
/// // Epoch(1)
/// (Epoch::from(0) / (2, 3)) + 1
/// // Epoch(1, (2, 4))
/// ```
impl Add<u32> for Epoch {
    type Output = Epoch;
    fn add(mut self, rhs: u32) -> Epoch {
        match self.path.last_mut() {
            Some(seg) => seg.epoch += rhs,
            None => self.root += rhs,
        }
        self
    }
}

impl Add<u32> for &Epoch {
    type Output = Epoch;
    fn add(self, rhs: u32) -> Epoch {
        self.clone() + rhs
    }
}

/// Decrement the lowest layer of the epoch.
///
/// # Examples:
/// ```
/// Epoch::from(1) - 1
/// // Epoch(0)
/// (Epoch::from(0) / (2, 3)) - 1
/// // Epoch(1, (2, 2))
/// ```
///
impl Sub<u32> for Epoch {
    type Output = Epoch;
    fn sub(self, rhs: u32) -> Epoch {
        self.checked_sub(rhs).expect("Negative epochs are invalid")
    }
}

impl Sub<u32> for &Epoch {
    type Output = Epoch;
    fn sub(self, rhs: u32) -> Epoch {
        self.clone() - rhs
    }
}

impl fmt::Display for Epoch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_root() {
            write!(f, "{}", self.root())
        } else {
            write!(f, "(")?;
            write!(f, "{}", self.root)?;
            for seg in self.path.iter() {
                write!(f, ", {seg}")?;
            }
            write!(f, ")")
        }
    }
}

#[cfg(test)]
#[path = "tests/epoch.rs"]
mod tests;
