use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};

use indexmap::{IndexMap, IndexSet};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple};

use crate::epoch::{fmt_epoch, get_ready_order, parent, parents, root_int, Ep, EpochArg, EpochKey};
// use crate::errors::{CoreError, CoreResult};
use crate::item::{Item};
use crate::sorter::{extract_nodes, EdgeRec, NodeFlags, Sorter};

#[pyclass(module = "noob_core")]
pub struct CoreScheduler {
    nodes: IndexMap<String, NodeFlags>,
    edges: Vec<EdgeRec>,
    source_nodes: Vec<String>,
    epochs: IndexMap<EpochKey, SharedSorter>,
    subepochs: HashMap<EpochKey, IndexSet<EpochKey>>,
}