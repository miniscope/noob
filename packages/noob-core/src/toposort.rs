use std::collections::HashMap;

use indexmap::{IndexMap, IndexSet};
use pyo3::prelude::*;
use pyo3::types::PyDict;

// use crate::exceptions::{CoreError, CoreResult};
// use crate::item::Interner;

#[derive(Clone, Debug, PartialEq, FromPyObject)]
pub struct EdgeRec {
    pub source_node: String,
    pub source_signal: String,
    pub target_node: String,
    pub required: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, FromPyObject)]
pub struct NodeFlags {
    pub enabled: bool,
    pub stateful: Option<bool>,
}

impl NodeFlags {
    /// Statefulness with the `None` (unresolved) default flattened to stateless
    pub fn is_stateful(&self) -> bool {
        self.stateful.unwrap_or(false)
    }
}

pub fn extract_nodes(nodes: &Bound<'_, PyDict>) -> PyResult<IndexMap<String, NodeFlags>> {
    nodes
        .iter()
        .map(|(node_id, spec)| PyResult::Ok((node_id.extract()?, spec.extract()?)))
        .collect()
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct NodeRec {
    pub nqueue: i64,
    pub successors: IndexSet<u32>,
    pub predecessors: IndexSet<u32>,
    pub optional_predecessors: IndexSet<u32>,
    pub optional_successors: IndexSet<u32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sorter {
    /// node item id -> signal items emitted by that node that the graph depends on
    pub signals: HashMap<u32, IndexSet<u32>>,
    /// insertion-ordered, mirrors `TopoSorter._node2info`
    pub info: IndexMap<u32, NodeRec>,
    pub ready: IndexSet<u32>,
    pub out: IndexSet<u32>,
    pub done: IndexSet<u32>,
    pub disabled: IndexSet<u32>,
    pub ran: IndexSet<u32>,
    pub npassedout: i64,
    pub nfinished: i64,
}