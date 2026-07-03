use std::collections::{BTreeMap, BTreeSet};

use indexmap::IndexMap;

use crate::epoch::Epoch;
use crate::exceptions::CoreResult;
use crate::item::Interner;
use crate::toposort::{EdgeRec, NodeFlags, Sorter};

pub struct Scheduler {
    nodes: IndexMap<String, NodeFlags>,
    edges: Vec<EdgeRec>,
    /// Mapping between string and int representation of node ids and signals.
    interner: Interner,

    /// A frozen initial-state topo sorter to copy from
    template: Sorter,

    epochs: BTreeMap<Epoch, Sorter>,
    epoch_log: BTreeSet<u32>,

    next_epoch: u32,
    // TODO: the rest of the fields
    // subepochs: HashMap<Epoch, IndexSet<Epoch>>,
    // source_nodes: Vec<String>,
}

impl Scheduler {
    pub fn from_graph(
        nodes: IndexMap<String, NodeFlags>,
        edges: Vec<EdgeRec>,
    ) -> CoreResult<Scheduler> {
        let mut interner = Interner::default();
        let template = Sorter::from_graph(&mut interner, &nodes, &edges)?;
        Ok(Scheduler {
            nodes,
            edges,
            interner,
            template,
            epochs: BTreeMap::new(),
            epoch_log: BTreeSet::new(),
            next_epoch: 0,
        })
    }
}

#[cfg(test)]
#[path = "tests/scheduler.rs"]
mod tests;
