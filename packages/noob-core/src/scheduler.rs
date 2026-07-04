use std::collections::{BTreeMap, BTreeSet};

use indexmap::IndexMap;

use crate::epoch::Epoch;
use crate::exceptions::{CoreError, CoreResult};
use crate::item::{Interner, ASSETS_NODE, INPUT_NODE, PREVIOUS_EPOCH};
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

    pub fn add_epoch(&mut self) -> Epoch {
        let this_epoch = Epoch::from(self.next_epoch);
        self.next_epoch += 1;
        self.init_graph(this_epoch.clone())
            .expect("Fresh epoch clones should only be Ok() or NotAdded if no stateful nodes are in the graph");
        this_epoch
    }

    pub fn add_epoch_at(&mut self, epoch: impl Into<Epoch>) -> CoreResult<Epoch> {
        let epoch = epoch.into();
        if self.epochs.contains_key(&epoch) {
            Err(CoreError::EpochExists(epoch))
        } else if self.epoch_log.contains(&epoch.root()) {
            Err(CoreError::EpochCompleted(epoch))
        } else {
            self.next_epoch = self.next_epoch.max(epoch.root() + 1);
            self.init_graph(epoch.clone())?;
            Ok(epoch)
        }
    }

    /// Clone the topo sorter, add it to the epochs map, and mark previous epoch if completed
    /// TODO: subgraphs for subepochs
    fn init_graph(&mut self, epoch: Epoch) -> CoreResult<()> {
        let mut graph = self.template.clone();
        if epoch.root() == 0 || self.epoch_log.contains(&(epoch.root() - 1)) {
            match graph.done(&self.interner, &[PREVIOUS_EPOCH]) {
                Ok(()) | Err(CoreError::NotAdded(_)) => {}
                Err(e) => return Err(e),
            }
        }

        self.epochs.insert(epoch, graph);
        Ok(())
    }

    /// Is the scheduler active in any epoch?
    pub fn is_active(&self) -> bool {
        self.epochs.values().any(|sorter| sorter.is_active())
    }

    /// Is the scheduler active in a specific epoch?
    /// TODO: Subepochs
    pub fn is_active_at(&self, epoch: &Epoch) -> bool {
        self.epochs
            .get(epoch)
            .is_some_and(|sorter| sorter.is_active())
    }

    fn get_ready(&mut self) -> Vec<(Epoch, u16)> {
        self.epochs
            .iter_mut()
            .flat_map(|(epoch, graph)| {
                graph
                    .get_ready(&self.interner)
                    .into_iter()
                    .map(|ready| (epoch.clone(), ready))
            })
            .collect()
    }

    fn get_ready_at(&mut self, epoch: &Epoch) -> Vec<(Epoch, u16)> {
        let graph = self.epochs.get_mut(&epoch);
        match graph {
            Some(graph) => graph
                .get_ready(&self.interner)
                .into_iter()
                .map(|ready| (epoch.clone(), ready))
                .collect(),
            None => Vec::new(),
        }
    }
}

#[cfg(test)]
#[path = "tests/scheduler.rs"]
mod tests;
