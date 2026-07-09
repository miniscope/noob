use std::collections::{BTreeMap, BTreeSet};

use indexmap::IndexMap;

use crate::epoch::Epoch;
use crate::exceptions::{CoreError, CoreResult};
use crate::item::{Interner, PREVIOUS_EPOCH};
use crate::toposort::{EdgeRec, NodeFlags, Sorter};
use crate::FxIndexSet;

const DEFAULT_EPOCH_LOG_LEN: u32 = 1000;

pub struct Scheduler {
    nodes: IndexMap<String, NodeFlags>,
    edges: Vec<EdgeRec>,
    /// Mapping between string and int representation of node ids and signals.
    interner: Interner,

    /// A frozen initial-state topo sorter to copy from
    template: Sorter,
    source_nodes: FxIndexSet<u16>,

    epochs: BTreeMap<Epoch, Sorter>,
    epoch_log: BTreeSet<u32>,
    epoch_log_len: u32,

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
        let source_nodes = template.source_nodes();
        Ok(Scheduler {
            nodes,
            edges,
            interner,
            template,
            source_nodes,
            epochs: BTreeMap::new(),
            epoch_log: BTreeSet::new(),
            epoch_log_len: DEFAULT_EPOCH_LOG_LEN,
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
        } else if self.epoch_completed(&epoch) {
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
        let graph = self.epochs.get_mut(epoch);
        match graph {
            Some(graph) => graph
                .get_ready(&self.interner)
                .into_iter()
                .map(|ready| (epoch.clone(), ready))
                .collect(),
            None => Vec::new(),
        }
    }

    pub fn done(&mut self, epoch: &Epoch, item: u16, with_signals: bool) -> CoreResult<Vec<Epoch>> {
        if self.epoch_completed(epoch) {
            // TODO: debug logging
            return Ok(Vec::new());
        }

        if !self.epochs.contains_key(epoch) {
            self.add_epoch_at(epoch.clone())?;
        }
        let graph = self.epochs.get_mut(epoch).expect("Epoch was just added");

        // TODO: Suppress error if subepochs
        graph.done(&self.interner, &[item])?;

        if !self.interner.is_signal(item) && with_signals {
            if let Some(signals) = graph.signals.get(&item) {
                let signals: Vec<u16> = signals.difference(&graph.done).copied().collect();
                graph.done(&self.interner, &signals)?;
            }
        }

        // TODO: mark subepochs done
        let mut current = epoch.clone();
        while let Some(parent) = current.parent() {
            let parent_graph = self
                .epochs
                .get_mut(&parent)
                .expect("Subepoch parents should always be active while subepochs are");
            parent_graph.mark_expired(&[item], false);
            current = parent;
        }

        // TODO: general add operator for epoch to check next subepoch
        let next = Epoch::from(epoch.root() + 1);
        if epoch.segments().len() == 1
            && self.source_nodes.contains(&item)
            && !self.epochs.contains_key(&next)
            && !self.epoch_log.contains(&next.root())
            && self.sources_finished(epoch)
        {
            self.add_epoch_at(next)?;
        }

        if !self.is_active_at(epoch) {
            return self.end_epoch(epoch.clone());
        }

        Ok(Vec::new())
    }

    pub fn expire(
        &mut self,
        epoch: &Epoch,
        item: u16,
        with_signals: bool,
        unlock_optionals: bool,
    ) -> CoreResult<Vec<Epoch>> {
        let mut events: Vec<Epoch> = Vec::new();
        if self.epoch_completed(epoch) {
            // TODO: debug logging
            return Ok(Vec::new());
        }

        if !self.epochs.contains_key(epoch) {
            self.add_epoch_at(epoch.clone())?;
        }

        let graph = self.epochs.get_mut(epoch).expect("Epoch was just added");
        graph.mark_expired(&[item], unlock_optionals);

        if !self.interner.is_signal(item) && with_signals {
            if let Some(signals) = graph.signals.get(&item) {
                let signals = signals.clone();
                for signal in signals {
                    events.append(&mut self.expire(
                        epoch,
                        signal,
                        with_signals,
                        unlock_optionals,
                    )?);
                }
            }
        }

        // TODO: expire subepochs

        // end the epoch if it's over, don't double-tap in case expiring subepochs ended this epoch
        if !self.is_active_at(epoch) && !self.epoch_completed(epoch) {
            events.append(&mut self.end_epoch(epoch.clone())?);
        }

        Ok(events)
    }

    pub fn end_epoch(&mut self, epoch: impl Into<Epoch>) -> CoreResult<Vec<Epoch>> {
        let epoch = epoch.into();
        // signal this epoch has been completed to any successive epochs
        // we create root epoch graphs here -
        // the most common place to do so for tubes with stateful nodes.
        // epochs are created elsewhere when explicitly iterating epochs with `iter_epoch`
        // or when we receive out of order events e.g. in `update`
        // TODO: subepochs
        let mut events: Vec<Epoch> = Vec::new();
        let next = Epoch::from(epoch.root() + 1);
        if !self.epochs.contains_key(&next) && !self.epoch_log.contains(&next.root()) {
            self.add_epoch_at(next.clone())?;
        }

        // Mark this epoch done to unlock stateful nodes in successor epoch
        if self.epochs.contains_key(&next) {
            match self.done(&next, PREVIOUS_EPOCH, false) {
                Ok(mut ended) => events.append(&mut ended),
                Err(
                    CoreError::AlreadyDone(_)
                    | CoreError::NotAdded(_)
                    | CoreError::EpochCompleted(_),
                ) => {}
                Err(e) => return Err(e),
            }
        }

        // Log the epoch as completed
        // TODO: Subepochs
        self.epoch_log.insert(epoch.root());
        if self.epoch_log.len() > self.epoch_log_len as usize {
            self.epoch_log.pop_first();
        }
        self.epochs.remove(&epoch);
        events.push(epoch);
        Ok(events)
    }

    fn sources_finished(&self, epoch: &Epoch) -> bool {
        // TODO: use epoch_completed check to handle trimmed epoch_log
        if self.epoch_completed(epoch) {
            return true;
        }
        self.epochs
            .get(epoch)
            .is_some_and(|sorter| self.source_nodes.is_subset(&sorter.done))
    }

    pub fn epoch_completed(&self, epoch: &Epoch) -> bool {
        match self.epoch_log.first() {
            None => false,
            Some(first) => {
                (&epoch.root() < first || self.epoch_log.contains(&epoch.root()))
                    && !self.epochs.contains_key(epoch)
            }
        }
    }
}

#[cfg(test)]
#[path = "tests/scheduler.rs"]
mod tests;
