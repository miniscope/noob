use std::collections::{BTreeMap, BTreeSet};

use crate::epoch::Epoch;
use crate::exceptions::{CoreError, CoreResult};
use crate::item::{Interner, Item, PREVIOUS_EPOCH};
use crate::toposort::{EdgeRec, NodeFlags, Sorter};
use crate::tube::downstream_nodes;
use crate::FxIndexSet;
use indexmap::IndexMap;
use rustc_hash::FxHashMap;

const DEFAULT_EPOCH_LOG_LEN: u32 = 1000;

pub struct Scheduler {
    nodes: IndexMap<String, NodeFlags>,
    edges: Vec<EdgeRec>,
    /// Mapping between string and int representation of node ids and signals.
    interner: Interner,

    /// A frozen initial-state topo sorter to copy from
    template: Sorter,
    subgraph_templates: FxHashMap<u16, Sorter>,
    source_nodes: FxIndexSet<u16>,

    epochs: BTreeMap<Epoch, Sorter>,
    epoch_log: BTreeSet<u32>,
    epoch_log_len: u32,

    next_epoch: u32,
    // TODO: the rest of the fields
    subepochs: FxHashMap<Epoch, FxIndexSet<Epoch>>,
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
            subgraph_templates: FxHashMap::default(),
            source_nodes,
            epochs: BTreeMap::new(),
            epoch_log: BTreeSet::new(),
            epoch_log_len: DEFAULT_EPOCH_LOG_LEN,
            next_epoch: 0,
            subepochs: FxHashMap::default(),
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

    pub fn add_subepoch(&mut self, epoch: impl Into<Epoch>) -> CoreResult<Epoch> {
        let epoch = epoch.into();
        if self.epochs.contains_key(&epoch) {
            Err(CoreError::EpochExists(epoch))
        } else if self.epoch_completed(&epoch) {
            Err(CoreError::EpochCompleted(epoch))
        } else {
            self.init_subgraph(epoch.clone())?;
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

    fn init_subgraph(&mut self, epoch: Epoch) -> CoreResult<()> {
        let Some(immediate_parent) = epoch.parent() else {
            return Err(CoreError::Value(format!(
                "Cannot create a subepoch for root epoch {epoch}"
            )));
        };
        let node_id = epoch.segments().last().unwrap().node;

        let mut subgraph = self.get_subgraph_template(&node_id)?;
        let parent = match self.epochs.get(&immediate_parent) {
            Some(parent) => parent,
            None => {
                if immediate_parent.segments().len() > 1 {
                    self.init_subgraph(immediate_parent.clone())?;
                } else {
                    self.init_graph(immediate_parent.clone())?;
                }
                self.epochs
                    .get(&immediate_parent)
                    .expect("Epoch was just created")
            }
        };

        // update the subgraph to match the parent state
        // mark any nodes that are completed in the parent as completed in the subepoch
        // EXCEPT don't expire the node that induced the subepoch or its signals -
        // we expect that the subepoch is typically created during an `update` call
        // where we'll be handling done or expiredness of the signals separately.

        let mut exclude_current = parent
            .signals
            .get(&node_id)
            .cloned()
            .unwrap_or(FxIndexSet::default());
        exclude_current.insert(node_id);
        let subgraph_keys: Vec<u16> = subgraph.info.keys().copied().collect();
        for parent_dep in subgraph_keys {
            if parent.ran.contains(&parent_dep) {
                subgraph.done(&self.interner, &[parent_dep])?;
            } else if parent.done.contains(&parent_dep) && !exclude_current.contains(&parent_dep) {
                subgraph.mark_expired(&[parent_dep], false);
            } else if parent.out.contains(&parent_dep) {
                subgraph.mark_out(&FxIndexSet::from_iter([parent_dep]));
            }
        }

        let done_in_parent = parent.done.contains(&node_id);

        for parent_ep in epoch.parents() {
            self.subepochs
                .entry(parent_ep)
                .or_default()
                .insert(epoch.clone());
        }

        self.epochs.insert(epoch, subgraph);

        // a node inducing subepochs expires the node in the (immediate) parent epoch
        if !done_in_parent {
            self.expire(&immediate_parent, node_id, false, false)?;
        }

        Ok(())
    }

    /// Get or make a cached subgraph template
    fn get_subgraph_template(&mut self, node_id: &u16) -> CoreResult<Sorter> {
        if let Some(template) = self.subgraph_templates.get(&node_id) {
            Ok(template.clone())
        } else {
            let node_name = match self.interner.resolve(*node_id) {
                Item::Node(node_name) => node_name,
                Item::Signal(_, _) => {
                    return Err(CoreError::Value(
                        "Subgraphs can only be created by nodes".parse().unwrap(),
                    ))
                }
            };

            let downstream = downstream_nodes(&self.edges, &node_name, &FxIndexSet::default());
            let nodes: IndexMap<String, NodeFlags> = downstream
                .iter()
                .copied()
                .filter(|n| self.nodes.contains_key(*n))
                .map(|n| (String::from(n), self.nodes.get(n).unwrap().clone()))
                .collect();
            let edges: Vec<EdgeRec> = self
                .edges
                .iter()
                .filter(|e| downstream.contains(e.target_node.as_str()))
                .cloned()
                .collect();
            let subgraph = Sorter::from_graph(&mut self.interner, &nodes, &edges)?;
            self.subgraph_templates.insert(*node_id, subgraph.clone());
            Ok(subgraph)
        }
    }

    pub fn iter_epoch(&mut self) -> EpochIter<'_> {
        let epoch = self
            .epochs
            .iter()
            .find(|(_, sorter)| sorter.is_active())
            .map(|(epoch, _)| epoch.clone())
            .unwrap_or_else(|| self.add_epoch());

        EpochIter {
            scheduler: self,
            epoch,
        }
    }

    pub fn iter_epoch_at(&mut self, epoch: impl Into<Epoch>) -> CoreResult<EpochIter<'_>> {
        let epoch = epoch.into();
        if !self.epochs.contains_key(&epoch) {
            self.add_epoch_at(epoch.clone())?;
        }
        Ok(EpochIter {
            scheduler: self,
            epoch,
        })
    }

    pub fn iter_ready(&mut self) -> ReadyIter<'_> {
        if !self.is_active() {
            self.add_epoch();
        }
        ReadyIter { scheduler: self }
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
        for parent in epoch.parents() {
            let parent_graph = self
                .epochs
                .get_mut(&parent)
                .expect("Subepoch parents should always be active while subepochs are");
            parent_graph.mark_expired(&[item], false);
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

pub struct EpochIter<'a> {
    scheduler: &'a mut Scheduler,
    epoch: Epoch,
}

impl EpochIter<'_> {
    pub fn done(&mut self, item: u16, with_signals: bool) -> CoreResult<Vec<Epoch>> {
        self.scheduler.done(&self.epoch, item, with_signals)
    }

    pub fn expire(
        &mut self,
        item: u16,
        with_signals: bool,
        unlock_optionals: bool,
    ) -> CoreResult<Vec<Epoch>> {
        self.scheduler
            .expire(&self.epoch, item, with_signals, unlock_optionals)
    }
}

impl Iterator for EpochIter<'_> {
    type Item = Vec<(Epoch, u16)>;
    fn next(&mut self) -> Option<Self::Item> {
        if !self.scheduler.is_active_at(&self.epoch) {
            None
        } else {
            Some(self.scheduler.get_ready_at(&self.epoch))
        }
    }
}

pub struct ReadyIter<'a> {
    scheduler: &'a mut Scheduler,
}

impl ReadyIter<'_> {
    pub fn done(&mut self, epoch: &Epoch, item: u16, with_signals: bool) -> CoreResult<Vec<Epoch>> {
        self.scheduler.done(epoch, item, with_signals)
    }

    pub fn expire(
        &mut self,
        epoch: &Epoch,
        item: u16,
        with_signals: bool,
        unlock_optionals: bool,
    ) -> CoreResult<Vec<Epoch>> {
        self.scheduler
            .expire(epoch, item, with_signals, unlock_optionals)
    }
}

impl Iterator for ReadyIter<'_> {
    type Item = Vec<(Epoch, u16)>;
    fn next(&mut self) -> Option<Self::Item> {
        let ready = self.scheduler.get_ready();
        if ready.is_empty() {
            None
        } else {
            Some(ready)
        }
    }
}

#[cfg(test)]
#[path = "tests/scheduler.rs"]
mod tests;
