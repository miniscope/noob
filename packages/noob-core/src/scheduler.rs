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
    subepochs: FxHashMap<Epoch, FxIndexSet<Epoch>>,
    /// cache - sets of graph items that are downstream from the key node
    subgraphs: FxHashMap<u16, FxIndexSet<u16>>,
    /// cache - the set of graph items that in the subgraph of the first key,
    /// but *not* downstream of the second.
    /// used for expiring items in subepochs when they are closed in a parent epoch,
    /// see `done_subepochs`
    ///
    /// Stored as a vec because it is deduplicated on construction and passed directly
    /// into the sorter's `mark_expired`.
    exclusive_subgraphs: FxHashMap<(u16, u16), Vec<u16>>,
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
            subgraphs: FxHashMap::default(),
            exclusive_subgraphs: FxHashMap::default(),
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
            if epoch.segments().len() == 1 {
                self.init_graph(epoch.clone())?;
            } else {
                self.init_subgraph(epoch.clone())?;
            }
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
    fn init_graph(&mut self, epoch: Epoch) -> CoreResult<()> {
        let mut graph = self.template.clone();
        if epoch.root() == 0 || self.epoch_completed(&Epoch::from(epoch.root() - 1)) {
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
        let node_id = epoch.leaf().node;

        let mut subgraph = self.get_subgraph_template(node_id)?;
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
    fn get_subgraph_template(&mut self, node_id: u16) -> CoreResult<Sorter> {
        if let Some(template) = self.subgraph_templates.get(&node_id) {
            Ok(template.clone())
        } else {
            let node_name = match self.interner.resolve(node_id) {
                Item::Node(node_name) => node_name,
                Item::Signal(_, _) => {
                    return Err(CoreError::Value(
                        "Subgraphs can only be created by nodes".to_string(),
                    ))
                }
            };

            let downstream = downstream_nodes(&self.edges, node_name, &FxIndexSet::default());
            let nodes: IndexMap<String, NodeFlags> = downstream
                .iter()
                .copied()
                .filter(|n| self.nodes.contains_key(*n))
                .map(|n| (String::from(n), *self.nodes.get(n).unwrap()))
                .collect();
            let edges: Vec<EdgeRec> = self
                .edges
                .iter()
                .filter(|e| downstream.contains(e.target_node.as_str()))
                .cloned()
                .collect();
            let subgraph = Sorter::from_graph(&mut self.interner, &nodes, &edges)?;
            self.subgraph_templates.insert(node_id, subgraph.clone());
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
    pub fn is_active_at(&self, epoch: &Epoch) -> bool {
        self.epochs
            .get(epoch)
            .is_some_and(|sorter| sorter.is_active())
            || self
                .subepochs
                .get(epoch)
                .is_some_and(|subeps| subeps.iter().any(|subep| self.is_active_at(subep)))
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
        let epochs: Vec<&Epoch> = match self.subepochs.get(epoch) {
            Some(epochs) => {
                let mut epoch_vec: Vec<&Epoch> = epochs.iter().collect();
                epoch_vec.push(epoch);
                epoch_vec.sort();
                epoch_vec
            }
            None => vec![epoch],
        };

        epochs
            .into_iter()
            .filter_map(|e| {
                let sorter = self.epochs.get_mut(e)?;
                Some(
                    sorter
                        .get_ready(&self.interner)
                        .into_iter()
                        .map(|node| (e.clone(), node)),
                )
            })
            .flatten()
            .collect()
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

        match graph.done(&self.interner, &[item]) {
            Err(CoreError::AlreadyDone(_))
                if self.subepochs.get(epoch).is_some_and(|s| !s.is_empty()) => {}
            other => other?,
        }

        self.done_subepochs(epoch, item)?;

        let graph = self.epochs.get_mut(epoch).expect("Epoch was just added");

        if !self.interner.is_signal(item) && with_signals {
            if let Some(signals) = graph.signals.get(&item) {
                let signals: Vec<u16> = signals.difference(&graph.done).copied().collect();
                graph.done(&self.interner, &signals)?;
            }
        }

        for parent in epoch.parents() {
            // parents are only ever garbage collected when they are the root,
            // and the root is always active while any of its subepochs are.
            // so we always have parent graphs created (because adding the subepoch graph creates them)
            // and they haven't been removed (because this subepoch graph would have been removed)
            let parent_graph = self
                .epochs
                .get_mut(&parent)
                .expect("Subepoch parents should always be active while subepochs are");
            parent_graph.mark_expired(&[item], false);
        }

        // Eagerly add the next epoch whenever the source nodes in a root epoch are done -
        // for async/multi-epoch runners,
        // this allows nodes to run as soon as they are topologically available.
        if epoch.segments().len() == 1
            && self.source_nodes.contains(&item)
            && self.sources_finished(epoch)
        {
            let next = epoch + 1;
            if !self.epochs.contains_key(&next) && !self.epoch_completed(&next) {
                self.add_epoch_at(next)?;
            }
        }

        if !self.is_active_at(epoch) {
            return self.end_epoch(epoch.clone());
        }

        Ok(Vec::new())
    }

    /// Called when a node in a parent epoch is marked done -
    /// mark the node done in all subepochs,
    /// but ensure that nodes that are exclusively downstream of this node
    /// (i.e. no dependencies on nodes within the mapped subepoch)
    /// are removed from the graph.
    ///
    /// This is to support gather-like operations from non-gather nodes in 3rd party tubes:
    /// nodes downstream of both this node and other nodes in the subepoch run in subepochs,
    /// but nodes that are exclusively downstream of this node only run in the parent epoch
    fn done_subepochs(&mut self, epoch: &Epoch, item: u16) -> CoreResult<()> {
        let Some(subepochs) = self.subepochs.get(epoch) else {
            return Ok(());
        };
        if subepochs.is_empty() {
            return Ok(());
        };
        let node_part = self.interner.node_part(item);

        let our_subgraph = self
            .subgraphs
            .entry(node_part)
            .or_insert_with(|| Self::make_subgraph(&mut self.interner, &self.edges, node_part));

        for subepoch in subepochs.iter() {
            let Some(sorter) = self.epochs.get_mut(subepoch) else {
                continue;
            };
            if sorter.ran.contains(&item) || !sorter.info.contains_key(&item) {
                continue;
            }
            if sorter.done.contains(&item) {
                sorter.resurrect(&self.interner, &[item])?;
            }

            sorter.done(&self.interner, &[item])?;

            let leaf = subepoch.leaf().node;
            let exclusive_subgraph = self
                .exclusive_subgraphs
                .entry((node_part, leaf))
                .or_insert_with(|| {
                    Self::make_exclusive_subgraph(
                        &mut self.interner,
                        our_subgraph,
                        &self.edges,
                        node_part,
                        leaf,
                    )
                });

            sorter.mark_expired(exclusive_subgraph, true);
        }
        Ok(())
    }

    /// utility for done_subepochs
    /// create a u16'd version of the downstream nodes from a graph item
    fn make_subgraph(interner: &mut Interner, edges: &[EdgeRec], item: u16) -> FxIndexSet<u16> {
        let node_str = interner.resolve(item).node_id().to_owned();
        let subgraph = downstream_nodes(edges, &node_str, &FxIndexSet::default());
        subgraph.iter().map(|n| interner.intern_node(n)).collect()
    }

    /// utility for done_subepochs
    /// set(item_subgraph) - set(excluded_subgraph)
    /// finds the nodes that are in the penumbra of the subgraph (induced by item)
    /// but *not* in the penumbra of the excluded item.
    fn make_exclusive_subgraph(
        interner: &mut Interner,
        subgraph: &FxIndexSet<u16>,
        edges: &[EdgeRec],
        item: u16,
        excluded: u16,
    ) -> Vec<u16> {
        let node_str = interner.resolve(item).node_id().to_owned();
        let subep_name = interner.resolve(excluded).node_id().to_owned();
        let downstream = downstream_nodes(
            edges,
            &subep_name,
            &FxIndexSet::from_iter([node_str.as_str()]),
        );
        let downstream_ids: FxIndexSet<u16> =
            downstream.iter().map(|n| interner.intern_node(n)).collect();
        subgraph
            .difference(&downstream_ids)
            .filter(|n| **n != item)
            .copied()
            .collect()
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

        // if a node that *did not induce* the subepochs is expired,
        // it expires itself in subepochs.
        if let Some(subeps) = self.subepochs.get(epoch) {
            let node = self.interner.node_part(item);
            let subeps: Vec<Epoch> = subeps
                .iter()
                .filter(|ep| ep.leaf().node != node && self.epochs.contains_key(ep))
                .cloned()
                .collect();
            for subep in subeps {
                events.append(&mut self.expire(&subep, item, with_signals, unlock_optionals)?);
            }
        }

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
        let mut events: Vec<Epoch> = Vec::new();
        let next = &epoch + 1;
        if next.segments().len() == 1
            && !self.epochs.contains_key(&next)
            && !self.epoch_completed(&next)
        {
            self.add_epoch_at(next.clone())?;
        }

        // Mark this epoch done to unlock stateful nodes in successor epoch
        if epoch.segments().len() == 1 || self.epochs.contains_key(&next) {
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
        if epoch.segments().len() == 1 {
            self.epoch_log.insert(epoch.root());
            if self.epoch_log.len() > self.epoch_log_len as usize {
                self.epoch_log.pop_first();
            }

            if let Some(subeps) = self.subepochs.remove(&epoch) {
                for subep in subeps {
                    self.epochs.remove(&subep);
                }
            }
            self.epochs.remove(&epoch);
        } else {
            let parent = epoch
                .parent()
                .expect("Epochs with more than 1 segment always have a parent");
            if !self.is_active_at(&parent) {
                events.append(&mut self.end_epoch(parent)?);
            }
        }

        events.push(epoch);
        Ok(events)
    }

    fn sources_finished(&self, epoch: &Epoch) -> bool {
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
