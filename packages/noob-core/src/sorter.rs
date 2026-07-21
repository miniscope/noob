use rustc_hash::{FxHashMap, FxHashSet};

use crate::exceptions::{CoreError, CoreResult};
use crate::item::{ASSETS_NODE, INPUT_NODE, Interner, Item, ItemID, PREVIOUS_EPOCH};
use crate::{FxIndexMap, FxIndexSet};

/// The fields of `noob.edge.Edge` the sorter cares about.
/// The boundary layer is responsible for extracting these from python.
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeRec {
    pub source_node: String,
    pub source_signal: String,
    pub target_node: String,
    pub required: bool,
}

impl From<(&str, &str, &str, bool)> for EdgeRec {
    fn from(value: (&str, &str, &str, bool)) -> Self {
        EdgeRec {
            source_node: value.0.to_string(),
            source_signal: value.1.to_string(),
            target_node: value.2.to_string(),
            required: value.3,
        }
    }
}
/// Default edge to required
impl From<(&str, &str, &str)> for EdgeRec {
    fn from(value: (&str, &str, &str)) -> Self {
        EdgeRec {
            source_node: value.0.to_string(),
            source_signal: value.1.to_string(),
            target_node: value.2.to_string(),
            required: true,
        }
    }
}

/// The fields of `noob.node.NodeSpecification` the sorter cares about.
/// `stateful` is `bool | None` in python - `None` (unresolved) is treated
/// as stateless, see [`NodeFlags::is_stateful`].
#[derive(Clone, Copy, Debug, PartialEq)]
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

#[derive(Clone, Debug, Default, PartialEq)]
pub struct NodeRec {
    pub nqueue: i64,
    pub successors: FxIndexSet<ItemID>,
    pub predecessors: FxIndexSet<ItemID>,
    pub optional_predecessors: FxIndexSet<ItemID>,
    pub optional_successors: FxIndexSet<ItemID>,
}

/// Port of `noob.toposort.TopoSorter`, operating on interned item ids.
///
/// Uses `IndexSet`/`IndexMap` rather than the std hash containers
/// because their iteration is deterministic (reproducible scheduling and
/// directly comparable test output, where std's per-instance random hash
/// seeds are not) and iterates a dense vec rather than sparse hash buckets.
/// These sets are iterated constantly.
// TODO: interned ids are dense, so the endgame is Vec/bitset storage
// indexed by id, with no hashing at all. Benchmark once the port is correct.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sorter {
    /// node item id -> signal items emitted by that node that the graph depends on
    pub signals: FxHashMap<ItemID, FxIndexSet<ItemID>>,
    /// mirrors `TopoSorter._node2info`
    pub info: FxIndexMap<ItemID, NodeRec>,
    pub ready: FxIndexSet<ItemID>,
    pub out: FxIndexSet<ItemID>,
    pub done: FxIndexSet<ItemID>,
    pub disabled: FxIndexSet<ItemID>,
    pub ran: FxIndexSet<ItemID>,
    pub npassedout: i64,
    pub nfinished: i64,
}

/// Independent, cloned state of the sorter to be used when debugging
#[derive(Debug)]
pub struct SorterState {
    pub ready: FxHashSet<ItemID>,
    pub out: FxHashSet<ItemID>,
    pub done: FxHashSet<ItemID>,
    pub disabled: FxHashSet<ItemID>,
    pub ran: FxHashSet<ItemID>,
    pub pending: FxHashSet<ItemID>,
    pub npassedout: i64,
    pub nfinished: i64,
}

impl Sorter {
    /// Port of `TopoSorter.__init__` from a node map and edge list
    pub fn from_graph(
        interner: &mut Interner,
        nodes: &FxIndexMap<String, NodeFlags>,
        edges: &[EdgeRec],
    ) -> CoreResult<Sorter> {
        let mut sorter = Sorter::default();
        // filter on disabled rather than enabled nodes: edges may reference
        // nodes we have no specification for, which should pass through
        for (node_id, flags) in nodes {
            if !flags.enabled {
                sorter.disabled.insert(interner.intern_node(node_id));
            }
        }
        for edge in edges {
            let target = interner.intern_node(&edge.target_node);
            if sorter.disabled.contains(&target) {
                continue;
            }
            let signal = interner.intern_signal(&edge.source_node, &edge.source_signal);
            sorter.add(interner, target, &[signal], edge.required)?;
        }
        for (node_id, flags) in nodes {
            let id = interner.intern_node(node_id);
            // add enabled nodes that have no edges
            if flags.enabled && !sorter.info.contains_key(&id) {
                sorter.add(interner, id, &[], true)?;
            }
            // stateful nodes wait on their own previous epoch.
            if flags.enabled && flags.is_stateful() {
                sorter.add(interner, id, &[PREVIOUS_EPOCH], true)?;
            }
        }
        Ok(sorter)
    }

    pub fn get_nodeinfo(&mut self, id: ItemID) -> &mut NodeRec {
        self.info.entry(id).or_default()
    }

    pub fn mark_ready(&mut self, nodes: &[ItemID]) {
        for n in nodes {
            self.ready.insert(*n);
        }
    }

    pub fn add(
        &mut self,
        interner: &mut Interner,
        node: ItemID,
        predecessors: &[ItemID],
        required: bool,
    ) -> CoreResult<()> {
        // refuse to add nodes that are out / done
        let mut reasons: Vec<&str> = Vec::new();
        if self.out.contains(&node) {
            reasons.push("already out");
        }
        if self.done.contains(&node) {
            reasons.push("already done");
        }
        if !reasons.is_empty() {
            return Err(CoreError::Value(format!(
                "{} cannot be added: {}",
                interner.resolve(node),
                reasons.join(", ")
            )));
        }

        // create the predecessor -> node edges,
        // filtering predecessors to those that are newly being created
        let mut new_predecessors: Vec<ItemID> = Vec::new();
        for &pred in predecessors {
            if self.get_nodeinfo(pred).successors.contains(&node) {
                continue;
            }
            new_predecessors.push(pred);
            self.get_nodeinfo(pred).successors.insert(node);

            if pred != PREVIOUS_EPOCH && interner.is_signal(pred) {
                // (node, signal) predecessors must always depend on their node
                let pred_node = interner.node_part(pred);
                self.signals.entry(pred_node).or_default().insert(pred);
                self.add(interner, pred, &[pred_node], true)?;
            }

            // re-read after the recursive add: python holds a reference to
            // the info object, which the recursion may have mutated
            let nqueue = self.get_nodeinfo(pred).nqueue;
            if nqueue == 0
                && !self.out.contains(&pred)
                && !self.done.contains(&pred)
                && !self.disabled.contains(&pred)
            {
                self.mark_ready(&[pred]);
            }
        }

        // create the node -> predecessor edges
        let rec = self.get_nodeinfo(node);
        for &p in &new_predecessors {
            rec.predecessors.insert(p);
        }
        // note: python passes *all* given predecessors here, not just new ones
        self.update_optionals(interner, node, predecessors, required);

        let ndone_predecessors = new_predecessors
            .iter()
            .filter(|p| self.done.contains(*p))
            .count() as i64;
        let rec = self.get_nodeinfo(node);
        rec.nqueue += new_predecessors.len() as i64 - ndone_predecessors;
        let nqueue = rec.nqueue;
        if nqueue == 0 {
            self.mark_ready(&[node]);
        } else {
            // in case the node is added multiple times
            self.ready.swap_remove(&node);
        }
        Ok(())
    }

    fn update_optionals(
        &mut self,
        interner: &Interner,
        node: ItemID,
        predecessors: &[ItemID],
        required: bool,
    ) {
        if interner.is_signal(node) {
            return;
        }
        let info = self.get_nodeinfo(node);
        if required {
            predecessors.iter().for_each(|p| {
                info.optional_predecessors.swap_remove(p);
            });
        } else {
            predecessors.iter().for_each(|p| {
                info.optional_predecessors.insert(*p);
            });
        }

        let mut to_visit: FxIndexSet<ItemID> = info.successors.clone();
        let mut new_successors: Vec<ItemID> = Vec::new();
        let mut seen: FxIndexSet<ItemID> = FxIndexSet::default();
        while let Some(current) = to_visit.pop() {
            let current_info = self.get_nodeinfo(current);
            let successors = current_info.successors.clone();
            for next_successor in successors {
                let next_info = self.get_nodeinfo(next_successor);
                if !interner.is_signal(next_successor)
                    && next_info.optional_predecessors.contains(&current)
                {
                    new_successors.push(next_successor);
                } else {
                    to_visit.extend(next_info.successors.difference(&seen));
                    seen.extend(next_info.successors.iter().copied());
                }
            }
        }
        self.get_nodeinfo(node)
            .optional_successors
            .extend(new_successors);

        // update upstream - since optional/non-optional can overlap,
        // and we can add required/optional out of order with overwriting,
        // we do this in two passes - first clearing all optionals and re-adding
        // first pass - remove optionals
        let info = self.get_nodeinfo(node);
        let mut to_visit: FxIndexSet<ItemID> = info
            .predecessors
            .difference(&info.optional_predecessors)
            .copied()
            .collect();
        let mut seen: FxIndexSet<ItemID> = FxIndexSet::default();
        while let Some(current) = to_visit.pop() {
            let current_info = self.get_nodeinfo(current);
            current_info.optional_successors.swap_remove(&node);
            to_visit.extend(
                current_info
                    .predecessors
                    .difference(&current_info.optional_predecessors)
                    .copied()
                    .filter(|p| !seen.contains(p)),
            );
            seen.extend(current_info.predecessors.iter().copied());
        }

        // second pass - re-add optionals
        let info = self.get_nodeinfo(node);
        let mut to_visit: FxIndexSet<ItemID> = info.optional_predecessors.clone();
        let mut seen: FxIndexSet<ItemID> = FxIndexSet::default();
        while let Some(current) = to_visit.pop() {
            let current_info = self.get_nodeinfo(current);
            if interner.is_signal(current) {
                current_info.optional_successors.insert(node);
            }
            if current_info.optional_predecessors.is_empty() {
                to_visit.extend(current_info.predecessors.difference(&seen));
                seen.extend(current_info.predecessors.clone());
            }
        }
    }

    pub fn mark_out(&mut self, nodes: &FxIndexSet<ItemID>) {
        nodes.iter().for_each(|n| {
            self.ready.swap_remove(n);
            self.out.insert(*n);
        });
        self.npassedout += nodes.len() as i64;
    }

    pub fn get_ready(&mut self, interner: &Interner) -> Vec<ItemID> {
        let ready: Vec<ItemID> = self
            .ready
            .iter()
            .copied()
            .filter(|n| !interner.is_signal(*n))
            .collect();
        let mut to_mark_out: FxIndexSet<ItemID> = ready.iter().copied().collect();
        for node in &ready {
            if let Some(sigs) = self.signals.get(node) {
                to_mark_out.extend(sigs);
            }
        }
        self.mark_out(&to_mark_out);
        ready
    }

    pub fn is_active(&self) -> bool {
        self.nfinished < self.npassedout || !self.ready.is_empty()
    }

    fn expire_node(&mut self, node: ItemID) -> bool {
        if self.done.contains(&node) {
            return false;
        }

        self.nfinished += 1;
        self.done.insert(node);
        self.ready.swap_remove(&node);
        if !self.out.swap_remove(&node) {
            self.npassedout += 1;
        }

        true
    }

    pub fn mark_expired(&mut self, nodes: &[ItemID], unlock_optionals: bool) {
        let mut newly_expired: Vec<ItemID> = Vec::with_capacity(nodes.len());
        for &node in nodes {
            if self.expire_node(node) {
                newly_expired.push(node);
            }
            // clear any immediate successors from ready, if they already were added
            if let Some(info) = self.info.get(&node) {
                self.ready.retain(|x| !info.successors.contains(x));
            }
        }
        if !unlock_optionals {
            return;
        }

        for node in newly_expired {
            let successors = self.optional_successors_of(node);
            for successor in successors {
                let successor_info = self.get_nodeinfo(successor);
                successor_info.nqueue -= 1;
                if successor_info.nqueue == 0
                    && !self.done.contains(&successor)
                    && !self.out.contains(&successor)
                {
                    if self.disabled.contains(&successor) {
                        self.expire_node(successor);
                    } else {
                        self.mark_ready(&[successor]);
                    }
                }
            }
        }
    }

    pub fn done(&mut self, interner: &Interner, nodes: &[ItemID]) -> CoreResult<()> {
        // TODO: Give the errors the formatting logic and just pass Vec<ItemID> without resolving
        let already_done: Vec<&Item> = nodes
            .iter()
            .copied()
            .filter(|n| self.done.contains(n))
            .map(|n| interner.resolve(n))
            .collect();
        if !already_done.is_empty() {
            return Err(CoreError::AlreadyDone(format!(
                "node(s) {already_done:?} were already marked done"
            )));
        }
        let not_added: Vec<&Item> = nodes
            .iter()
            .copied()
            .filter(|n| !self.info.contains_key(n))
            .map(|n| interner.resolve(n))
            .collect();
        if !not_added.is_empty() {
            return Err(CoreError::NotAdded(format!(
                "node(s) {not_added:?} were not added using add()"
            )));
        }

        let mut newly_done: Vec<ItemID> = Vec::with_capacity(nodes.len());
        for &node in nodes {
            if self.expire_node(node) {
                newly_done.push(node);
                self.ran.insert(node);
            }
        }

        for node in newly_done {
            let successors = self.successors_of(node);
            for successor in successors {
                if self.done.contains(&successor) || self.out.contains(&successor) {
                    continue;
                }
                let successor_info = self.get_nodeinfo(successor);
                successor_info.nqueue -= 1;
                if successor_info.nqueue == 0 {
                    if self.disabled.contains(&successor) {
                        self.mark_expired(&[successor], true);
                    } else {
                        self.mark_ready(&[successor]);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn resurrect(&mut self, interner: &Interner, nodes: &[ItemID]) -> CoreResult<()> {
        let already_ran: Vec<&Item> = nodes
            .iter()
            .copied()
            .filter(|n| self.ran.contains(n))
            .map(|n| interner.resolve(n))
            .collect();
        if !already_ran.is_empty() {
            return Err(CoreError::AlreadyDone(format!(
                "node(s) {already_ran:?} were marked done, not expired! can only resurrect expired nodes."
            )));
        }
        for node in nodes {
            if self.disabled.contains(node) {
                continue;
            }
            if self.done.swap_remove(node) {
                self.nfinished -= 1;
                self.npassedout -= 1;
                if let Some(info) = self.info.get(node)
                    && info.nqueue == 0
                {
                    self.mark_ready(&[*node]);
                }
            }
        }
        Ok(())
    }

    fn successors_of(&mut self, node: ItemID) -> Vec<ItemID> {
        self.get_nodeinfo(node).successors.iter().copied().collect()
    }

    fn optional_successors_of(&mut self, node: ItemID) -> Vec<ItemID> {
        self.get_nodeinfo(node)
            .optional_successors
            .iter()
            .copied()
            .collect()
    }

    /// Nodes within the graph that have no dependencies (except PREVIOUS_EPOCH)
    pub fn source_nodes(&self) -> FxIndexSet<ItemID> {
        let ignore = FxIndexSet::from_iter([PREVIOUS_EPOCH]);
        self.info
            .iter()
            .filter(|&(&id, info)| {
                info.predecessors.is_subset(&ignore)
                    && !matches!(id, PREVIOUS_EPOCH | INPUT_NODE | ASSETS_NODE)
                    && !self.disabled.contains(&id)
            })
            .map(|(id, _)| *id)
            .collect()
    }

    pub fn find_cycle(&self) -> Option<Vec<ItemID>> {
        let mut colors: FxHashMap<ItemID, Color> = FxHashMap::default();
        let mut path: Vec<ItemID> = Vec::new();
        for &start in self.info.keys() {
            if colors.contains_key(&start) {
                continue;
            }
            if let Some(cycle) = self.dfs(start, &mut colors, &mut path) {
                return Some(cycle);
            }
        }
        None
    }

    /// One depth-first descent from `node`. `path` holds the gray nodes in
    /// descent order; a node's `Gray(i)` color records its position in it.
    fn dfs(
        &self,
        node: ItemID,
        colors: &mut FxHashMap<ItemID, Color>,
        path: &mut Vec<ItemID>,
    ) -> Option<Vec<ItemID>> {
        colors.insert(node, Color::Gray(path.len()));
        path.push(node);

        for &successor in &self.info[&node].successors {
            match colors.get(&successor) {
                // on our own descent path: the cycle is everything from the
                // successor's position down to us, closed by the successor
                Some(&Color::Gray(i)) => {
                    let mut cycle = path[i..].to_vec();
                    cycle.push(successor);
                    return Some(cycle);
                }
                // fully explored, known cycle-free
                Some(Color::Black) => {}
                // unvisited: descend, and unwind immediately on a find
                None => {
                    if let Some(cycle) = self.dfs(successor, colors, path) {
                        return Some(cycle);
                    }
                }
            }
        }

        path.pop();
        colors.insert(node, Color::Black);
        None
    }

    /// Get a cloned version of the internal sorter state
    /// This is an expensive method, since it clones... everything
    /// To be used when debugging
    pub fn clone_state(&self) -> SorterState {
        let ready = FxHashSet::from_iter(self.ready.iter().copied());
        let out = FxHashSet::from_iter(self.out.iter().copied());
        let done = FxHashSet::from_iter(self.done.iter().copied());
        let ran = FxHashSet::from_iter(self.ran.iter().copied());
        let disabled = FxHashSet::from_iter(self.disabled.iter().copied());
        let pending: FxHashSet<ItemID> = self
            .info
            .keys()
            .filter(|id| {
                !ready.contains(id)
                    && !out.contains(id)
                    && !done.contains(id)
                    && !ran.contains(id)
                    && !disabled.contains(id)
            })
            .copied()
            .collect();
        SorterState {
            ready,
            out,
            done,
            ran,
            disabled,
            pending,
            npassedout: self.npassedout,
            nfinished: self.nfinished,
        }
    }
}

/// graph coloring for cycle detection: a node absent from the color map
/// is unvisited ("white" in the classic three-color scheme)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Color {
    /// on the current descent path, at this position in it
    Gray(usize),
    /// fully explored and known to be cycle-free
    Black,
}

#[cfg(test)]
#[path = "tests/sorter.rs"]
mod tests;
