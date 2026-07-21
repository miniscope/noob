use crate::FxIndexMap;
use crate::epoch::Epoch;
use crate::exceptions::CoreError;
use crate::item::{Interner, Item, ItemID, interner};
use crate::scheduler::Scheduler;
use crate::sorter::{EdgeRec, NodeFlags};
use pyo3::exceptions::PyValueError;
use pyo3::import_exception;
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{BTreeSet, HashSet};

/// PyO3 class that bridges between the pure-rust Scheduler and its python counterpart
/// Only those methods that do something beyond forwarding calls with string interning
/// have public docstrings.
/// For the rest, see either the Python or Rust scheduler.
/// (Methods being marked public is purely a documentation detail because the PyO3 macro makes them so anyway)
#[pyclass(name = "Scheduler", module = "noob_core._core")]
pub struct PyScheduler(Scheduler);

#[pymethods]
impl PyScheduler {
    #[new]
    fn new(
        nodes: Vec<(String, bool, Option<bool>)>,
        edges: Vec<(String, String, String, bool)>,
    ) -> PyResult<Self> {
        let nodes: FxIndexMap<String, NodeFlags> = nodes
            .into_iter()
            .map(|(name, enabled, stateful)| (name, NodeFlags { enabled, stateful }))
            .collect();
        let edges: Vec<EdgeRec> = edges
            .into_iter()
            .map(
                |(source_node, source_signal, target_node, required)| EdgeRec {
                    source_node,
                    source_signal,
                    target_node,
                    required,
                },
            )
            .collect();
        Ok(PyScheduler(Scheduler::from_graph(nodes, edges)?))
    }

    /// Accept recast events from the python scheduler,
    /// intern the strings to ints,
    /// filter the events to only those whose signals are in the graph,
    /// (e.g., not disabled, etc.)
    /// and pass to the internal update method.
    pub(crate) fn update(
        &mut self,
        events: Vec<(EpochArg, String, String, bool)>,
    ) -> PyResult<Vec<Epoch>> {
        let interner = interner();
        let mut core_events = Vec::with_capacity(events.len());
        for (epoch, node_id, signal, no_event) in events {
            // filter the events to only those that we want to process further with the relevant sorters
            // do this here, as soon as we can, to avoid any unnecessary work/iteration later.
            let Some(node) = interner
                .get(&Item::Node(node_id.clone()))
                .filter(|&node| self.0.graph_items.contains(&node))
            else {
                continue; // not in the graph
            };
            // some nodes are present in our graph without any dependencies
            // to keep them cheap, we include just them in the graph,
            // without modeling all the signals they could possibly have.
            // in those cases, even if we are aware of some signal,
            // we want to pass just the node with a `None` signal to mark it done
            let signal = interner
                .get(&Item::Signal(node_id, signal))
                .filter(|&sig| self.0.graph_items.contains(&sig));
            let epoch = epoch_from_python(epoch)?;
            core_events.push(UpdateEvent {
                epoch,
                node,
                signal,
                no_event,
            });
        }
        Ok(self.0.update(core_events)?)
    }

    fn add_epoch(&mut self) -> Epoch {
        self.0.add_epoch()
    }

    fn add_epoch_at(&mut self, epoch: EpochArg) -> PyResult<Epoch> {
        let ep = epoch_from_python(epoch)?;
        Ok(self.0.add_epoch_at(ep)?)
    }

    fn is_active(&self) -> bool {
        self.0.is_active()
    }

    fn is_active_at(&self, epoch: EpochArg) -> PyResult<bool> {
        let ep = epoch_from_python(epoch)?;
        Ok(self.0.is_active_at(&ep))
    }

    fn epoch_completed(&self, epoch: EpochArg) -> PyResult<bool> {
        let ep = epoch_from_python(epoch)?;
        Ok(self.0.epoch_completed(&ep))
    }

    fn first_active_epoch(&self) -> Option<Epoch> {
        self.0.first_active_epoch()
    }

    fn sources_finished(&self, epoch: EpochArg) -> PyResult<bool> {
        let ep = epoch_from_python(epoch)?;
        Ok(self.0.sources_finished(&ep))
    }

    fn get_ready(&mut self) -> Vec<(Epoch, String)> {
        let interner = interner();
        let ready = self.0.get_ready();
        ready_to_python(&interner, ready)
    }

    fn get_ready_at(&mut self, epoch: EpochArg) -> PyResult<Vec<(Epoch, String)>> {
        let interner = interner();
        let epoch = epoch_from_python(epoch)?;
        let ready = self.0.get_ready_at(&epoch);
        Ok(ready_to_python(&interner, ready))
    }

    /// Check if a node is ready in a given epoch without marking it as `out`
    ///
    /// Raises a NotAdded error if the node has not been previously added to the graph,
    /// rather than automatically interning:
    /// differentiates simply not being ready from not existing at all
    pub(crate) fn node_is_ready(
        &self,
        node: String,
        epoch: Epoch,
        subepochs: bool,
    ) -> PyResult<bool> {
        let interner = interner();
        let item = Item::Node(node);
        let Some(node_id) = interner.get(&item) else {
            return Err(CoreError::NotAdded(format!(
                "Node {item} was not added when creating the scheduler"
            ))
            .into());
        };
        Ok(self.0.node_is_ready(&node_id, &epoch, subepochs))
    }

    /// Check if a node has been either run or expired in the given epoch
    ///
    /// Similarly to node_is_ready, raises NotAddedError if node has not been previously added.
    pub(crate) fn node_is_done(&self, node: String, epoch: Epoch) -> PyResult<bool> {
        let interner = interner();
        let item = Item::Node(node);
        let Some(node_id) = interner.get(&item) else {
            return Err(CoreError::NotAdded(format!(
                "Node {item} was not added when creating the scheduler"
            ))
            .into());
        };
        Ok(self.0.node_is_done(&node_id, &epoch))
    }

    #[pyo3(signature = (epoch, node_id, signal=None, with_signals=true))]
    fn done(
        &mut self,
        epoch: EpochArg,
        node_id: String,
        signal: Option<String>,
        with_signals: bool,
    ) -> PyResult<Vec<Epoch>> {
        let interner = interner();
        let item = item_from_python(&interner, node_id, signal);
        let epoch = epoch_from_python(epoch)?;
        Ok(self.0.done(&epoch, item, with_signals)?)
    }

    #[pyo3(signature = (epoch, node_id, signal=None, with_signals=true, unlock_optionals=true))]
    fn expire(
        &mut self,
        epoch: EpochArg,
        node_id: String,
        signal: Option<String>,
        with_signals: bool,
        unlock_optionals: bool,
    ) -> PyResult<Vec<Epoch>> {
        let interner = interner();
        let item = item_from_python(&interner, node_id, signal);
        let epoch = epoch_from_python(epoch)?;
        Ok(self
            .0
            .expire(&epoch, item, with_signals, unlock_optionals)?)
    }

    fn end_epoch(&mut self, epoch: EpochArg) -> PyResult<Vec<Epoch>> {
        let epoch = epoch_from_python(epoch)?;
        Ok(self.0.end_epoch(epoch)?)
    }

    fn has_cycle(&self) -> bool {
        self.0.has_cycle()
    }

    fn generations(&self) -> Vec<Vec<String>> {
        let interner = interner();
        self.0
            .generations()
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|id| interner.resolve(*id).node_id().to_owned())
                    .collect()
            })
            .collect()
    }

    fn source_nodes(&self) -> Vec<String> {
        let interner = interner();
        self.0
            .source_nodes()
            .iter()
            .map(|id| interner.resolve(*id).node_id().to_owned())
            .collect()
    }

    fn upstream_nodes(&self, node: String) -> PyResult<FxHashSet<String>> {
        let interner = interner();
        let item = Item::Node(node);
        let Some(node_id) = interner.get(&item) else {
            return Err(CoreError::NotAdded(format!(
                "Node {item} was not added when the scheduler was created"
            ))
            .into());
        };
        Ok(self
            .0
            .upstream_nodes(node_id)?
            .iter()
            .map(|id| interner.resolve(*id).node_id().to_owned())
            .collect())
    }

    fn get_epoch_state(&self, epoch: Epoch) -> PyResult<PySorterState> {
        let state = self.0.get_epoch_state(&epoch)?;
        let interner = interner();
        Ok(PySorterState {
            ready: state
                .ready
                .iter()
                .map(|id| interner.resolve(*id))
                .cloned()
                .collect(),
            out: state
                .out
                .iter()
                .map(|id| interner.resolve(*id))
                .cloned()
                .collect(),
            done: state
                .done
                .iter()
                .map(|id| interner.resolve(*id))
                .cloned()
                .collect(),
            disabled: state
                .disabled
                .iter()
                .map(|id| interner.resolve(*id))
                .cloned()
                .collect(),
            ran: state
                .ran
                .iter()
                .map(|id| interner.resolve(*id))
                .cloned()
                .collect(),
            pending: state
                .pending
                .iter()
                .map(|id| interner.resolve(*id))
                .cloned()
                .collect(),
            npassedout: state.npassedout,
            nfinished: state.nfinished,
        })
    }

    #[getter(epoch_log)]
    fn epoch_log(&self) -> BTreeSet<u32> {
        self.0.epoch_log()
    }

    #[getter(subepochs)]
    fn subepochs(&self) -> FxHashMap<Epoch, HashSet<Epoch>> {
        self.0
            .subepochs()
            .iter()
            .map(|(epoch, subeps)| (epoch.clone(), HashSet::from_iter(subeps.iter().cloned())))
            .collect()
    }
}

/// Python-compatible counterpart of SorterState
#[derive(IntoPyObject)]
struct PySorterState {
    pub ready: FxHashSet<Item>,
    pub out: FxHashSet<Item>,
    pub done: FxHashSet<Item>,
    pub disabled: FxHashSet<Item>,
    pub ran: FxHashSet<Item>,
    pub pending: FxHashSet<Item>,
    pub npassedout: i64,
    pub nfinished: i64,
}

#[derive(FromPyObject)]
pub(crate) enum EpochArg {
    Handle(Py<Epoch>),
    Root(u32),
}

fn epoch_from_python(arg: EpochArg) -> PyResult<Epoch> {
    match arg {
        EpochArg::Root(root) => Ok(Epoch::from(root)),
        EpochArg::Handle(epoch) => Ok(epoch.get().clone()),
    }
}

fn item_from_python(interner: &Interner, node_id: String, signal: Option<String>) -> ItemID {
    let item = match signal {
        Some(signal) => Item::Signal(node_id, signal),
        None => Item::Node(node_id),
    };
    interner
        .get(&item)
        .expect("Nodes and signals must be added when the scheduler is created")
}

fn ready_to_python(interner: &Interner, ready: Vec<(Epoch, ItemID)>) -> Vec<(Epoch, String)> {
    ready
        .iter()
        .map(|(epoch, node)| (epoch.clone(), interner.resolve(*node).node_id().to_owned()))
        .collect()
}

import_exception!(noob_core.exceptions, AlreadyDoneError);
import_exception!(noob_core.exceptions, NotAddedError);
import_exception!(noob_core.exceptions, EpochExistsError);
import_exception!(noob_core.exceptions, EpochCompletedError);

impl From<CoreError> for PyErr {
    fn from(err: CoreError) -> PyErr {
        match err {
            CoreError::AlreadyDone(msg) => AlreadyDoneError::new_err(msg),
            CoreError::NotAdded(msg) => NotAddedError::new_err(msg),
            CoreError::EpochExists(msg) => EpochExistsError::new_err(msg.to_string()),
            CoreError::EpochCompleted(msg) => EpochCompletedError::new_err(msg.to_string()),
            other => PyValueError::new_err(other.to_string()),
        }
    }
}

/// A single event to update the scheduler state from emitted by a node.
/// After interning python strings to ints and extracting other string values.
pub struct UpdateEvent {
    pub epoch: Epoch,
    pub node: ItemID,
    pub signal: Option<ItemID>,
    pub no_event: bool,
}
