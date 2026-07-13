use crate::epoch::{Epoch, EpochSegment};
use crate::event::UpdateEvent;
use crate::exceptions::CoreError;
use crate::item::{interner, Interner, Item, ItemID};
use crate::scheduler::Scheduler;
use crate::toposort::{EdgeRec, NodeFlags};
use crate::FxIndexMap;
use pyo3::exceptions::PyValueError;
use pyo3::import_exception;
use pyo3::prelude::*;

#[pyclass(name = "Scheduler")]
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

    fn update(&mut self, events: Vec<(EpochArg, String, String, bool)>) -> PyResult<Vec<PyEpoch>> {
        let interner = interner();
        let mut core_events = Vec::with_capacity(events.len());
        for (epoch, node_id, signal, no_event) in events {
            let epoch = epoch_from_python(&interner, epoch)?;
            let Some(node) = interner.get(&Item::Node(node_id.clone())) else {
                continue; // not in the graph — logged divergence vs python's epoch-creating no-op
            };
            let signal = interner.get(&Item::Signal(node_id, signal));
            core_events.push(UpdateEvent {
                epoch,
                node,
                signal,
                no_event,
            });
        }
        let ended = self.0.update(core_events)?;
        Ok(ended
            .iter()
            .map(|ep| epoch_to_python(&interner, ep))
            .collect())
    }

    fn add_epoch(&mut self) -> PyEpoch {
        let interner = interner();
        let ep = self.0.add_epoch();
        epoch_to_python(&interner, &ep)
    }

    fn add_epoch_at(&mut self, epoch: EpochArg) -> PyResult<PyEpoch> {
        let interner = interner();
        let ep = epoch_from_python(&interner, epoch)?;
        let ep = self.0.add_epoch_at(ep)?;
        Ok(epoch_to_python(&interner, &ep))
    }

    fn is_active(&self) -> bool {
        self.0.is_active()
    }

    fn is_active_at(&self, epoch: EpochArg) -> PyResult<bool> {
        let interner = interner();
        let ep = epoch_from_python(&interner, epoch)?;
        Ok(self.0.is_active_at(&ep))
    }

    fn epoch_completed(&self, epoch: EpochArg) -> PyResult<bool> {
        let interner = interner();
        let ep = epoch_from_python(&interner, epoch)?;
        Ok(self.0.epoch_completed(&ep))
    }

    fn first_active_epoch(&self) -> Option<PyEpoch> {
        let interner = interner();
        self.0
            .first_active_epoch()
            .map(|ep| epoch_to_python(&interner, &ep))
    }

    fn sources_finished(&self, epoch: EpochArg) -> PyResult<bool> {
        let interner = interner();
        let ep = epoch_from_python(&interner, epoch)?;
        Ok(self.0.sources_finished(&ep))
    }

    fn get_ready(&mut self) -> Vec<(PyEpoch, String)> {
        let interner = interner();
        let ready = self.0.get_ready();
        ready_to_python(&interner, ready)
    }

    fn get_ready_at(&mut self, epoch: EpochArg) -> PyResult<Vec<(PyEpoch, String)>> {
        let interner = interner();
        let epoch = epoch_from_python(&interner, epoch)?;
        let ready = self.0.get_ready_at(&epoch);
        Ok(ready_to_python(&interner, ready))
    }

    #[pyo3(signature = (epoch, node_id, signal=None, with_signals=true))]
    fn done(
        &mut self,
        epoch: EpochArg,
        node_id: String,
        signal: Option<String>,
        with_signals: bool,
    ) -> PyResult<Vec<PyEpoch>> {
        let interner = interner();
        let item = item_from_python(&interner, node_id, signal);
        let epoch = epoch_from_python(&interner, epoch)?;
        let res = self.0.done(&epoch, item, with_signals)?;
        let ended = res
            .iter()
            .map(|ep| epoch_to_python(&interner, ep))
            .collect();
        Ok(ended)
    }

    #[pyo3(signature = (epoch, node_id, signal=None, with_signals=true, unlock_optionals=true))]
    fn expire(
        &mut self,
        epoch: EpochArg,
        node_id: String,
        signal: Option<String>,
        with_signals: bool,
        unlock_optionals: bool,
    ) -> PyResult<Vec<PyEpoch>> {
        let interner = interner();
        let item = item_from_python(&interner, node_id, signal);
        let epoch = epoch_from_python(&interner, epoch)?;
        let res = self
            .0
            .expire(&epoch, item, with_signals, unlock_optionals)?;
        let ended = res
            .iter()
            .map(|ep| epoch_to_python(&interner, ep))
            .collect();
        Ok(ended)
    }

    fn end_epoch(&mut self, epoch: EpochArg) -> PyResult<Vec<PyEpoch>> {
        let interner = interner();
        let epoch = epoch_from_python(&interner, epoch)?;
        let res = self.0.end_epoch(epoch)?;
        Ok(res
            .iter()
            .map(|ep| epoch_to_python(&interner, ep))
            .collect())
    }
}

#[derive(FromPyObject)]
enum EpochArg {
    Root(u32),
    Path(Vec<(String, u32)>),
}

type PyEpoch = Vec<(String, u32)>;

fn epoch_from_python(interner: &Interner, arg: EpochArg) -> PyResult<Epoch> {
    match arg {
        EpochArg::Root(root) => Ok(Epoch::from(root)),
        EpochArg::Path(segments) => {
            let segments: Vec<EpochSegment> = segments
                .into_iter()
                .map(|(node, ep)| EpochSegment {
                    node: interner.get(&Item::Node(node)).unwrap(),
                    epoch: ep,
                })
                .collect();
            let ep = Epoch::try_from(segments)
                .expect("Nodes and signals must be added when the scheduler is created");
            Ok(ep)
        }
    }
}

fn epoch_to_python(interner: &Interner, epoch: &Epoch) -> PyEpoch {
    epoch
        .segments()
        .iter()
        .map(|seg| (interner.resolve(seg.node).node_id().to_owned(), seg.epoch))
        .collect()
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

fn ready_to_python(interner: &Interner, ready: Vec<(Epoch, ItemID)>) -> Vec<(PyEpoch, String)> {
    ready
        .iter()
        .map(|(epoch, node)| {
            (
                epoch_to_python(interner, epoch),
                interner.resolve(*node).node_id().to_owned(),
            )
        })
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
