//! NOOB Core - Minimal High-Performance Rust Extensions
//!
//! Simple, working Rust acceleration for NOOB with P2P and blockchain

use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use dashmap::DashMap;
use blake3;
use serde::{Serialize, Deserialize};

// P2P and crypto modules
mod p2p_simple;

/// Fast event structure
#[pyclass]
#[derive(Clone)]
pub struct FastEvent {
    #[pyo3(get)]
    pub id: u64,
    #[pyo3(get)]
    pub node_id: String,
    #[pyo3(get)]
    pub signal: String,
    #[pyo3(get)]
    pub epoch: u64,
    #[pyo3(get)]
    pub timestamp: f64,
}

/// Fast concurrent event store
#[pyclass]
pub struct FastEventStore {
    events: Arc<DashMap<(String, String, u64), Vec<FastEvent>>>,
    counter: Arc<AtomicU64>,
}

#[pymethods]
impl FastEventStore {
    #[new]
    fn new() -> Self {
        FastEventStore {
            events: Arc::new(DashMap::new()),
            counter: Arc::new(AtomicU64::new(0)),
        }
    }

    fn add_event(
        &self,
        node_id: String,
        signal: String,
        epoch: u64,
        timestamp: f64,
    ) -> u64 {
        let id = self.counter.fetch_add(1, Ordering::Relaxed);

        let event = FastEvent {
            id,
            node_id: node_id.clone(),
            signal: signal.clone(),
            epoch,
            timestamp,
        };

        let key = (node_id, signal, epoch);
        self.events.entry(key).or_insert_with(Vec::new).push(event);

        id
    }

    fn get_events(&self, node_id: String, signal: String, epoch: u64) -> Vec<FastEvent> {
        let key = (node_id, signal, epoch);
        match self.events.get(&key) {
            Some(events) => events.clone(),
            None => Vec::new(),
        }
    }

    fn clear(&self) {
        self.events.clear();
    }

    fn event_count(&self) -> u64 {
        self.counter.load(Ordering::Relaxed)
    }
}

/// Fast scheduler
#[pyclass]
pub struct FastScheduler {
    dependencies: HashMap<String, Vec<String>>,
    in_degree: HashMap<String, usize>,
    completed: Vec<String>,
}

#[pymethods]
impl FastScheduler {
    #[new]
    fn new(dependencies: HashMap<String, Vec<String>>) -> Self {
        let mut in_degree = HashMap::new();

        for (node, deps) in &dependencies {
            in_degree.entry(node.clone()).or_insert(deps.len());
        }

        FastScheduler {
            dependencies,
            in_degree,
            completed: Vec::new(),
        }
    }

    fn get_ready_nodes(&self) -> Vec<String> {
        self.in_degree
            .iter()
            .filter(|(node, &degree)| degree == 0 && !self.completed.contains(node))
            .map(|(node, _)| node.clone())
            .collect()
    }

    fn mark_completed(&mut self, node_id: String) {
        self.completed.push(node_id.clone());

        for (target, deps) in &self.dependencies {
            if deps.contains(&node_id) {
                if let Some(degree) = self.in_degree.get_mut(target) {
                    if *degree > 0 {
                        *degree -= 1;
                    }
                }
            }
        }
    }

    fn reset(&mut self) {
        self.completed.clear();
        for (node, deps) in &self.dependencies {
            self.in_degree.insert(node.clone(), deps.len());
        }
    }

    fn is_complete(&self) -> bool {
        self.completed.len() == self.dependencies.len()
    }
}

#[pymodule]
fn noob_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FastEvent>()?;
    m.add_class::<FastEventStore>()?;
    m.add_class::<FastScheduler>()?;

    // P2P and blockchain components
    p2p_simple::register_p2p(m)?;

    m.add("__version__", "0.3.0")?;
    Ok(())
}
