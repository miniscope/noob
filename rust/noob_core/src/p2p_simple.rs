//! Simplified P2P Content-Addressed Task Distribution
//!
//! Lightweight P2P implementation without full libp2p complexity

use pyo3::prelude::*;
use blake3::Hasher;
use std::collections::HashMap;
use std::sync::Arc;
use dashmap::DashMap;
use serde::{Serialize, Deserialize};

/// Content ID using Blake3 hash
#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentId {
    #[pyo3(get)]
    pub hash: String,  // Hex-encoded hash
}

#[pymethods]
impl ContentId {
    #[new]
    fn new(data: Vec<u8>) -> Self {
        let hash = blake3::hash(&data);
        ContentId {
            hash: hash.to_hex().to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!("ContentId({}...)", &self.hash[..8])
    }

    fn __str__(&self) -> String {
        self.hash.clone()
    }
}

/// Task metadata for distributed processing
#[pyclass]
#[derive(Clone, Serialize, Deserialize)]
pub struct TaskMetadata {
    #[pyo3(get)]
    pub task_id: String,
    #[pyo3(get)]
    pub node_id: String,
    #[pyo3(get)]
    pub worker_address: String,  // Ethereum address
    #[pyo3(get)]
    pub status: String,  // pending, claimed, completed, verified
    #[pyo3(get)]
    pub result_hash: Option<String>,
    #[pyo3(get)]
    pub gas_paid: u64,
    #[pyo3(get)]
    pub timestamp: u64,
}

#[pymethods]
impl TaskMetadata {
    #[new]
    fn new(
        task_id: String,
        node_id: String,
        worker_address: String,
        gas_paid: u64,
    ) -> Self {
        TaskMetadata {
            task_id,
            node_id,
            worker_address,
            status: "pending".to_string(),
            result_hash: None,
            gas_paid,
            timestamp: current_timestamp(),
        }
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    #[staticmethod]
    fn from_json(json: String) -> PyResult<Self> {
        serde_json::from_str(&json)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
}

/// Content-addressed storage for tasks and results
#[pyclass]
pub struct ContentStore {
    data: Arc<DashMap<String, Vec<u8>>>,
}

#[pymethods]
impl ContentStore {
    #[new]
    fn new() -> Self {
        ContentStore {
            data: Arc::new(DashMap::new()),
        }
    }

    fn put(&self, data: Vec<u8>) -> ContentId {
        let cid = ContentId::new(data.clone());
        self.data.insert(cid.hash.clone(), data);
        cid
    }

    fn get(&self, cid: String) -> Option<Vec<u8>> {
        self.data.get(&cid).map(|entry| entry.clone())
    }

    fn contains(&self, cid: String) -> bool {
        self.data.contains_key(&cid)
    }

    fn size(&self) -> usize {
        self.data.len()
    }

    fn clear(&self) {
        self.data.clear();
    }
}

/// Task registry tracking all tasks
#[pyclass]
pub struct TaskRegistry {
    tasks: Arc<DashMap<String, TaskMetadata>>,
    by_status: Arc<DashMap<String, Vec<String>>>,
}

#[pymethods]
impl TaskRegistry {
    #[new]
    fn new() -> Self {
        TaskRegistry {
            tasks: Arc::new(DashMap::new()),
            by_status: Arc::new(DashMap::new()),
        }
    }

    fn register_task(&self, task: TaskMetadata) -> String {
        let task_id = task.task_id.clone();
        let status = task.status.clone();

        self.tasks.insert(task_id.clone(), task);

        // Add to status index
        self.by_status
            .entry(status)
            .or_insert_with(Vec::new)
            .push(task_id.clone());

        task_id
    }

    fn update_status(&self, task_id: String, new_status: String) -> PyResult<()> {
        if let Some(mut task) = self.tasks.get_mut(&task_id) {
            let old_status = task.status.clone();
            task.status = new_status.clone();
            task.timestamp = current_timestamp();

            // Update status index
            if let Some(mut tasks) = self.by_status.get_mut(&old_status) {
                tasks.retain(|id| id != &task_id);
            }

            self.by_status
                .entry(new_status)
                .or_insert_with(Vec::new)
                .push(task_id);

            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Task {} not found", task_id)
            ))
        }
    }

    fn set_result(&self, task_id: String, result_hash: String) -> PyResult<()> {
        if let Some(mut task) = self.tasks.get_mut(&task_id) {
            task.result_hash = Some(result_hash);
            task.timestamp = current_timestamp();
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Task {} not found", task_id)
            ))
        }
    }

    fn get_task(&self, task_id: String) -> Option<TaskMetadata> {
        self.tasks.get(&task_id).map(|entry| entry.clone())
    }

    fn get_by_status(&self, status: String) -> Vec<TaskMetadata> {
        if let Some(task_ids) = self.by_status.get(&status) {
            task_ids
                .iter()
                .filter_map(|id| self.tasks.get(id).map(|t| t.clone()))
                .collect()
        } else {
            Vec::new()
        }
    }

    fn count_by_status(&self, status: String) -> usize {
        self.by_status
            .get(&status)
            .map(|ids| ids.len())
            .unwrap_or(0)
    }

    fn total_tasks(&self) -> usize {
        self.tasks.len()
    }

    fn stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        for entry in self.by_status.iter() {
            stats.insert(entry.key().clone(), entry.value().len());
        }
        stats
    }
}

/// Simple P2P node for distributed task processing
#[pyclass]
pub struct P2PNode {
    #[pyo3(get)]
    pub node_id: String,
    #[pyo3(get)]
    pub ethereum_address: String,
    content_store: Arc<ContentStore>,
    task_registry: Arc<TaskRegistry>,
}

#[pymethods]
impl P2PNode {
    #[new]
    fn new(ethereum_address: String) -> Self {
        // Generate node ID from address
        let node_id = blake3::hash(ethereum_address.as_bytes())
            .to_hex()
            .to_string();

        P2PNode {
            node_id: node_id[..16].to_string(),  // Short ID
            ethereum_address,
            content_store: Arc::new(ContentStore::new()),
            task_registry: Arc::new(TaskRegistry::new()),
        }
    }

    fn submit_task(&self, node_id: String, data: Vec<u8>, gas_paid: u64) -> String {
        // Store task data
        let cid = self.content_store.put(data);

        // Create task metadata
        let task = TaskMetadata::new(
            cid.hash.clone(),
            node_id,
            self.ethereum_address.clone(),
            gas_paid,
        );

        self.task_registry.register_task(task)
    }

    fn claim_task(&self, task_id: String) -> PyResult<Option<Vec<u8>>> {
        // Update status
        self.task_registry.update_status(task_id.clone(), "claimed".to_string())?;

        // Get task data
        Ok(self.content_store.get(task_id))
    }

    fn submit_result(&self, task_id: String, result: Vec<u8>) -> PyResult<String> {
        // Store result
        let result_cid = self.content_store.put(result);

        // Update task
        self.task_registry.set_result(task_id.clone(), result_cid.hash.clone())?;
        self.task_registry.update_status(task_id, "completed".to_string())?;

        Ok(result_cid.hash)
    }

    fn verify_result(&self, task_id: String) -> PyResult<()> {
        self.task_registry.update_status(task_id, "verified".to_string())
    }

    fn get_result(&self, result_hash: String) -> Option<Vec<u8>> {
        self.content_store.get(result_hash)
    }

    fn get_pending_tasks(&self) -> Vec<TaskMetadata> {
        self.task_registry.get_by_status("pending".to_string())
    }

    fn get_task_status(&self, task_id: String) -> Option<String> {
        self.task_registry.get_task(task_id).map(|t| t.status)
    }

    fn get_stats(&self) -> HashMap<String, usize> {
        let mut stats = self.task_registry.stats();
        stats.insert("content_items".to_string(), self.content_store.size());
        stats
    }
}

fn current_timestamp() -> u64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Register P2P classes with Python module
pub fn register_p2p(m: &PyModule) -> PyResult<()> {
    m.add_class::<ContentId>()?;
    m.add_class::<ContentStore>()?;
    m.add_class::<TaskMetadata>()?;
    m.add_class::<TaskRegistry>()?;
    m.add_class::<P2PNode>()?;
    Ok(())
}
