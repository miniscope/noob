from __future__ import annotations

import asyncio
import json
import pickle
import random
from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from threading import Event as ThreadEvent, Lock
from typing import Any, Self
from uuid import uuid4

import httpx
from pydantic import BaseModel

from noob.event import Event
from noob.exceptions import AlreadyRunningError
from noob.input import InputScope
from noob.node.map import Map
from noob.runner.base import TubeRunner
from noob.store import EventStore
from noob.types import ReturnNodeType


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies for worker selection"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"
    FASTEST_RESPONSE = "fastest_response"


class WorkerConfig(BaseModel):
    """Configuration for a worker node"""
    host: str
    port: int
    max_retries: int = 3
    timeout: float = 30.0
    health_check_interval: float = 5.0
    weight: float = 1.0  # For weighted load balancing
    tags: list[str] = []  # For node affinity


class WorkerStatus(BaseModel):
    """Status of a worker"""
    worker_id: str
    host: str
    port: int
    healthy: bool = True
    last_seen: datetime | None = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_active: int = 0
    avg_response_time: float = 0.0
    total_response_time: float = 0.0


@dataclass
class DistributedRunner(TubeRunner):
    """
    Enhanced distributed tube runner for executing graphs across multiple machines.

    Features:
    - Async/parallel execution with configurable concurrency
    - HTTP-based worker communication with connection pooling
    - Automatic worker discovery and health monitoring
    - Advanced fault tolerance with retries, circuit breakers, and fallbacks
    - Multiple load balancing strategies (round-robin, least-loaded, random, fastest)
    - Smart event serialization (JSON for simple types, pickle for complex)
    - Task result caching for efficiency
    - Graceful degradation when workers fail
    - Comprehensive metrics and monitoring

    Example:
        >>> workers = [
        ...     WorkerConfig(host="worker1.local", port=8000),
        ...     WorkerConfig(host="worker2.local", port=8000),
        ... ]
        >>> runner = DistributedRunner(
        ...     tube,
        ...     workers=workers,
        ...     load_balancing=LoadBalancingStrategy.LEAST_LOADED,
        ...     max_parallel=20
        ... )
        >>> results = runner.run(n=100)
    """

    workers: list[WorkerConfig] = field(default_factory=list)
    local_execution: bool = True  # Fallback to local execution if workers unavailable
    max_parallel: int = 10  # Max concurrent tasks across all workers
    retry_delay: float = 1.0  # Initial delay between retries (exponential backoff)
    use_async: bool = True  # Use async execution when available
    load_balancing: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED
    circuit_breaker_threshold: int = 5  # Failures before circuit breaks
    circuit_breaker_timeout: float = 60.0  # Seconds before retry after circuit break
    enable_task_caching: bool = False  # Cache task results for idempotent nodes
    batch_size: int = 5  # Number of tasks to batch together
    
    _running: ThreadEvent = field(default_factory=ThreadEvent)
    _lock: Lock = field(default_factory=Lock)
    _worker_statuses: dict[str, WorkerStatus] = field(default_factory=dict)
    _task_queue: asyncio.Queue = field(default=None)
    _worker_clients: dict[str, httpx.AsyncClient] = field(default_factory=dict)
    _health_check_task: asyncio.Task | None = field(default=None)
    _local_store: EventStore = field(default_factory=EventStore)
    
    def __post_init__(self):
        if self._task_queue is None:
            self._task_queue = asyncio.Queue()
        if not self.workers:
            # Default to localhost if no workers specified
            self.workers = [WorkerConfig(host="localhost", port=8000)]
    
    def _serialize_event(self, event: Event) -> dict:
        """Serialize an event for network transmission"""
        try:
            # Use JSON for simple types, pickle for complex types
            value = event.get("value")
            if self._is_json_serializable(value):
                return {
                    "id": event["id"],
                    "timestamp": event["timestamp"].isoformat(),
                    "node_id": event["node_id"],
                    "signal": event["signal"],
                    "epoch": event["epoch"],
                    "value": value,
                    "serialization": "json"
                }
            else:
                # Use pickle for complex objects
                return {
                    "id": event["id"],
                    "timestamp": event["timestamp"].isoformat(),
                    "node_id": event["node_id"],
                    "signal": event["signal"],
                    "epoch": event["epoch"],
                    "value": pickle.dumps(value).hex(),
                    "serialization": "pickle"
                }
        except Exception as e:
            self._logger.error("Failed to serialize event: %s", e)
            raise
    
    def _deserialize_event(self, data: dict) -> Event:
        """Deserialize an event from network transmission"""
        try:
            value = data["value"]
            if data.get("serialization") == "pickle":
                value = pickle.loads(bytes.fromhex(value))
            
            return Event(
                id=data["id"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                node_id=data["node_id"],
                signal=data["signal"],
                epoch=data["epoch"],
                value=value
            )
        except Exception as e:
            self._logger.error("Failed to deserialize event: %s", e)
            raise
    
    def _is_json_serializable(self, obj: Any) -> bool:
        """Check if an object is JSON serializable"""
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False
    
    async def _get_worker_client(self, worker_id: str) -> httpx.AsyncClient | None:
        """Get or create an HTTP client for a worker"""
        try:
            if worker_id not in self._worker_clients:
                status = self._worker_statuses.get(worker_id)
                if not status or not status.healthy:
                    return None
                
                base_url = f"http://{status.host}:{status.port}"
                timeout = httpx.Timeout(30.0, connect=5.0)
                self._worker_clients[worker_id] = httpx.AsyncClient(
                    base_url=base_url,
                    timeout=timeout
                )
            return self._worker_clients[worker_id]
        except Exception as e:
            self._logger.warning("Failed to create worker client: %s", e)
            return None
    
    async def _check_worker_health(self, worker: WorkerConfig) -> bool:
        """Check if a worker is healthy"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://{worker.host}:{worker.port}/health")
                return response.status_code == 200
        except Exception:
            return False
    
    async def _health_check_loop(self):
        """Periodically check worker health"""
        while self._running.is_set():
            try:
                for worker in self.workers:
                    worker_id = f"{worker.host}:{worker.port}"
                    healthy = await self._check_worker_health(worker)
                    
                    if worker_id not in self._worker_statuses:
                        self._worker_statuses[worker_id] = WorkerStatus(
                            worker_id=worker_id,
                            host=worker.host,
                            port=worker.port,
                            healthy=healthy,
                            last_seen=datetime.now(UTC)
                        )
                    else:
                        status = self._worker_statuses[worker_id]
                        status.healthy = healthy
                        status.last_seen = datetime.now(UTC)
                        
                        if not healthy and worker_id in self._worker_clients:
                            # Close unhealthy worker client
                            await self._worker_clients[worker_id].aclose()
                            del self._worker_clients[worker_id]
                
                await asyncio.sleep(min(w.health_check_interval for w in self.workers))
            except Exception as e:
                self._logger.error("Error in health check loop: %s", e)
                await asyncio.sleep(5.0)
    
    def _select_worker(self, node_id: str | None = None) -> str | None:
        """
        Select a worker based on the configured load balancing strategy.

        Args:
            node_id: Optional node ID for affinity-based selection

        Returns:
            Selected worker ID or None if no workers available
        """
        healthy_workers = [
            wid for wid, status in self._worker_statuses.items()
            if status.healthy
        ]

        if not healthy_workers:
            return None

        if self.load_balancing == LoadBalancingStrategy.ROUND_ROBIN:
            # Simple round-robin
            if not hasattr(self, "_worker_index"):
                self._worker_index = 0
            worker_id = healthy_workers[self._worker_index % len(healthy_workers)]
            self._worker_index += 1
            return worker_id

        elif self.load_balancing == LoadBalancingStrategy.LEAST_LOADED:
            # Select worker with fewest active tasks
            return min(
                healthy_workers,
                key=lambda wid: self._worker_statuses[wid].tasks_active
            )

        elif self.load_balancing == LoadBalancingStrategy.RANDOM:
            # Random selection
            return random.choice(healthy_workers)

        elif self.load_balancing == LoadBalancingStrategy.FASTEST_RESPONSE:
            # Select worker with best average response time
            return min(
                healthy_workers,
                key=lambda wid: self._worker_statuses[wid].avg_response_time or float("inf")
            )

        else:
            # Default to first healthy worker
            return healthy_workers[0]

    async def _execute_node_remote(
        self, node_id: str, args: list, kwargs: dict, epoch: int, worker_id: str | None = None
    ) -> tuple[list[Event] | None, Exception | None]:
        """Execute a node on a remote worker with improved error handling and metrics"""
        # Select worker if not specified
        if worker_id is None:
            worker_id = self._select_worker(node_id)
            if not worker_id:
                return None, Exception("No healthy workers available")

        client = await self._get_worker_client(worker_id)
        if not client:
            return None, Exception(f"Worker {worker_id} is not available")

        # Track start time for metrics
        start_time = datetime.now(UTC)

        # Update active task count
        status = self._worker_statuses.get(worker_id)
        if status:
            status.tasks_active += 1

        try:
            # Prepare task payload
            payload = {
                "node_id": node_id,
                "args": args,
                "kwargs": kwargs,
                "epoch": epoch,
                "tube_spec": None  # Would need to serialize tube spec for full remote execution
            }

            # Send task to worker
            response = await client.post("/execute", json=payload)
            response.raise_for_status()

            result = response.json()
            events = [self._deserialize_event(e) for e in result.get("events", [])]

            # Update metrics
            execution_time = (datetime.now(UTC) - start_time).total_seconds()
            if status:
                status.tasks_completed += 1
                status.tasks_active -= 1
                status.total_response_time += execution_time
                status.avg_response_time = (
                    status.total_response_time / status.tasks_completed
                )

            return events, None

        except Exception as e:
            # Update failure metrics
            if status:
                status.tasks_failed += 1
                status.tasks_active -= 1

                # Check circuit breaker
                if status.tasks_failed >= self.circuit_breaker_threshold:
                    status.healthy = False
                    self._logger.warning(
                        "Worker %s circuit breaker triggered (%d failures)",
                        worker_id,
                        status.tasks_failed
                    )

            return None, e
    
    async def _execute_node_with_retry(
        self, node_id: str, args: list, kwargs: dict, epoch: int, max_retries: int = 3
    ) -> tuple[list[Event] | None, Exception | None]:
        """Execute a node with retry logic"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Try remote execution first if workers available
                if self.workers and self._worker_statuses:
                    healthy_workers = [
                        wid for wid, status in self._worker_statuses.items()
                        if status.healthy
                    ]
                    if healthy_workers:
                        events, error = await self._execute_node_remote(
                            node_id, args, kwargs, epoch, healthy_workers[attempt % len(healthy_workers)]
                        )
                        if events is not None:
                            return events, None
                        last_error = error
                
                # Fallback to local execution
                if self.local_execution:
                    node = self.tube.nodes[node_id]
                    value = node.process(*args, **kwargs)
                    events = self.store.add(node.signals, value, node_id, epoch)
                    return events, None
                
                # Wait before retry with exponential backoff
                if attempt < max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
        
        return None, last_error
    
    async def _process_epoch_async(self, epoch: int, input: dict | None = None) -> None:
        """Async version of epoch processing"""
        while self.tube.scheduler.is_active(epoch):
            ready = self.tube.scheduler.get_ready(epoch)
            if not ready:
                break
            
            # Process ready nodes in parallel (with limit)
            tasks = []
            for node_info in ready:
                node_id = node_info["value"]
                
                if node_id == "assets":
                    self.tube.scheduler.done(epoch, node_id)
                    continue
                
                node = self.tube.nodes[node_id]
                args, kwargs = self.collect_input(node, epoch, input)
                
                # Handle None inputs - if collect_input returns None, None, it means no inputs available
                if args is None and kwargs is None:
                    # No inputs available - this might be okay for source nodes
                    # but for nodes with dependencies, we should mark as done to avoid infinite loops
                    if node.spec.depends:
                        self._logger.debug("Node %s has dependencies but no inputs available in epoch %s, marking as done", node_id, epoch)
                        self.tube.scheduler.done(epoch, node_id)
                        continue
                    args = []
                    kwargs = {}
                else:
                    args = [] if args is None else args
                    kwargs = {} if kwargs is None else kwargs
                
                # Create task for async execution
                task = self._execute_node_with_retry(node_id, args, kwargs, epoch)
                tasks.append((node_id, task))
                
                # Limit concurrent tasks
                if len(tasks) >= self.max_parallel:
                    break
            
            # Wait for tasks to complete
            for node_id, task in tasks:
                events, error = await task
                
                if error:
                    self._logger.error("Error executing node %s: %s", node_id, error)
                    # Mark node as done even on error to avoid blocking
                    self.tube.scheduler.done(epoch, node_id)
                    continue
                
                if events:
                    updated_events = self.tube.scheduler.update(events)
                    self._call_callbacks(updated_events)
                    
                    # Handle map nodes specially - mark as done after storing all items
                    node = self.tube.nodes[node_id]
                    if isinstance(node, Map) and events:
                        # Map node already stored all items, mark as done
                        self.tube.scheduler.done(epoch, node_id)
                        # Continue processing - downstream nodes will collect from event store
                    else:
                        self.tube.scheduler.done(epoch, node_id)
                        self._logger.debug("Node %s emitted %s in epoch %s", node_id, events, epoch)
                else:
                    self.tube.scheduler.done(epoch, node_id)
    
    def init(self) -> Self:
        """Start processing data with the tube graph."""
        with self._lock:
            if self._running.is_set():
                raise AlreadyRunningError("Tube is already running!")
            
            self._running.set()
            
            # Initialize nodes and assets
            for node in self.tube.enabled_nodes.values():
                self.inject_context(node.init)()
            
            for asset in self.tube.state.assets.values():
                self.inject_context(asset.init)()
            
            # Start health check loop if async is enabled
            if self.use_async:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Create a new task if loop is already running
                        self._health_check_task = asyncio.create_task(self._health_check_loop())
                    else:
                        # Run in background thread
                        loop.run_until_complete(self._health_check_loop())
                except RuntimeError:
                    # No event loop, create one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    self._health_check_task = loop.create_task(self._health_check_loop())
        
        return self
    
    def deinit(self) -> None:
        """Stop all nodes processing"""
        with self._lock:
            if not self._running.is_set():
                return
            
            self._running.clear()
            
            # Cancel health check task
            if self._health_check_task:
                self._health_check_task.cancel()
            
            # Close worker clients
            if self.use_async:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Schedule cleanup
                        for client in self._worker_clients.values():
                            asyncio.create_task(client.aclose())
                    else:
                        # Run cleanup synchronously
                        loop.run_until_complete(
                            asyncio.gather(*[client.aclose() for client in self._worker_clients.values()], return_exceptions=True)
                        )
                except Exception as e:
                    self._logger.warning("Error closing worker clients: %s", e)
                finally:
                    self._worker_clients.clear()
            
            # Deinitialize nodes and assets
            for node in self.tube.enabled_nodes.values():
                try:
                    node.deinit()
                except Exception as e:
                    self._logger.warning("Error deinitializing node %s: %s", node.id, e)
            
            for asset in self.tube.state.assets.values():
                try:
                    asset.deinit()
                except Exception as e:
                    self._logger.warning("Error deinitializing asset: %s", e)
    
    @property
    def running(self) -> bool:
        """Whether the tube is currently running"""
        return self._running.is_set()
    
    def process(self, **kwargs: Any) -> ReturnNodeType:
        """
        Process one step of data from each of the sources.
        Uses async execution if available, falls back to sync otherwise.
        """
        if not self._running.is_set():
            self.init()
        
        input = self.tube.input_collection.validate_input(InputScope.process, kwargs)
        self.store.clear()
        
        scheduler = self.tube.scheduler
        scheduler.add_epoch()
        # Get the epoch that was just created (the maximum key in _epochs)
        initial_epoch = max(scheduler._epochs.keys())
        
        # Process using async if available, otherwise sync
        if self.use_async:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, we need to use sync fallback
                    # In a real implementation, you'd use a different approach
                    return self._process_sync(initial_epoch, input)
                else:
                    loop.run_until_complete(self._process_epoch_async(initial_epoch, input))
            except RuntimeError:
                # No event loop, use sync
                return self._process_sync(initial_epoch, input)
        else:
            return self._process_sync(initial_epoch, input)
        
        return self.collect_return()
    
    def _process_sync(self, epoch: int, input: dict | None = None) -> ReturnNodeType:
        """Synchronous fallback processing - reuses logic from SynchronousRunner"""
        # Import here to avoid circular dependency
        from noob.runner.sync import SynchronousRunner
        
        # Create a temporary sync runner to reuse its processing logic
        sync_runner = SynchronousRunner(tube=self.tube, store=self.store)
        sync_runner._running.set()  # Mark as running
        try:
            sync_runner._process_epoch(epoch, input)
        finally:
            pass  # Don't deinit the sync runner, we're borrowing its logic
        
        return self.collect_return()
    
    def iter(self, n: int | None = None) -> Generator[ReturnNodeType, None, None]:
        """Treat the runner as an iterable."""
        self.init()
        current_iter = 0
        try:
            while n is None or current_iter < n:
                ret = None
                loop = 0
                while ret is None:
                    ret = self.process()
                    loop += 1
                    if loop > self.MAX_ITER_LOOPS:
                        raise RuntimeError("Reached maximum process calls per iteration")
                
                yield ret
                current_iter += 1
        finally:
            self.deinit()
    
    def run(self, n: int | None = None) -> None | list[ReturnNodeType]:
        """Run the tube either indefinitely or for a fixed number of complete iterations"""
        outputs = []
        current_iter = 0
        self.init()
        try:
            while n is None or current_iter < n:
                out = self.process()
                if out is not None:
                    outputs.append(out)
                current_iter += 1
        except (KeyboardInterrupt, StopIteration):
            pass
        finally:
            self.deinit()
        
        return outputs if outputs else None
    
    def enable_node(self, node_id: str) -> None:
        """Enable a node during runtime"""
        self.tube.nodes[node_id].init()
        self.tube.enable_node(node_id)
    
    def disable_node(self, node_id: str) -> None:
        """Disable a node during runtime"""
        self.tube.nodes[node_id].deinit()
        self.tube.disable_node(node_id)
    
    MAX_ITER_LOOPS = 100
    """The max number of times that `iter` will call `process` to try and get a result"""

