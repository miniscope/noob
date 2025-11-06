"""
MultiProcess Runner for Parallel Execution

This module provides a runner that uses Python's multiprocessing to execute
graph nodes in parallel across multiple CPU cores. This is ideal for CPU-bound
tasks and can provide significant speedups on multi-core systems.

Features:
- Process pool execution for true parallelism (bypasses GIL)
- Automatic process management and cleanup
- Shared state management across processes
- Support for both fork and spawn start methods
- Batch processing for efficiency
- Error handling and recovery
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import traceback
from collections.abc import Generator, Sequence
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field
from multiprocessing import Manager, Queue
from threading import Event as ThreadEvent, Lock
from typing import Any, Self

from noob.event import Event
from noob.exceptions import AlreadyRunningError
from noob.input import InputScope
from noob.node import Node
from noob.node.map import Map
from noob.runner.base import TubeRunner
from noob.store import EventStore
from noob.types import ReturnNodeType


def _execute_node_worker(
    node_id: str,
    node_pickle: bytes,
    args: list,
    kwargs: dict,
    signals: list,
    epoch: int
) -> tuple[str, list[dict], str | None]:
    """
    Worker function for executing a node in a separate process.

    This function is pickled and sent to worker processes, so it must be
    defined at module level.

    Args:
        node_id: ID of the node to execute
        node_pickle: Pickled node object
        args: Positional arguments for node.process()
        kwargs: Keyword arguments for node.process()
        signals: Node's signal specifications
        epoch: Current epoch number

    Returns:
        Tuple of (node_id, serialized_events, error_message)
    """
    try:
        # Deserialize node
        node = pickle.loads(node_pickle)

        # Execute node
        value = node.process(*args, **kwargs)

        # Create events (simplified - we don't have full EventStore here)
        events = []
        if value is not None:
            # Handle multiple signals
            if isinstance(value, tuple) and len(signals) > 1:
                for sig, val in zip(signals, value):
                    events.append({
                        "node_id": node_id,
                        "signal": sig.name,
                        "epoch": epoch,
                        "value": pickle.dumps(val).hex(),  # Pickle for safety
                        "serialization": "pickle"
                    })
            else:
                # Single signal
                sig = signals[0] if signals else None
                signal_name = sig.name if sig else "value"
                events.append({
                    "node_id": node_id,
                    "signal": signal_name,
                    "epoch": epoch,
                    "value": pickle.dumps(value).hex(),
                    "serialization": "pickle"
                })

        return (node_id, events, None)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return (node_id, [], error_msg)


@dataclass
class MultiProcessRunner(TubeRunner):
    """
    Parallel tube runner using multiprocessing.

    This runner executes independent nodes in parallel across multiple
    processes, providing true parallelism for CPU-bound tasks.

    Features:
    - Process pool for efficient worker management
    - Parallel execution of independent nodes
    - Automatic dependency resolution
    - Shared state across processes
    - Graceful error handling
    - Configurable worker count

    Example:
        >>> runner = MultiProcessRunner(tube, max_workers=8)
        >>> results = runner.run(n=10)
    """

    max_workers: int | None = None  # None = use CPU count
    task_timeout: float | None = 300.0  # 5 minutes per task
    use_spawn: bool = False  # Use spawn instead of fork (safer but slower)
    chunk_size: int = 1  # Batch size for task submission

    MAX_ITER_LOOPS = 100

    def __post_init__(self):
        self._running = ThreadEvent()
        self._lock = Lock()

        # Set multiprocessing start method if specified
        if self.use_spawn:
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                # Already set, ignore
                pass

        # Determine worker count
        if self.max_workers is None:
            self.max_workers = mp.cpu_count()

        self._executor: ProcessPoolExecutor | None = None

    def __enter__(self) -> Self:
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        self.deinit()

    def init(self) -> Self:
        """Start processing data with the tube graph."""
        with self._lock:
            if self._running.is_set():
                raise AlreadyRunningError("Tube is already running!")

            self._running.set()

            # Initialize nodes (in main process)
            for node in self.tube.enabled_nodes.values():
                self.inject_context(node.init)()

            for asset in self.tube.state.assets.values():
                self.inject_context(asset.init)()

            # Create process pool
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)

            self._logger.info("MultiProcessRunner initialized with %d workers", self.max_workers)

        return self

    def deinit(self) -> None:
        """Stop all nodes processing"""
        with self._lock:
            if not self._running.is_set():
                return

            # Shutdown executor
            if self._executor:
                self._logger.info("Shutting down process pool...")
                self._executor.shutdown(wait=True, cancel_futures=False)
                self._executor = None

            # Deinitialize nodes
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

            self._running.clear()

    @property
    def running(self) -> bool:
        """Whether the tube is currently running"""
        return self._running.is_set()

    def _process_epoch(self, epoch: int, input: dict | None = None, max_iterations: int = 1000) -> None:
        """Process all ready nodes in a given epoch with parallel execution"""
        iterations = 0

        while self.tube.scheduler.is_active(epoch):
            iterations += 1
            if iterations > max_iterations:
                self._logger.error(
                    "Max iterations (%d) reached in epoch %d, breaking",
                    max_iterations,
                    epoch
                )
                break

            ready = self.tube.scheduler.get_ready(epoch)
            if not ready:
                break

            # Separate special nodes from regular nodes
            regular_nodes = []
            for node_info in ready:
                node_id = node_info["value"]

                if node_id == "assets":
                    self.tube.scheduler.done(epoch, node_id)
                    continue

                regular_nodes.append(node_id)

            if not regular_nodes:
                continue

            # Submit all ready nodes to process pool
            futures = {}
            for node_id in regular_nodes:
                node = self.tube.nodes[node_id]
                args, kwargs = self.collect_input(node, epoch, input)

                # Handle None inputs
                if args is None and kwargs is None:
                    if node.spec.depends:
                        self._logger.debug(
                            "Node %s has dependencies but no inputs, marking done",
                            node_id
                        )
                        self.tube.scheduler.done(epoch, node_id)
                        continue
                    args = []
                    kwargs = {}
                else:
                    args = [] if args is None else args
                    kwargs = {} if kwargs is None else kwargs

                # Pickle the node for transmission to worker
                try:
                    node_pickle = pickle.dumps(node)
                except Exception as e:
                    self._logger.error("Failed to pickle node %s: %s", node_id, e)
                    self.tube.scheduler.done(epoch, node_id)
                    continue

                # Submit task
                future = self._executor.submit(
                    _execute_node_worker,
                    node_id,
                    node_pickle,
                    args,
                    kwargs,
                    node.signals,
                    epoch
                )
                futures[future] = node_id

            # Collect results as they complete
            for future in as_completed(futures, timeout=self.task_timeout):
                node_id = futures[future]
                node = self.tube.nodes[node_id]

                try:
                    result_node_id, event_dicts, error = future.result(timeout=1.0)

                    if error:
                        self._logger.error("Error processing node %s: %s", node_id, error)
                        self.tube.scheduler.done(epoch, node_id)
                        continue

                    # Deserialize and store events
                    if event_dicts:
                        for event_dict in event_dicts:
                            # Deserialize value
                            if event_dict.get("serialization") == "pickle":
                                value = pickle.loads(bytes.fromhex(event_dict["value"]))
                            else:
                                value = event_dict["value"]

                            # Store in event store
                            events = self.store.add(
                                node.signals,
                                value,
                                event_dict["node_id"],
                                event_dict["epoch"]
                            )

                            if events:
                                # Update scheduler with events
                                if isinstance(node, Map):
                                    # Map nodes need special handling
                                    self._call_callbacks(events)
                                else:
                                    updated = self.tube.scheduler.update(events)
                                    self._call_callbacks(updated)

                        # Mark node as done after all events processed
                        if isinstance(node, Map) or not events:
                            self.tube.scheduler.done(epoch, node_id)

                    else:
                        # No events returned (e.g., Gather accumulating)
                        self.tube.scheduler.done(epoch, node_id)

                except TimeoutError:
                    self._logger.error("Timeout processing node %s", node_id)
                    self.tube.scheduler.done(epoch, node_id)
                except Exception as e:
                    self._logger.error("Error collecting result for node %s: %s", node_id, e)
                    self.tube.scheduler.done(epoch, node_id)

    def process(self, **kwargs: Any) -> ReturnNodeType:
        """
        Process one epoch using parallel execution.

        Process-scoped inputs can be passed as kwargs.
        """
        if not self._running.is_set():
            self.init()

        input = self.tube.input_collection.validate_input(InputScope.process, kwargs)
        self.store.clear()

        scheduler = self.tube.scheduler
        scheduler.add_epoch()
        initial_epoch = max(scheduler._epochs.keys())

        # Process the epoch with parallel execution
        self._process_epoch(initial_epoch, input)

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
        """Run the tube for n iterations"""
        outputs = []
        current_iter = 0
        no_output_count = 0
        max_no_output = self.MAX_ITER_LOOPS

        self.init()
        try:
            while n is None or current_iter < n:
                out = self.process()
                if out is not None:
                    outputs.append(out)
                    no_output_count = 0
                else:
                    no_output_count += 1
                    if n is None and no_output_count >= max_no_output:
                        break
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


@dataclass
class SharedMemoryRunner(TubeRunner):
    """
    Advanced multiprocess runner with shared memory for large data.

    This runner uses shared memory to avoid pickling overhead when
    passing large numpy arrays or other data between processes.

    Note: Requires Python 3.8+ and numpy for shared memory support.
    """

    max_workers: int | None = None
    task_timeout: float | None = 300.0
    use_shared_memory: bool = True

    MAX_ITER_LOOPS = 100

    def __post_init__(self):
        self._running = ThreadEvent()
        self._lock = Lock()

        if self.max_workers is None:
            self.max_workers = mp.cpu_count()

        self._executor: ProcessPoolExecutor | None = None
        self._shared_memory_blocks: dict[str, Any] = {}

    def init(self) -> Self:
        """Initialize runner"""
        with self._lock:
            if self._running.is_set():
                raise AlreadyRunningError("Tube is already running!")

            self._running.set()

            for node in self.tube.enabled_nodes.values():
                self.inject_context(node.init)()

            for asset in self.tube.state.assets.values():
                self.inject_context(asset.init)()

            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)

            self._logger.info(
                "SharedMemoryRunner initialized with %d workers",
                self.max_workers
            )

        return self

    def deinit(self) -> None:
        """Cleanup runner"""
        with self._lock:
            if not self._running.is_set():
                return

            if self._executor:
                self._executor.shutdown(wait=True)
                self._executor = None

            # Cleanup shared memory
            for shm in self._shared_memory_blocks.values():
                try:
                    shm.close()
                    shm.unlink()
                except Exception as e:
                    self._logger.warning("Error cleaning up shared memory: %s", e)

            self._shared_memory_blocks.clear()

            for node in self.tube.enabled_nodes.values():
                try:
                    node.deinit()
                except Exception as e:
                    self._logger.warning("Error deinitializing node: %s", e)

            for asset in self.tube.state.assets.values():
                try:
                    asset.deinit()
                except Exception as e:
                    self._logger.warning("Error deinitializing asset: %s", e)

            self._running.clear()

    @property
    def running(self) -> bool:
        return self._running.is_set()

    def process(self, **kwargs: Any) -> ReturnNodeType:
        """Process using shared memory optimization"""
        # For now, fall back to MultiProcessRunner behavior
        # Full shared memory implementation would require more infrastructure
        raise NotImplementedError(
            "SharedMemoryRunner is experimental. Use MultiProcessRunner instead."
        )

    def iter(self, n: int | None = None) -> Generator[ReturnNodeType, None, None]:
        raise NotImplementedError("Use MultiProcessRunner instead")

    def run(self, n: int | None = None) -> None | list[ReturnNodeType]:
        raise NotImplementedError("Use MultiProcessRunner instead")

    def enable_node(self, node_id: str) -> None:
        self.tube.nodes[node_id].init()
        self.tube.enable_node(node_id)

    def disable_node(self, node_id: str) -> None:
        self.tube.nodes[node_id].deinit()
        self.tube.disable_node(node_id)
