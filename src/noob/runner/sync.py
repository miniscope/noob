from __future__ import annotations

from collections.abc import Generator, Sequence
from dataclasses import dataclass
from threading import Event as ThreadEvent, Lock
from typing import Any, Self

from noob.exceptions import AlreadyRunningError
from noob.input import InputScope
from noob.node.map import Map
from noob.runner.base import TubeRunner
from noob.types import ReturnNodeType


@dataclass
class SynchronousRunner(TubeRunner):
    """
    Simple, synchronous tube runner.

    Just run the nodes in topological order and return from return nodes.
    """

    MAX_ITER_LOOPS = 100
    """The max number of times that `iter` will call `process` to try and get a result"""

    def __post_init__(self):
        self._running = ThreadEvent()
        self._lock = Lock()

    def __enter__(self) -> Self:
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        self.deinit()

    def init(self) -> Self:
        """
        Start processing data with the tube graph.
        """
        with self._lock:
            if self._running.is_set():
                raise AlreadyRunningError("Tube is already running!")

            self._running.set()
            for node in self.tube.enabled_nodes.values():
                self.inject_context(node.init)()

            for asset in self.tube.state.assets.values():
                self.inject_context(asset.init)()

        return self

    def deinit(self) -> None:
        """Stop all nodes processing"""
        with self._lock:
            if not self._running.is_set():
                return  # Already deinitialized, safe to ignore

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
        """Process all ready nodes in a given epoch until completion"""
        iterations = 0
        while self.tube.scheduler.is_active(epoch):
            iterations += 1
            if iterations > max_iterations:
                self._logger.error("Max iterations (%d) reached in epoch %d, breaking to avoid infinite loop", max_iterations, epoch)
                break
            
            ready = self.tube.scheduler.get_ready(epoch)
            if not ready:
                break
            
            # Process all ready nodes
            for node_info in ready:
                node_id = node_info["value"]

                if node_id == "assets":
                    self.tube.scheduler.done(epoch, node_id)
                    continue
                
                node = self.tube.nodes[node_id]
                args, kwargs = self.collect_input(node, epoch, input)

                # Handle None inputs - if collect_input returns None, None, it means no inputs available
                # This is different from empty args/kwargs which means inputs were found but empty
                if args is None and kwargs is None:
                    # No inputs available - this might be okay for source nodes
                    # but for nodes with dependencies, we should mark as done to avoid infinite loops
                    if node.spec.depends:
                        # Node has dependencies but no inputs - mark as done to avoid infinite loop
                        # This can happen if dependencies haven't produced outputs yet
                        self._logger.debug("Node %s has dependencies but no inputs available in epoch %s, marking as done", node_id, epoch)
                        self.tube.scheduler.done(epoch, node_id)
                        continue
                    args = []
                    kwargs = {}
                else:
                    args = [] if args is None else args
                    kwargs = {} if kwargs is None else kwargs
                
                try:
                    value = node.process(*args, **kwargs)
                except Exception as e:
                    self._logger.error("Error processing node %s: %s", node_id, e)
                    # Mark node as done even on error to avoid blocking
                    self.tube.scheduler.done(epoch, node_id)
                    continue

                # Special handling for Map nodes: they return a list that should be split
                # into separate events, all in the same epoch
                if isinstance(node, Map) and isinstance(value, Sequence) and len(value) > 0:
                    # For map nodes, store all items as events in the same epoch
                    for item in value:
                        # Store the map node's output event for this item (use same epoch)
                        item_events = self.store.add(node.signals, item, node_id, epoch)
                        if item_events:
                            self._call_callbacks(item_events)
                            self._logger.debug("Map node %s emitted item %s in epoch %s", node_id, item, epoch)
                    
                    # Mark the map node as done after storing all items
                    self.tube.scheduler.done(epoch, node_id)
                    # Continue processing - downstream nodes will pick up map items from the event store
                else:
                    # Normal node processing
                    # Handle None returns specially for nodes like Gather that accumulate state
                    if value is None:
                        # Node returned None - mark as done but don't emit events
                        # This allows nodes like Gather to accumulate items without blocking
                        self.tube.scheduler.done(epoch, node_id)
                        self._logger.debug("Node %s returned None in epoch %s (accumulating state)", node_id, epoch)
                    else:
                        # Node returned a value - store it and update scheduler
                        events = self.store.add(node.signals, value, node_id, epoch)
                        if events:
                            events = self.tube.scheduler.update(events)
                            self._call_callbacks(events)
                            self._logger.debug("Node %s emitted %s in epoch %s", node_id, value, epoch)
                        else:
                            # If no events were created, mark node as done anyway
                            self.tube.scheduler.done(epoch, node_id)

    def process(self, **kwargs: Any) -> ReturnNodeType:
        """
        Iterate through nodes in topological order,
        calling their process method and passing events as they are emitted.

        Process-scoped ``input`` s can be passed as kwargs.
        """
        if not self._running.is_set():
            self.init()

        input = self.tube.input_collection.validate_input(InputScope.process, kwargs)
        self.store.clear()

        scheduler = self.tube.scheduler
        scheduler.add_epoch()
        # Get the epoch that was just created (the maximum key in _epochs)
        initial_epoch = max(scheduler._epochs.keys())

        # Process the initial epoch (which will recursively process any map-generated epochs)
        self._process_epoch(initial_epoch, input)

        return self.collect_return()

    def iter(self, n: int | None = None) -> Generator[ReturnNodeType, None, None]:
        """
        Treat the runner as an iterable.

        Calls :meth:`.TubeRunner.process` until it yields a result
        (e.g. multiple times in the case of any ``gather`` s
        that change the cardinality of the graph.)
        """

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
        outputs = []
        current_iter = 0
        no_output_count = 0
        max_no_output = self.MAX_ITER_LOOPS  # Stop if we get no outputs for this many iterations
        self.init()
        try:
            while n is None or current_iter < n:
                out = self.process()
                if out is not None:
                    outputs.append(out)
                    no_output_count = 0  # Reset counter when we get output
                else:
                    no_output_count += 1
                    # If n is None and we've had no outputs for max_no_output iterations, stop
                    if n is None and no_output_count >= max_no_output:
                        break
                current_iter += 1
        except (KeyboardInterrupt, StopIteration):
            # fine, just return
            pass
        finally:
            self.deinit()

        return outputs if outputs else None

    def enable_node(self, node_id: str) -> None:
        self.tube.nodes[node_id].init()
        self.tube.enable_node(node_id)

    def disable_node(self, node_id: str) -> None:
        self.tube.nodes[node_id].deinit()
        self.tube.disable_node(node_id)