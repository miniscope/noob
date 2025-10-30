from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from threading import Event as ThreadEvent
from typing import Any, Self

from noob.exceptions import AlreadyRunningError
from noob.input import InputScope
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

    def __enter__(self) -> Self:
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        self.deinit()

    def init(self) -> Self:
        """
        Start processing data with the tube graph.
        """
        # TODO: lock for re-entry
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
        # TODO: lock to ensure we've been started
        for node in self.tube.enabled_nodes.values():
            node.deinit()

        for asset in self.tube.state.assets.values():
            asset.deinit()

        self._running.clear()

    @property
    def running(self) -> bool:
        """Whether the tube is currently running"""
        return self._running.is_set()

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

        while scheduler.is_active():
            ready = scheduler.get_ready()
            if not ready:
                break
            for node_info in ready:
                node_id, epoch = node_info["value"], node_info["epoch"]

                if node_id == "assets":
                    # graph autogenerates "assets" node if something depends on it
                    scheduler.done(epoch, node_id)
                    continue
                node = self.tube.nodes[node_id]
                args, kwargs = self.collect_input(node, epoch, input)

                # need to eventually distinguish "still waiting" vs "there is none"
                args = [] if args is None else args
                kwargs = {} if kwargs is None else kwargs
                value = node.process(*args, **kwargs)

                # take the value from state first. if it's taken by an asset,
                # the value is converted to its id, and returned again.
                events = self.store.add(node.signals, value, node_id, epoch)
                events = scheduler.update(events)
                self._call_callbacks(events)
                self._logger.debug("Node %s emitted %s in epoch %s", node_id, value, epoch)

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
        self.init()
        try:
            while n is None or current_iter < n:
                out = self.process()
                if out is not None:
                    outputs.append(out)
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
