from __future__ import annotations

from dataclasses import dataclass
from threading import Event as ThreadEvent
from typing import Any

from noob.exceptions import AlreadyRunningError
from noob.input import InputScope
from noob.node import Return
from noob.runner.base import TubeRunner
from noob.types import ReturnNodeType


@dataclass
class SynchronousRunner(TubeRunner):
    """
    Simple, synchronous tube runner.

    Just run the nodes in topological order and return from return nodes.
    """

    def __post_init__(self):
        self._running = ThreadEvent()

    def init(self) -> None:
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
                scheduler.end_epoch()
                break
            for node_info in ready:
                node_id, epoch = node_info["value"], node_info["epoch"]

                if node_id == "assets":
                    # graph autogenerates "assets" node if something depends on it
                    scheduler.done(epoch, node_id)
                    continue
                node = self.tube.nodes[node_id]
                maybe_args, maybe_kwargs = self.collect_input(node, epoch, input)

                # need to eventually distinguish "still waiting" vs "there is none"
                args = [] if maybe_args is None else maybe_args
                kwargs = {} if maybe_kwargs is None else maybe_kwargs
                value = node.process(*args, **kwargs)

                # take the value from state first. if it's taken by an asset,
                # the value is converted to its id, and returned again.
                events = self.store.add_value(node.signals, value, node_id, epoch)
                if events is None:
                    continue
                all_events = scheduler.update(events)
                self._call_callbacks(all_events)
                self._logger.debug("Node %s emitted %s in epoch %s", node_id, value, epoch)

        return self.collect_return()

    def enable_node(self, node_id: str) -> None:
        self.tube.nodes[node_id].init()
        self.tube.enable_node(node_id)

    def disable_node(self, node_id: str) -> None:
        self.tube.nodes[node_id].deinit()
        self.tube.disable_node(node_id)

    def collect_return(self, epoch: int | None = None) -> ReturnNodeType:
        """The return node holds values from a single epoch, get and transform them"""
        if epoch is not None:
            raise ValueError("Sync runner only stores a single epoch at a time")
        ret_nodes = [n for n in self.tube.enabled_nodes.values() if isinstance(n, Return)]
        if not ret_nodes:
            return None
        ret_node = ret_nodes[0]
        return ret_node.get(keep=False)
