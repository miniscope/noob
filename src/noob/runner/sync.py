from __future__ import annotations

from dataclasses import dataclass
from threading import Event as ThreadEvent
from typing import Any

from noob.event import MetaEvent
from noob.node import Return
from noob.runner.base import TubeRunner
from noob.scheduler import Scheduler
from noob.types import ReturnNodeType


@dataclass
class SynchronousRunner(TubeRunner):
    """
    Simple, synchronous tube runner.

    Just run the nodes in topological order and return from return nodes.
    """

    def __post_init__(self):
        super().__post_init__()
        self._running = ThreadEvent()

    def init(self) -> None:
        """
        Start processing data with the tube graph.
        """
        # TODO: lock for re-entry
        if self._running.is_set():
            # fine!
            return

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
        synchronously calling their process method and passing events as they are emitted.

        Process-scoped ``input`` s can be passed as kwargs.
        """
        # mostly overriding to set the docstring
        return super().process(**kwargs)

    def _before_process(self) -> None:
        """
        Clear the eventstore and add a new epoch.
        Initialize if not already done.
        """
        self.store.clear()
        self.tube.scheduler.add_epoch()
        if not self._running.is_set():
            self.init()

    def _filter_ready(self, nodes: list[MetaEvent], scheduler: Scheduler) -> list[MetaEvent]:
        # graph autogenerates "assets" and "inputs" nodes if something depends on it
        # but in the sync runner we always have assets and inputs handy
        evts = []
        for node in nodes:
            if node["value"] in ("assets", "input"):
                scheduler.done(node["epoch"], node["value"])
            else:
                evts.append(node)
        return evts

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
