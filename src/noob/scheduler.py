from collections import deque
from datetime import UTC, datetime
from enum import StrEnum
from graphlib import _NODE_DONE, _NODE_OUT, TopologicalSorter
from itertools import count
from threading import Condition
from typing import Self
from uuid import uuid4

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from noob.event import Event, MetaEvent
from noob.node import Edge, NodeSpecification
from noob.types import NodeID


class SchedulerMode(StrEnum):
    sync = "sync"
    """
    All nodes from a single epoch are run before yielding nodes from a new epoch.
    The calling runner is expected to advance the epoch with add_epoch
    """
    parallel = "parallel"
    """
    Nodes are yielded as soon as they can be from multiple epochs
    Epoch is advanced by "source" nodes (without any dependencies)
    or by calling `process` when there is tube-scoped input. 
    """


class Scheduler(BaseModel):
    nodes: dict[str, NodeSpecification]
    edges: list[Edge]
    source_nodes: list[NodeID] = Field(default_factory=list)
    mode: SchedulerMode = SchedulerMode.sync
    _clock: count = PrivateAttr(default_factory=count)
    _epochs: dict[int, TopologicalSorter] = PrivateAttr(default_factory=dict)
    _ready_condition: Condition = PrivateAttr(default_factory=Condition)
    _epoch_condition: Condition = PrivateAttr(default_factory=Condition)
    _epoch_log: deque = PrivateAttr(default_factory=lambda: deque(maxlen=100))

    @classmethod
    def from_specification(cls, nodes: dict[str, NodeSpecification], edges: list[Edge]) -> Self:
        """
        Create an instance of a Scheduler from :class:`.NodeSpecification` and :class:`.Edge`

        """
        return cls(nodes=nodes, edges=edges)

    @model_validator(mode="after")
    def get_sources(self) -> Self:
        """
        Get the IDs of the nodes that do not depend on other nodes.

        * `input` nodes are special implicit source nodes. Other nodes
        * CAN depend on it and still be a source node.

        """
        if not self.source_nodes:
            graph = self._init_graph(nodes=self.nodes, edges=self.edges)
            self.source_nodes = [
                id_ for id_, info in graph._node2info.items() if info.npredecessors == 0
            ]
        return self

    def add_epoch(self, epoch: int | None = None) -> None:
        """
        Add another epoch with a prepared graph to the scheduler.

        """
        with self._ready_condition:
            this_epoch = next(self._clock) if epoch is None else epoch
            if this_epoch in self._epochs:
                raise ValueError(f"Epoch {this_epoch} is already scheduled")
            graph = self._init_graph(nodes=self.nodes, edges=self.edges)
            graph.prepare()
            self._epochs[this_epoch] = graph
            self._ready_condition.notify_all()

    def is_active(self, epoch: int | None = None) -> bool:
        """
        Graph remains active while it holds at least one epoch that is active.

        """
        if epoch is not None:
            if epoch not in self._epochs:
                # if an epoch has been completed and had its graph cleared, it's no longer active
                # if an epoch has not been started, it is also not active.
                return False
            return self._epochs[epoch].is_active()
        else:
            return any(graph.is_active() for graph in self._epochs.values())

    def get_ready(self, epoch: int | None = None) -> list[MetaEvent]:
        """
        Output the set of nodes that are ready across different epochs.

        Args:
            epoch (int | None): if an int, get ready events for that epoch,
                if ``None`` , get ready events for all epochs.
        """

        graphs = self._epochs.items() if epoch is None else [(epoch, self._epochs[epoch])]

        with self._ready_condition:
            if epoch is None and self.mode == SchedulerMode.parallel and self.sources_finished():
                self.add_epoch()
            ready_nodes = [
                MetaEvent(
                    id=uuid4().int,
                    timestamp=datetime.now(),
                    node_id="meta",
                    signal="NodeReady",
                    epoch=epoch,
                    value=node_id,
                )
                for epoch, graph in graphs
                for node_id in graph.get_ready()
            ]
            if epoch is None and self.mode == SchedulerMode.parallel:
                ready_nodes += self._get_ready_sources()

        return ready_nodes

    def node_is_ready(self, node: NodeID, epoch: int | None = None) -> bool:
        """
        Check if a single node is ready in a single or any epoch

        Args:
            node (NodeID): the node to check
            epoch (int | None): the epoch to check, if ``None`` , any epoch
        """
        # slight duplication of the above because we don't want to *get* the ready nodes,
        # which marks them as "out" in the TopologicalSorter

        if node in self.source_nodes and epoch is None and self.mode == SchedulerMode.parallel:
            # source nodes are ready when they have completed their last epoch
            return self.sources_finished()
        else:
            graphs = self._epochs.items() if epoch is None else [(epoch, self._epochs[epoch])]
            return any(node_id == node for epoch, graph in graphs for node_id in graph._ready_nodes)

    def __getitem__(self, epoch: int) -> TopologicalSorter:
        if epoch == -1:
            return next(reversed(self._epochs.values()))
        return self._epochs[epoch]

    def sources_finished(self, epoch: int | None = None) -> bool:
        """
        Check the source nodes of the given epoch have been processed.
        If epoch is None, check the source nodes of the latest epoch.

        """
        if len(self._epochs) == 0 and epoch is None:
            return True

        graph = self[-1] if epoch is None else self._epochs[epoch]
        return all(
            graph._node2info[src].npredecessors == _NODE_DONE or src in graph._ready_nodes
            for src in self.source_nodes
        )

    def update(self, events: list[Event]) -> list[Event]:
        """
        When a set of events are received, update the graphs within the scheduler.
        Currently only has :method:`TopologicalSorter.done` implemented.

        """
        if not events:
            return events
        end_events = []
        with self._ready_condition, self._epoch_condition:
            is_done = set((e["epoch"], e["node_id"]) for e in events)
            for epoch, node_id in is_done:
                epoch_ended = self.done(epoch=epoch, node_id=node_id)
                if epoch_ended:
                    end_events.append(epoch_ended)

            # condition uses an RLock, so waiters only run here,
            # even though `done` also notifies.
            self._ready_condition.notify_all()
        events.extend(end_events)

        return events

    def done(self, epoch: int, node_id: str) -> MetaEvent | None:
        """
        Mark a node in a given epoch as done.

        """
        with self._ready_condition, self._epoch_condition:
            self[epoch].done(node_id)
            self._ready_condition.notify_all()

            if not self[epoch].is_active():
                self._end_epoch(epoch)
                return MetaEvent(
                    id=uuid4().int,
                    timestamp=datetime.now(UTC),
                    node_id="meta",
                    signal="EpochEnded",
                    epoch=epoch,
                    value=epoch,
                )
        return None

    def await_node(self, node_id: NodeID, epoch: int | None = None) -> MetaEvent:
        """
        Block until a node is ready

        Args:
            node_id:
            epoch (int, None): if `int` , wait until the node is ready in the given epoch,
                otherwise wait until the node is ready in any epoch

        Returns:

        """
        with self._ready_condition:
            if not self.node_is_ready(node_id, epoch):
                self._ready_condition.wait_for(lambda: self.node_is_ready(node_id, epoch))

            # FIXME: instead of modifying the scheduler with a mode,
            # let's move this into the node runner
            if node_id in self.source_nodes and self.mode == SchedulerMode.parallel:
                self.add_epoch()

            # be FIFO-like and get the earliest epoch the node is ready in
            if epoch is None:
                for ep in self._epochs:
                    if self.node_is_ready(node_id, ep):
                        epoch = ep

            # do a little graphlib surgery to mark just one event as done.
            # threadsafe because we are holding the lock that protects graph mutation
            self._epochs[epoch]._node2info[node_id].npredecessors = _NODE_OUT
            self._epochs[epoch]._ready_nodes.remove(node_id)

        return MetaEvent(
            id=uuid4().int,
            timestamp=datetime.now(),
            node_id="meta",
            signal="NodeReady",
            epoch=epoch,
            value=node_id,
        )

    def await_epoch(self, epoch: int | None = None) -> int:
        """
        Block until an epoch is completed.

        Args:
            epoch (int, None): if `int` , wait until the epoch is ready,
                otherwise wait until the next epoch is finished, in whatever order.

        Returns:
            int: the epoch that was completed.
        """
        with self._epoch_condition:
            # check if we have already completed this epoch
            if isinstance(epoch, int) and self.epoch_completed(epoch):
                return epoch

            if epoch is None:
                self._epoch_condition.wait()
                return self._epoch_log[-1]
            else:
                self._epoch_condition.wait_for(lambda: self.epoch_completed(epoch))
                return epoch

    def epoch_completed(self, epoch: int) -> bool:
        """
        Check if the epoch has been completed.
        """
        with self._epoch_condition:
            return bool(
                (len(self._epochs) > 0 and epoch not in self._epochs and epoch < max(self._epochs))
                or (epoch in self._epochs and not self._epochs[epoch].is_active())
            )

    def _end_epoch(self, epoch: int) -> None:
        with self._epoch_condition:
            self._epoch_condition.notify_all()
            self._epoch_log.append(epoch)
            del self._epochs[epoch]

    def enable_node(self, node_id: str) -> None:
        """
        Enable edges attached to the node and the
        NodeSpecification enable switches to True

        """
        self.nodes[node_id].enabled = True

    def disable_node(self, node_id: str) -> None:
        """
        Disable edges attached to the node and the
        NodeSpecification enable switches to False

        """
        self.nodes[node_id].enabled = False

    @staticmethod
    def _init_graph(nodes: dict[str, NodeSpecification], edges: list[Edge]) -> TopologicalSorter:
        """
        Produce a :class:`.TopologicalSorter` based on the graph induced by
        a set of :class:`.Node` and a set of :class:`.Edge` that yields node ids.

        .. note:: Optional params

            Dependency graph only includes edges where `required == True` -
            aka even if we declare some dependency that passes a value to an
            optional (type annotation is `type | None`), default == `None`
            param, we still call that node even if that optional param is absent.

            Additionally, nodes with `enabled == False` are excluded, even when
            other nodes declare dependency to the disabled node. This means the
            signal of the disabled node will not be emitted and thus will not reach
            the dependent nodes. Disable nodes at your own risk.

            This behavior will likely change,
            allowing explicit parameterization of how optional values are handled,
            see: https://github.com/miniscope/noob/issues/26,

        """
        sorter = TopologicalSorter()
        enabled_nodes = [node_id for node_id, node in nodes.items() if node.enabled]
        for node_id in enabled_nodes:
            required_edges = [
                e.source_node
                for e in edges
                if e.target_node == node_id
                # and e.source_node in enabled_nodes
                and e.target_node in enabled_nodes
            ]
            sorter.add(node_id, *required_edges)
        return sorter
