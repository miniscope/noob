import contextlib
import logging
from collections import deque
from collections.abc import MutableSequence
from copy import deepcopy
from datetime import UTC, datetime
from enum import StrEnum
from graphlib import _NODE_OUT  # type: ignore[attr-defined]
from itertools import count
from threading import Condition
from typing import Self
from uuid import uuid4

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from noob.const import META_SIGNAL
from noob.event import Event, MetaEvent, MetaEventType, MetaSignal
from noob.exceptions import EpochCompletedError, EpochExistsError
from noob.logging import init_logger
from noob.node import Edge, NodeSpecification
from noob.types import NodeID


class NodeState(StrEnum):
    ready = "ready"
    out = "out"
    done = "done"
    n_preds = "n_preds"


class TopoSorter(nx.DiGraph):
    @property
    def source_nodes(self) -> set[NodeID]:
        return {id_ for id_, deg in self.in_degree if deg == 0}

    @property
    def ready_nodes(self) -> set[NodeID]:
        return {id_ for id_, node in self.nodes.items() if node[NodeState.ready]}

    @property
    def out_nodes(self) -> set[NodeID]:
        return {id_ for id_, node in self.nodes.items() if node[NodeState.out]}

    @property
    def done_nodes(self) -> set[NodeID]:
        return {id_ for id_, node in self.nodes.items() if node[NodeState.done]}

    def init_attrs(self) -> None:
        nx.set_node_attributes(self, dict(self.in_degree), NodeState.n_preds)
        for state in (NodeState.ready, NodeState.out, NodeState.done):
            nx.set_node_attributes(self, False, state)

    def mark_ready(self, *nodes: NodeID) -> None:
        nx.set_node_attributes(self, {node: True for node in nodes}, NodeState.ready)
        nx.set_node_attributes(self, {node: False for node in nodes}, NodeState.out)
        nx.set_node_attributes(self, {node: False for node in nodes}, NodeState.done)

    def mark_out(self, *nodes: NodeID) -> None:
        nx.set_node_attributes(self, {node: False for node in nodes}, NodeState.ready)
        nx.set_node_attributes(self, {node: True for node in nodes}, NodeState.out)
        nx.set_node_attributes(self, {node: False for node in nodes}, NodeState.done)

    def mark_done(self, *nodes: NodeID) -> None:
        nx.set_node_attributes(self, {node: False for node in nodes}, NodeState.ready)
        nx.set_node_attributes(self, {node: False for node in nodes}, NodeState.out)
        nx.set_node_attributes(self, {node: True for node in nodes}, NodeState.done)

    def add(self, node_id: NodeID, *preds: NodeID) -> None:
        self.add_node(node_id)
        self.add_edges_from([(pred, node_id) for pred in preds])

    def prepare(self) -> None:
        """
        We're allowing cycles when assets are involved,
        so we make sure there are no cycles among nodes only.

        Source nodes are marked ready.
        """
        G_nodes = deepcopy(self)
        cycle = None
        if "assets" in G_nodes:
            G_nodes.remove_node("assets")
        with contextlib.suppress(nx.NetworkXNoCycle):
            cycle = nx.find_cycle(G_nodes)
        if cycle:
            raise nx.NetworkXError(f"Cycle among nodes detected: {cycle}")
        nx.set_node_attributes(self, {node: True for node in self.source_nodes}, NodeState.ready)

    def is_active(self) -> bool:
        """
        Active if there are nodes currently processing or
        ready to be processed
        """
        return any(self.out_nodes or self.ready_nodes)

    def get_ready(self) -> set[NodeID]:
        """
        Mark ready nodes as out before sending them out
        """
        ready_nodes = self.ready_nodes
        self.mark_out(*ready_nodes)
        return ready_nodes

    def done(self, *node_ids: NodeID) -> None:
        """
        Marks a set of nodes returned by "get_ready" as processed.

        This method unblocks any successor of each node in *nodes* for being returned
        in the future by a call to "get_ready".

        Raises ValueError if any node in *nodes* has already been marked as
        processed by a previous call to this method, if a node was not added to the
        graph by using "add" or if called without calling "prepare" previously or if
        node has not yet been returned by "get_ready".
        """
        nx.set_node_attributes(self, {node: False for node in node_ids}, NodeState.out)
        nx.set_node_attributes(self, {node: True for node in node_ids}, NodeState.done)

        # next set of nodes need to be ready
        for node in node_ids:
            for succ in self.successors(node):
                self.nodes[succ][NodeState.n_preds] -= 1
                if self.nodes[succ][NodeState.n_preds] == 0:
                    self.mark_ready(succ)


class Scheduler(BaseModel):
    nodes: dict[str, NodeSpecification]
    edges: list[Edge]
    source_nodes: list[NodeID] = Field(default_factory=list)
    logger: logging.Logger = Field(default_factory=lambda: init_logger("noob.scheduler"))

    _clock: count = PrivateAttr(default_factory=count)
    _epochs: dict[int, TopoSorter] = PrivateAttr(default_factory=dict)
    _ready_condition: Condition = PrivateAttr(default_factory=Condition)
    _epoch_condition: Condition = PrivateAttr(default_factory=Condition)
    _epoch_log: deque = PrivateAttr(default_factory=lambda: deque(maxlen=100))

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
                id_ for id_, deg in graph.in_degree if deg == 0 and id_ not in ("input", "assets")
            ]
        return self

    def add_epoch(self, epoch: int | None = None) -> int:
        """
        Add another epoch with a prepared graph to the scheduler.
        """
        with self._ready_condition:
            if epoch:
                this_epoch = epoch
                # ensure that the next iteration of the clock will return the next number
                # if we create epochs out of order
                self._clock = count(max([this_epoch, *self._epochs.keys(), *self._epoch_log]) + 1)
            else:
                this_epoch = next(self._clock)

            if this_epoch in self._epochs:
                raise EpochExistsError(f"Epoch {this_epoch} is already scheduled")
            elif this_epoch in self._epoch_log:
                raise EpochCompletedError(f"Epoch {this_epoch} has already been completed!")

            graph = self._init_graph(nodes=self.nodes, edges=self.edges)
            graph.prepare()
            self._epochs[this_epoch] = graph
            self._ready_condition.notify_all()
        return this_epoch

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
            ready_nodes = [
                MetaEvent(
                    id=uuid4().int,
                    timestamp=datetime.now(),
                    node_id="meta",
                    signal=MetaEventType.NodeReady,
                    epoch=epoch,
                    value=node_id,
                )
                for epoch, graph in graphs
                for node_id in graph.get_ready()
            ]

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

        # if we've already run this, the node is ready - don't create another epoch
        if epoch in self._epoch_log:
            return True

        graphs = self._epochs.items() if epoch is None else [(epoch, self[epoch])]
        is_ready = any(node_id == node for epoch, graph in graphs for node_id in graph._ready_nodes)  # type: ignore[attr-defined]
        return is_ready

    def __getitem__(self, epoch: int) -> TopoSorter:
        if epoch == -1:
            return self._epochs[max(self._epochs.keys())]

        if epoch not in self._epochs:
            self.add_epoch(epoch)
        return self._epochs[epoch]

    def sources_finished(self, epoch: int | None = None) -> bool:
        """
        Check the source nodes of the given epoch have been processed.
        If epoch is None, check the source nodes of the latest epoch.

        """
        if epoch is None and len(self._epochs) == 0:
            return True

        graph = self[-1] if epoch is None else self._epochs[epoch]
        return all(
            graph._node2info[src].npredecessors != _NODE_OUT  # type: ignore[attr-defined]
            and src not in graph._ready_nodes  # type: ignore[attr-defined]
            for src in self.source_nodes
        )

    def update(
        self, events: MutableSequence[Event | MetaEvent] | MutableSequence[Event]
    ) -> MutableSequence[Event] | MutableSequence[Event | MetaEvent]:
        """
        When a set of events are received, update the graphs within the scheduler.
        Currently only has :meth:`TopologicalSorter.done` implemented.

        """
        if not events:
            return events
        end_events: MutableSequence[MetaEvent] = []
        with self._ready_condition, self._epoch_condition:
            marked_done = set()
            for e in events:
                if (done_marker := (e["epoch"], e["node_id"])) in marked_done or e[
                    "node_id"
                ] == "meta":
                    continue
                else:
                    marked_done.add(done_marker)

                if e["signal"] == META_SIGNAL and e["value"] == MetaSignal.NoEvent:
                    epoch_ended = self.cancel(epoch=e["epoch"], node_id=e["node_id"])
                else:
                    epoch_ended = self.done(epoch=e["epoch"], node_id=e["node_id"])

                if epoch_ended:
                    end_events.append(epoch_ended)

            # condition uses an RLock, so waiters only run here,
            # even though `done` also notifies.
            self._ready_condition.notify_all()

        ret_events = [*events, *end_events]

        return ret_events

    def done(self, epoch: int, node_id: str) -> MetaEvent | None:
        """
        Mark a node in a given epoch as done.
        """

        with self._ready_condition, self._epoch_condition:
            if epoch in self._epoch_log:
                self.logger.debug(
                    "Marking node %s as done in epoch %s, "
                    "but epoch was already completed. ignoring",
                    node_id,
                    epoch,
                )
                return None

            try:
                self[epoch].done(node_id)
            except ValueError as e:
                # in parallel mode, we don't `get_ready` the preceding ready nodes
                # so we have to manually mark them as "out"
                # FIXME: so ugly - need to make our own topo sorter
                if node_id not in self[epoch]._node2info:  # type: ignore[attr-defined]
                    raise e
                self[epoch]._node2info[node_id].npredecessors = _NODE_OUT  # type: ignore[attr-defined]
                self[epoch]._nfinished += 1  # type: ignore[attr-defined]
                if node_id in self[epoch]._ready_nodes:  # type: ignore[attr-defined]
                    self[epoch]._ready_nodes.remove(node_id)  # type: ignore[attr-defined]
                self[epoch].done(node_id)

            self._ready_condition.notify_all()
            if not self[epoch].is_active():
                return self.end_epoch(epoch)
        return None

    def cancel(self, epoch: int, node_id: str) -> MetaEvent | None:
        """
        Mark a node as completed without making its dependent nodes ready
        """
        with self._ready_condition, self._epoch_condition:
            self[epoch].mark_done(node_id)
            self._ready_condition.notify_all()
            if not self[epoch].is_active():
                return self.end_epoch(epoch)

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

            # be FIFO-like and get the earliest epoch the node is ready in
            if epoch is None:
                for ep in self._epochs:
                    if self.node_is_ready(node_id, ep):
                        epoch = ep
                        break

            if epoch is None:
                raise RuntimeError(
                    "Could not find ready epoch even though node ready condition passed, "
                    "something is wrong with the way node status checking is "
                    "locked between threads."
                )

            # do a little graphlib surgery to mark just one event as done.
            # threadsafe because we are holding the lock that protects graph mutation
            self._epochs[epoch]._node2info[node_id].npredecessors = _NODE_OUT  # type: ignore[attr-defined]
            self._epochs[epoch]._ready_nodes.remove(node_id)  # type: ignore[attr-defined]

        return MetaEvent(
            id=uuid4().int,
            timestamp=datetime.now(),
            node_id="meta",
            signal=MetaEventType.NodeReady,
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
            previously_completed = (
                len(self._epoch_log) > 0 and epoch not in self._epochs and epoch in self._epoch_log
            )
            active_completed = epoch in self._epochs and not self._epochs[epoch].is_active()
            return previously_completed or active_completed

    def end_epoch(self, epoch: int | None = None) -> MetaEvent | None:
        if epoch is None:
            if len(self._epochs) == 0:
                return None
            epoch = list(self._epochs)[-1]

        with self._epoch_condition:
            self._epoch_condition.notify_all()
            self._epoch_log.append(epoch)
            del self._epochs[epoch]

        return MetaEvent(
            id=uuid4().int,
            timestamp=datetime.now(UTC),
            node_id="meta",
            signal=MetaEventType.EpochEnded,
            epoch=epoch,
            value=epoch,
        )

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

    def clear(self) -> None:
        """
        Remove epoch records, restarting the scheduler
        """
        self._epochs = {}
        self._epoch_log = deque(maxlen=100)

    @staticmethod
    def _init_graph(nodes: dict[str, NodeSpecification], edges: list[Edge]) -> TopoSorter:
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
        sorter: TopoSorter = TopoSorter()
        for node_id in nodes:
            required_edges = [
                e.source_node for e in edges if e.target_node == node_id and e.target_node in nodes
            ]
            sorter.add(node_id, *required_edges)
        disabled_nodes = [node_id for node_id, node in nodes.items() if not node.enabled]
        for node_id in disabled_nodes:
            sorter.remove_node(node_id)
        sorter.init_attrs()
        return sorter
