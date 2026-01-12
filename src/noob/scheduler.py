import logging
from collections import deque
from collections.abc import MutableSequence
from datetime import UTC, datetime
from itertools import count
from threading import Condition
from typing import Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from noob.const import META_SIGNAL
from noob.event import Event, MetaEvent, MetaEventType, MetaSignal
from noob.exceptions import EpochCompletedError, EpochExistsError, NotOutYetError
from noob.logging import init_logger
from noob.node import Edge, NodeSpecification
from noob.toposort import TopoSorter
from noob.types import NodeID

_VIRTUAL_NODES = ("input", "assets")
"""
Virtual nodes that don't actually exist as nodes,
but can be depended on 
(and can be present or absent, and so shouldn't be marked as trivially done)
"""


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
            self.source_nodes = [id_ for id_ in graph.ready_nodes if id_ not in _VIRTUAL_NODES]
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
                if self.nodes[node_id].enabled
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
        # which marks them as "out" in the TopoSorter

        # if we've already run this, the node is ready - don't create another epoch
        if epoch in self._epoch_log:
            return True

        graphs = self._epochs.items() if epoch is None else [(epoch, self[epoch])]
        is_ready = any(node_id == node for epoch, graph in graphs for node_id in graph.ready_nodes)
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
        return all(src in graph.done_nodes for src in self.source_nodes)

    def update(
        self, events: MutableSequence[Event | MetaEvent] | MutableSequence[Event]
    ) -> MutableSequence[Event] | MutableSequence[Event | MetaEvent]:
        """
        When a set of events are received, update the graphs within the scheduler.
        Currently only has :meth:`TopoSorter.done` implemented.

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
                    epoch_ended = self.expire(epoch=e["epoch"], node_id=e["node_id"])
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
            except NotOutYetError:
                # in parallel mode, we don't `get_ready` the preceding ready nodes
                # so we have to manually mark them as "out"
                self[epoch].mark_out(node_id)
                self[epoch].done(node_id)

            self._ready_condition.notify_all()
            if not self[epoch].is_active():
                return self.end_epoch(epoch)
        return None

    def expire(self, epoch: int, node_id: str) -> MetaEvent | None:
        """
        Mark a node as having been completed without making its dependent nodes ready.
        i.e. when the node emitted ``NoEvent``
        """
        with self._ready_condition, self._epoch_condition:
            self[epoch].mark_expired(node_id)
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

            # mark just one event as "out."
            # threadsafe because we are holding the lock that protects graph mutation
            self._epochs[epoch].mark_out(node_id)

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
        Produce a :class:`.TopoSorter` based on the graph induced by
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
        return TopoSorter(nodes, edges)

    def has_cycle(self) -> bool:
        """
        Checks that the graph is acyclic.
        """
        graph = self._init_graph(nodes=self.nodes, edges=self.edges)
        cycle = graph.find_cycle()
        return bool(cycle)
