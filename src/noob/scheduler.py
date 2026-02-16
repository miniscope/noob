import logging
from collections import defaultdict, deque
from collections.abc import MutableSequence
from datetime import UTC, datetime
from itertools import count
from typing import Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from noob.const import META_SIGNAL
from noob.event import Event, MetaEvent, MetaEventType, MetaSignal
from noob.exceptions import AlreadyDoneError, EpochCompletedError, EpochExistsError, NotOutYetError
from noob.logging import init_logger
from noob.node import Edge, NodeSpecification
from noob.toposort import TopoSorter
from noob.types import Epoch, NodeID

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
    _epochs: dict[Epoch, TopoSorter] = PrivateAttr(default_factory=dict)
    _subepochs: dict[Epoch, set[Epoch]] = PrivateAttr(default_factory=lambda: defaultdict(set))
    _epoch_log: deque[int] = PrivateAttr(default_factory=lambda: deque(maxlen=100))
    _subgraphs: dict[NodeID, tuple[dict[str, NodeSpecification], list[Edge]]] = PrivateAttr(
        default_factory=dict
    )

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
            graph = self._init_graph()
            self.source_nodes = [id_ for id_ in graph.ready_nodes if id_ not in _VIRTUAL_NODES]
        return self

    def add_epoch(self, epoch: int | Epoch | None = None) -> Epoch:
        """
        Add another epoch with a prepared graph to the scheduler.
        """
        if epoch is not None:
            if isinstance(epoch, int):
                this_epoch = Epoch(epoch)
            elif isinstance(epoch, Epoch):
                this_epoch = epoch
            else:
                raise TypeError("Can only create an epoch from an epoch or integer")
            # ensure that the next iteration of the clock will return the next number
            # if we create epochs out of order
            self._clock = count(
                max([this_epoch[0].epoch, *[ep[0].epoch for ep in self._epochs], *self._epoch_log])
                + 1
            )
        else:
            this_epoch = Epoch(next(self._clock))

        if this_epoch in self._epochs:
            raise EpochExistsError(f"Epoch {this_epoch} is already scheduled")
        elif this_epoch in self._epoch_log:
            raise EpochCompletedError(f"Epoch {this_epoch} has already been completed!")

        graph = self._init_graph(epoch=this_epoch)
        self._epochs[this_epoch] = graph

        # handle subepochs
        if this_epoch.parent is not None:
            for parent in this_epoch.parents:
                self._subepochs[parent].add(this_epoch)

            # a node inducing subepochs expires the node in the (immediate) parent epoch
            if this_epoch[-1].node_id not in self._epochs[this_epoch.parent].done_nodes:
                self._epochs[this_epoch.parent].mark_expired(this_epoch[-1].node_id)

        return this_epoch

    def is_active(self, epoch: Epoch | None = None) -> bool:
        """
        Graph remains active while it holds at least one epoch that is active.

        """
        if epoch is not None:
            if epoch not in self._epochs:
                # if an epoch has been completed and had its graph cleared, it's no longer active
                # if an epoch has not been started, it is also not active.
                return False
            return any(self._epochs[e].is_active() for e in {*self._subepochs[epoch], epoch})
        else:
            return any(graph.is_active() for graph in self._epochs.values())

    def get_ready(self, epoch: Epoch | None = None) -> list[MetaEvent]:
        """
        Output the set of nodes that are ready across different epochs.

        Args:
            epoch (Epoch | None): if an Epoch, get ready events for that epoch,
                if ``None`` , get ready events for all epochs.
        """

        if epoch is not None:
            graphs = [(ep, self._epochs[ep]) for ep in {*self._subepochs[epoch], epoch}]
        else:
            graphs = list(self._epochs.items())

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
            if node_id in _VIRTUAL_NODES or self.nodes[node_id].enabled
        ]
        return ready_nodes

    def node_is_ready(self, node: NodeID, epoch: Epoch | None = None) -> bool:
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

    def __getitem__(self, epoch: Epoch | int) -> TopoSorter:
        if epoch == -1:
            if len(self._epochs) == 1:
                return next(iter(self._epochs.values()))
            else:
                max_epoch = max(*[e[0].epoch for e in self._epochs])
                return self._epochs[Epoch(max_epoch)]
        elif isinstance(epoch, int):
            epoch = Epoch(epoch)

        if epoch not in self._epochs:
            self.add_epoch(epoch)
        return self._epochs[epoch]

    def sources_finished(self, epoch: Epoch | None = None) -> bool:
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
        marked_done = set()
        for e in events:
            if (done_marker := (e["epoch"], e["node_id"])) in marked_done or e["node_id"] == "meta":
                continue
            else:
                marked_done.add(done_marker)

            if e["signal"] == META_SIGNAL and e["value"] == MetaSignal.NoEvent:
                epoch_ended = self.expire(epoch=e["epoch"], node_id=e["node_id"])
            else:
                epoch_ended = self.done(epoch=e["epoch"], node_id=e["node_id"])

            if epoch_ended:
                end_events.append(epoch_ended)

        ret_events = [*events, *end_events]

        return ret_events

    def done(self, epoch: Epoch, node_id: str) -> MetaEvent | None:
        """
        Mark a node in a given epoch as done.
        """
        if epoch[0].epoch in self._epoch_log:
            self.logger.debug(
                "Marking node %s as done in epoch %s, " "but epoch was already completed. ignoring",
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
        except AlreadyDoneError as e:
            # if we induced subepochs but emitted another event in the parent epoch,
            # we were marked expired when the subepochs were created
            # (which is fine)
            if node_id not in self[epoch].ran_nodes and any(
                subep[-1].node_id == node_id for subep in self._subepochs[epoch]
            ):
                return None
            else:
                raise e

        self._done_subepochs(epoch, node_id)

        if not self[epoch].is_active():
            return self.end_epoch(epoch)
        return None

    def expire(self, epoch: Epoch, node_id: str) -> MetaEvent | None:
        """
        Mark a node as having been completed without making its dependent nodes ready.
        i.e. when the node emitted ``NoEvent``
        """
        self[epoch].mark_expired(node_id)
        if not self[epoch].is_active():
            return self.end_epoch(epoch)

        return None

    def epoch_completed(self, epoch: Epoch) -> bool:
        """
        Check if the epoch has been completed.
        """
        previously_completed = (
            len(self._epoch_log) > 0
            and epoch not in self._epochs
            and (epoch in self._epoch_log or epoch < min(self._epoch_log))
        )
        active_completed = epoch in self._epochs and not self._epochs[epoch].is_active()
        return previously_completed or active_completed

    def end_epoch(self, epoch: Epoch | int | None = None) -> MetaEvent | None:
        if epoch is None or epoch == -1:
            if len(self._epochs) == 0:
                return None
            ep = list(self._epochs)[-1]
        elif isinstance(epoch, int):
            ep = Epoch(epoch)
        elif isinstance(epoch, Epoch):
            ep = epoch
        else:
            raise TypeError("Can only end an epoch with an integer or Epoch")
        self.logger.debug("Ending epoch %s", ep)
        if len(ep) == 1:
            self._epoch_log.append(ep[0].epoch)
            del self._epochs[ep]
            # FIXME: cleanup subepochs

        return MetaEvent(
            id=uuid4().int,
            timestamp=datetime.now(UTC),
            node_id="meta",
            signal=MetaEventType.EpochEnded,
            epoch=ep,
            value=ep,
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
        for graph in self._epochs.values():
            graph.mark_expired(node_id)

    def clear(self) -> None:
        """
        Remove epoch records, restarting the scheduler
        """
        self._epochs = {}
        self._epoch_log = deque(maxlen=100)

    def _init_graph(self, epoch: Epoch | None = None) -> TopoSorter:
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
        if epoch and epoch.parent:
            nodes, edges = self._subgraph(epoch[-1].node_id)
            sorter = TopoSorter(nodes, edges)
            # mark any nodes that are completed in the parent as completed in the subepoch
            parent_epoch = self[epoch.parent]
            parent_deps = set(sorter.node_info) - set(nodes)
            self.logger.debug("Marking parent deps expired or done: %s", parent_deps)
            for parent_dep in set(sorter.node_info) - set(nodes):
                if parent_dep in parent_epoch.ran_nodes:
                    sorter.mark_out(parent_dep)
                    sorter.done(parent_dep)
                elif parent_dep in parent_epoch.done_nodes:
                    sorter.mark_expired(parent_dep)
            return sorter
        else:
            return TopoSorter(self.nodes, self.edges)

    def has_cycle(self) -> bool:
        """
        Checks that the graph is acyclic.
        """
        graph = self._init_graph()
        cycle = graph.find_cycle()
        return bool(cycle)

    def generations(self) -> list[tuple[str, ...]]:
        """
        Get the topological generations of the graph:
        tuples for each set of nodes that can be run at the same time.

        Order within a generation is not guaranteed to be stable.
        """
        sorter = self._init_graph()
        generations = []
        while sorter.is_active():
            ready = sorter.get_ready()
            generations.append(ready)
            sorter.done(*ready)
        return generations

    def _subgraph(self, node_id: str) -> tuple[dict[str, NodeSpecification], list[Edge]]:
        """
        Subgraph that is downstream of a given node (including the node itself).
        """
        from noob.tube import downstream_nodes

        if node_id not in self._subgraphs:
            downstream = downstream_nodes(self.edges, node_id)
            self._subgraphs[node_id] = (
                {node_id: self.nodes[node_id] for node_id in downstream},
                [e for e in self.edges if e.target_node in downstream],
            )
        return self._subgraphs[node_id]

    def _done_subepochs(self, epoch: Epoch, node_id: NodeID) -> None:
        """
        Called when a node in a parent epoch is marked done -
        mark the node done in all subepochs,
        but ensure that nodes that are exclusively downstream of this node
        (i.e. no dependencies on nodes within the mapped subepoch)
        are removed from the graph.

        This is to support gather-like operations from non-gather nodes in 3rd party tubes:
        nodes downstream of both this node and other nodes in the subepoch run in subepochs,
        but nodes that are exclusively downstream of this node only run in the parent epoch
        """
        from noob.tube import downstream_nodes

        if not self._subepochs[epoch]:
            return

        our_subgraph = set(self._subgraph(node_id)[0])
        _exclusive_subgraphs = {}

        for subepoch in self._subepochs[epoch]:
            if (
                node_id in self._epochs[subepoch].ran_nodes
                or node_id not in self._epochs[subepoch].node_info
                or subepoch[-1].node_id == node_id
            ):
                # fine
                continue
            elif node_id in self._epochs[subepoch].done_nodes:
                # needs to be resurrected
                self._epochs[subepoch].resurrect(node_id)

            if node_id not in self._epochs[subepoch].out_nodes:
                self._epochs[subepoch].mark_out(node_id)
            self._epochs[subepoch].done(node_id)

            # mark all nodes that are exclusively downstream of this node expired
            subep_node = subepoch[-1].node_id
            if subep_node not in _exclusive_subgraphs:
                _exclusive_subgraphs[subep_node] = downstream_nodes(
                    self.edges, subep_node, exclude={node_id}
                )
            exclusive_subgraph = our_subgraph - _exclusive_subgraphs[subep_node] - {node_id}

            for exclusive in exclusive_subgraph:
                self._epochs[subepoch].mark_expired(exclusive)
