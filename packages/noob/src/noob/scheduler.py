from __future__ import annotations

import contextlib
import logging
from collections import defaultdict
from collections.abc import Iterator, MutableSequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

from noob_core import Scheduler as _RustScheduler

from noob.edge import Edge
from noob.event import Event, EventMaker, MetaEvent, MetaEventType, MetaSignal
from noob.exceptions import EpochCompletedError, EpochExistsError
from noob.logging import init_logger
from noob.node import NodeSpecification
from noob.types import Epoch, NodeID, NodeSignal, SignalName

if TYPE_CHECKING:
    from noob_core import SorterState

_VIRTUAL_NODES = ("input", "assets", "meta")
"""
Virtual nodes that don't actually exist as nodes,
but can be depended on 
(and can be present or absent, and so shouldn't be marked as trivially done)
"""


@dataclass()
class Scheduler:
    nodes: dict[str, NodeSpecification]
    edges: list[Edge]
    source_nodes: list[NodeID] = field(default_factory=list)
    event_maker: EventMaker = field(default_factory=EventMaker)
    _logger: logging.Logger = field(default_factory=lambda: init_logger("noob.scheduler"))

    def __post_init__(self):
        self._rebuild_core()

    @property
    def epoch_log(self) -> set[int]:
        return self._core.epoch_log

    @classmethod
    def from_specification(cls, nodes: dict[str, NodeSpecification], edges: list[Edge]) -> Self:
        """
        Create an instance of a Scheduler from :class:`.NodeSpecification` and :class:`.Edge`

        """
        return cls(nodes=nodes, edges=edges)

    @property
    def subepochs(self) -> dict[Epoch, set[Epoch]]:
        return self._core.subepochs

    @property
    def epoch(self) -> Epoch:
        """The current epoch that would run next/is running"""
        first = self._core.first_active_epoch()
        return first if first is not None else self.add_epoch()

    def iter_epoch(self, epoch: Epoch | None = None) -> Iterator[list[MetaEvent]]:
        """
        Iter batches of ready events from an epoch until it's completed.

        Args:
            epoch (Epoch): Epoch to iterate over.
                If ``None`` , adds an epoch to iterate over.

        Raises:
             EpochCompletedError if the requested epoch has already been completed
        """
        if epoch is None:
            first = self._core.first_active_epoch()
            epoch = first if first is not None else self.add_epoch()
        else:
            with contextlib.suppress(EpochExistsError):
                self._core.add_epoch_at(epoch)
        if self._core.epoch_completed(epoch):
            raise EpochCompletedError(f"Epoch {epoch} has already been completed")
        while self._core.is_active_at(epoch):
            yield self.get_ready(epoch)

    def iter_ready(self) -> Iterator[list[MetaEvent]]:
        """
        Iter batches of ready events from all epochs, infinitely,
        until no more nodes can be run in any epoch.

        TODO: document stateful/stateless behavior, need for callers to handle breaks in iteration
        """
        if not self._core.is_active():
            self._core.add_epoch()

        while self._core.is_active():
            ready = self.get_ready()
            if not ready:
                break
            yield ready

    def add_epoch(self, epoch: Epoch | None = None) -> Epoch:
        """
        Add another epoch with a prepared graph to the scheduler.
        """
        if epoch is None:
            return self._core.add_epoch()
        return self._core.add_epoch_at(epoch)

    def add_subepoch(self, epoch: Epoch) -> Epoch:
        """
        Add subepoch!

        Creates a topo sorter with all the nodes downstream of the node that created the epoch.
        """
        return self._core.add_epoch_at(epoch)

    def is_active(self, epoch: Epoch | None = None) -> bool:
        """
        Graph remains active while it holds at least one epoch that is active.
        """
        if epoch is None:
            return self._core.is_active()
        return self._core.is_active_at(epoch)

    def get_ready(self, epoch: Epoch | None = None) -> list[MetaEvent]:
        """
        Output the set of nodes that are ready across different epochs.

        Args:
            epoch (Epoch | None): if an Epoch, get ready events for that epoch,
                if ``None`` , get ready events for all epochs.
        """
        ready = self._core.get_ready() if epoch is None else self._core.get_ready_at(epoch)
        return [
            self.event_maker.new_meta_event(signal=MetaEventType.NodeReady, epoch=epoch, value=node)
            for epoch, node in ready
        ]

    def node_is_ready(self, node: NodeID, epoch: Epoch, subepochs: bool = False) -> bool:
        """
        Check if a single node is ready in a single or any epoch

        Args:
            node (NodeID): the node to check
            epoch (int | None): the epoch to check
            subepochs (bool): If ``True``, return ``True`` if ready in any subepoch too

        """
        return self._core.node_is_ready(node, epoch, subepochs)

    def node_is_done(self, node: NodeID, epoch: Epoch) -> bool:
        """Node is expired or done in specified epoch"""
        return self._core.node_is_done(node, epoch)

    def sources_finished(self, epoch: Epoch) -> bool:
        """
        Check the source nodes of the given epoch have been processed.
        If epoch is None, check the source nodes of the latest epoch.

        """
        return self._core.sources_finished(epoch)

    def update(
        self, events: MutableSequence[Event | MetaEvent] | MutableSequence[Event]
    ) -> MutableSequence[Event] | MutableSequence[Event | MetaEvent]:
        """
        When a set of events are received, update the graphs within the scheduler.
        Currently only has :meth:`TopoSorter.done` implemented.

        """
        if not events:
            return events

        ended = self._core.update(
            [
                (e["epoch"], e["node_id"], e["signal"], e["value"] is MetaSignal.NoEvent)
                for e in events
            ]
        )
        return [*events, *self._end_events(ended)]

    def done(
        self,
        epoch: Epoch,
        node_id: str,
        signal: SignalName | None = None,
        with_signals: bool = True,
    ) -> list[MetaEvent]:
        """
        Mark a node in a given epoch as done.

        Args:
            with_signals (bool): When marking this node as done, also mark all its signals as done.
        """
        return self._end_events(self._core.done(epoch, node_id, signal, with_signals))

    def expire(
        self,
        epoch: Epoch,
        node_id: str,
        signal: SignalName | None = None,
        with_signals: bool = True,
        unlock_optionals: bool = True,
    ) -> list[MetaEvent]:
        """
        Mark a node as having been completed without making its dependent nodes ready.
        i.e. when the node emitted ``NoEvent``
        """
        return self._end_events(
            self._core.expire(epoch, node_id, signal, with_signals, unlock_optionals)
        )

    def epoch_completed(self, epoch: Epoch) -> bool:
        """
        Check if the epoch has been completed -
        as in it is fully completed and should no longer be acted upon.
        Distinct from `is_active`, which is used to check if an epoch *should be* marked as complete
        """
        return self._core.epoch_completed(epoch)

    def end_epoch(self, epoch: Epoch) -> list[MetaEvent]:
        return self._end_events(self._core.end_epoch(epoch))

    def enable_node(self, node_id: str) -> None:
        """
        Enable edges attached to the node and the
        NodeSpecification enable switches to True.
        Enabling a node clears any existing epochs, so it should only be done
        between process calls.
        """
        self.nodes[node_id].enabled = True
        self._rebuild_core()

    def disable_node(self, node_id: str) -> None:
        """
        Disable edges attached to the node and the
        NodeSpecification enable switches to False

        """
        self.nodes[node_id].enabled = False
        self._rebuild_core()

    def clear(self) -> None:
        """
        Remove epoch records, restarting the scheduler
        """
        self._rebuild_core()

    def has_cycle(self) -> bool:
        """
        Checks that the graph is acyclic.
        """
        return self._core.has_cycle()

    def generations(self) -> list[list[NodeID]]:
        """
        Get the topological generations of the graph:
        tuples for each set of nodes that can be run at the same time.

        Order within a generation is not guaranteed to be stable.
        """
        return self._core.generations()

    def asset_generations(self) -> dict[NodeID, list[tuple[str, ...]]]:
        """
        :meth:`.generations` except only including nodes with direct dependencies on assets,
        to determine when the asset should be initialized vs. received in the ZMQ Runner.

        Packed in a dictionary with the asset ID as the key,
        and the value as the generations for that asset.
        """
        generations = defaultdict(list)
        asset_ids = set(e.source_signal for e in self.edges if e.source_node == "assets")
        for gen in self.generations():
            for asset in asset_ids:
                gen_deps = tuple(
                    [
                        g
                        for g in gen
                        if not isinstance(g, NodeSignal)
                        and any(
                            e.source_node == "assets"
                            and e.source_signal == asset
                            and e.target_node == g
                            for e in self.edges
                        )
                    ]
                )
                if gen_deps:
                    generations[asset].append(gen_deps)
        return generations

    def upstream_nodes(self, node: NodeID) -> set[NodeID]:
        """
        All the nodes that have an effect on the given node

        From:
        * Dependencies
        * If the node has optional dependencies, nodes whose NoEvents it should listen to
        """
        return self._core.upstream_nodes(node)

    def get_epoch_state(self, epoch: Epoch) -> SorterState:
        return self._core.get_epoch_state(epoch)

    # ---------------------
    # noob-core helpers
    # ---------------------

    def _rebuild_core(self) -> None:
        self._core = _RustScheduler(
            [(id_, bool(spec.enabled), spec.stateful) for id_, spec in self.nodes.items()],
            [(e.source_node, e.source_signal, e.target_node, e.required) for e in self.edges],
        )
        self.source_nodes = self._core.source_nodes()

    def _end_events(self, epochs: list[Epoch]) -> list[MetaEvent]:
        events = []
        for epoch in epochs:
            events.append(
                self.event_maker.new_meta_event(
                    signal=MetaEventType.EpochEnded, epoch=epoch, value=epoch
                )
            )
        return events
