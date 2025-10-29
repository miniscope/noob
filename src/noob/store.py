"""
Tube runners for running tubes
"""

from collections.abc import MutableSequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import count
from typing import Any

from noob.const import RESERVED_IDS
from noob.event import Event
from noob.node import Edge
from noob.node.base import Signal


@dataclass
class EventStore:
    """
    Container class for storing and retrieving events by node and slot
    """

    events: MutableSequence = field(default_factory=list)
    counter: count = field(default_factory=count)

    def add(
        self, signals: list[Signal], value: Any, node_id: str, epoch: int
    ) -> list[Event] | None:
        """
        Add the result of a :meth:`.Node.process` call to the event store.

        Split the dictionary of values into separate :class:`.Event` s,
        store along with current timestamp

        Args:
            signals (list[Signal]): Signals from which the value was emitted by
                a :meth:`.Node.process` call
            value (Any): Value emitted by a :meth:`.Node.process` call. Gets wrapped
                with a list in case the length of signals is 1. Otherwise, it's zipped
                with :signals:
            node_id (str): ID of the node that emitted the events
            epoch (int): Epoch count that the signal was emitted in
        """
        if value is None:
            return
        timestamp = datetime.now(UTC)

        values = [value] if len(signals) == 1 else value

        new_events = []
        for signal, val in zip(signals, values):
            new_event = Event(
                id=next(self.counter),
                timestamp=timestamp,
                node_id=node_id,
                epoch=epoch,
                signal=signal.name,
                value=val,
            )
            self.events.append(new_event)
            new_events.append(new_event)
        return new_events

    def get(self, node_id: str, signal: str, epoch: int) -> Event | None:
        """
        Get the event with the matching node_id and signal name

        Returns the most recent matching event, as for now we assume that
        each combination of `node_id` and `signal` is emitted only once per processing cycle,
        and we assume processing cycles are independent (and thus our events are cleared)

        ``None`` in the case that the event has not been emitted
        """
        event = [
            e
            for e in self.events
            if e["node_id"] == node_id and e["signal"] == signal and e["epoch"] == epoch
        ]
        return None if len(event) == 0 else event[-1]

    def collect(self, edges: list[Edge], epoch: int) -> dict | None:
        """
        Gather events into a form that can be consumed by a :meth:`.Node.process` method,
        given the collection of inbound edges (usually from :meth:`.Tube.in_edges` ).

        If none of the requested events have been emitted, return ``None``.

        If all of the requested events have been emitted, return a kwarg-like dict

        If some of the requested events are missing but others are present,
        return ``None`` for any missing events.

        .. todo::

            Add an example

        """
        args = {}
        for edge in edges:
            # FIXME: use reserved names in validation so we don't need this check
            if edge.source_node not in RESERVED_IDS:
                event = self.get(edge.source_node, edge.source_signal, epoch)
                value = None if event is None else event["value"]
                args[edge.target_slot] = value

        args = None if not args or all(val is None for val in args.values()) else args

        return args

    def clear(self, epoch: int | None = None) -> None:
        """
        Clear events for a specific or all epochs.

        Does not reset the counter (to continue giving unique ids to the next round's events)
        """

        if epoch is None:
            self.events = []
        else:
            self.events = [e for e in self.events if e["epoch"] != epoch]
