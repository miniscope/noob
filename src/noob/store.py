"""
Tube runners for running tubes
"""

from collections.abc import MutableSequence
from dataclasses import dataclass, field
from datetime import datetime
from itertools import count
from typing import Any

from noob.event import Event
from noob.node import Edge


@dataclass
class EventStore:
    """
    Container class for storing and retrieving events by node and slot
    """

    events: MutableSequence = field(default_factory=list)
    counter: count = field(default_factory=count)

    def add(self, signals: Any, value: Any, node_id: str) -> None:
        """
        Add the result of a :meth:`.Node.process` call to the event store.

        Split the dictionary of values into separate :class:`.Event` s,
        store along with current timestamp

        Args:
            signals (Any): Signals from which the value was emitted by a :meth:`.Node.process` call
            value (Any): Value emitted by a :meth:`.Node.process` call
            node_id (str): ID of the node that emitted the events
        """
        if value is None:
            return
        timestamp = datetime.now()

        values = [value] if len(signals) == 1 else value

        for signal, val in zip(signals, values):
            self.events.append(
                Event(
                    id=next(self.counter),
                    timestamp=timestamp,
                    node_id=node_id,
                    signal=signal,
                    value=val,
                )
            )

    def get(self, node_id: str, signal: str) -> Event | None:
        """
        Get the event with the matching node_id and signal name

        Returns the most recent matching event, as for now we assume that
        each combination of `node_id` and `signal` is emitted only once per processing cycle,
        and we assume processing cycles are independent (and thus our events are cleared)

        ``None`` in the case that the event has not been emitted
        """
        event = [e for e in self.events if e["node_id"] == node_id and e["signal"] == signal]
        return None if len(event) == 0 else event[-1]

    def gather(self, edges: list[Edge]) -> tuple[list | None, dict | None] | None:
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
        args = []
        kwargs = {}
        edges = self._sort_edges(edges)
        for edge in edges:
            event = self.get(edge.source_node, edge.source_signal)
            value = None if event is None else event["value"]
            if edge.target_slot:
                kwargs[edge.target_slot] = value
            else:
                args.append(value)

        args = None if not args or all(arg is None for arg in args) else args
        kwargs = None if not kwargs or all(val is None for val in kwargs.values()) else kwargs

        return args, kwargs

    def _sort_edges(self, edges: list[Edge]) -> list[Edge]:
        """
        Sort edges such that
        - positional arguments come first
        - positional arguments are in order
        - then kwargs come next and are also sorted
        """
        # FIXME: test this
        return sorted(
            edges, key=lambda item: (not isinstance(item.target_slot, int), item.target_slot)
        )

    def clear(self) -> None:
        """
        Clear events for this round of processing.

        Does not reset the counter (to continue giving unique ids to the next round's events)
        """
        self.events = []
