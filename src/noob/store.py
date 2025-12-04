"""
Tube runners for running tubes
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import count
from threading import Condition
from typing import Any, TypeAlias

from noob.event import Event, NoEvent
from noob.node import Edge
from noob.node.base import Signal
from noob.types import Epoch, NodeID, SignalName

EventDict: TypeAlias = dict[Epoch, dict[NodeID, dict[SignalName, list[Event]]]]
"""
A nested dictionary to store events for rapid access
(vs. the old implementation which was just a big list to filter).

Stores by epoch, node_id, and signal, like
events = {'epoch': {'node_id': {'signal': [...]}}}

Should be made with a `defaultdict` to avoid annoying nested indexing problems
"""


def _make_event_dict() -> EventDict:
    return defaultdict(lambda: defaultdict(lambda: defaultdict(list)))


@dataclass
class EventStore:
    """
    Container class for storing and retrieving events by node and slot
    """

    events: EventDict = field(default_factory=_make_event_dict)
    counter: count = field(default_factory=count)

    _event_condition: Condition = field(default_factory=Condition)

    def add(self, event: Event) -> Event:
        """
        Add an existing event to the store, returning it.

        Mostly an abstraction layer to give ourselves room above the `events` list
        in cast we want to change the internal implementation of how events are stored
        """
        with self._event_condition:
            self.events[event["epoch"]][event["node_id"]][event["signal"]].append(event)
            self._event_condition.notify_all()
        return event

    def add_value(
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
        if isinstance(value, NoEvent):
            return None

        with self._event_condition:
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
                self.add(new_event)
                new_events.append(new_event)

            self._event_condition.notify_all()
        return new_events

    def get(self, node_id: str, signal: str, epoch: int) -> Event:
        """
        Get the event with the matching node_id and signal name from a given epoch.

        If epoch is `-1`, return the most recent event.

        Raises:
            KeyError: if no event with the matching node_id and signal name exists
        """
        event = None
        if epoch == -1:
            for ep in reversed(self.events.keys()):
                if (evt := self.get(node_id, signal, ep)) is not None:
                    event = evt
                    break
        else:
            events = self.events[epoch][node_id][signal]
            event = events[-1] if events else None

        if event is None:
            raise KeyError(
                f"No event found for node_id: {node_id}, signal: {signal}, epoch: {epoch}"
            )
        else:
            return event

    def collect(self, edges: list[Edge], epoch: int) -> dict | None:
        """
        Gather events into a form that can be consumed by a :meth:`.Node.process` method,
        given the collection of inbound edges (usually from :meth:`.Tube.in_edges` ).

        If none of the requested events have been emitted, return ``None``.

        If all of the requested events have been emitted, return a kwarg-like dict

        If some of the requested events are missing but others are present,
        exclude the keys from the returned dict
        (`None` is a valid value for an event, so if a key is present and the value is `None`,
        the event was emitted with a value of `None`)

        If `epoch` is -1,
        get the the events from the most recent epoch where all events are present,
        and if no epochs are present with a full set of events,

        .. todo::

            Add an example

        """
        events = self.collect_events(edges, epoch)
        if events is None:
            return None

        return self.transform_events(edges=edges, events=events)

    def collect_events(self, edges: list[Edge], epoch: int) -> list[Event] | None:
        """
        Collect the event objects from a set of dependencies indicated by edges in a given epoch.

        If none of the requested events are present, return None

        If some of the requested events but others are present,
        return an incomplete list.

        If epoch is -1, get the latest complete set of events,
        returning None if no complete sets are present.

        Args:
            edges (list[Edge]): List of edges from which to collect events
            epoch (int): Epoch to select from, if -1, get the latest complete set of events.
        """

        if epoch == -1:
            for ep in reversed(self.events.keys()):
                events = self.collect_events(edges, ep)
                if self.dependencies_satisfied(edges, events):
                    return events

            return None

        else:
            events = []
            for edge in edges:
                try:
                    event = self.get(edge.source_node, edge.source_signal, epoch)
                    events.append(event)
                except KeyError:
                    # no matching event
                    continue

            return events if events else None

    def await_events(self, edges: list[Edge], epoch: int) -> list[Event]:
        """
        Block until a set of events implied by `edges` are all present, returning the events.

        Args:
            edges (list[:class:`.Edge`]): dependencies to wait for
            epoch (int | None): If an int, wait for this specific epoch to be ready,
              if `None` await *any* epoch to be ready.
        """
        with self._event_condition:
            # if we already have dependencies ready, just return them
            events = self.collect_events(edges, epoch)
            if events is not None and self.dependencies_satisfied(edges, events):
                return events

            # otherwise, wait until they are available
            self._event_condition.wait_for(lambda: self.dependencies_ready(edges, epoch))
            return self.collect_events(edges, epoch)

    def await_dependencies(self, edges: list[Edge], epoch: int | None = None) -> dict:
        """
        like :meth:`.await_events`, except transforming the events into node kwargs before returning
        """
        return self.transform_events(edges, self.await_events(edges, epoch))

    def dependencies_ready(self, edges: list[Edge], epoch: int | None = None) -> bool:
        """
        Check if dependencies are ready for a given dependency set and epoch
        Args:
            edges (list[:class:`.Edge`]): dependencies to check
            epoch (int | None): If an int, check this specific epoch, otherwise check any epoch.

        Returns:
            bool
        """
        epoch = -1 if epoch is None else epoch
        events = self.collect_events(edges, epoch=epoch)
        return events is not None and self.dependencies_satisfied(edges, events)

    def clear(self, epoch: int | None = None) -> None:
        """
        Clear events for a specific or all epochs.

        Does not reset the counter (to continue giving unique ids to the next round's events)
        """

        if epoch is None:
            self.events = _make_event_dict()
        else:
            del self.events[epoch]

    @staticmethod
    def dependencies_satisfied(edges: list[Edge], events: list[Event]) -> bool:
        """
        For a given set of edges and events, check if the events satisfy the edge's dependencies
        (i.e. that there is at least one event for each specified source node/signal pair)
        """
        edge_set = set((edge.source_node, edge.source_signal) for edge in edges)
        event_set = set((event["node_id"], event["signal"]) for event in events)
        return edge_set == event_set

    @staticmethod
    def transform_events(edges: list[Edge], events: list[Event]) -> dict:
        """
        Transform the values of a set of events to a dict that can be consumed by the
        target node's process method.

        i.e.: return a dictionary whose keys are the ``target_signal`` s of the edges
           using the ``value`` of the matching event.
        """
        args = {}
        for edge in edges:
            evts = [
                e
                for e in events
                if e["node_id"] == edge.source_node and e["signal"] == edge.source_signal
            ]
            if not evts:
                continue
            args[edge.target_slot] = evts[-1]["value"]
        return args

    @staticmethod
    def split_args_kwargs(inputs: dict) -> tuple[list, dict]:
        # TODO: integrate this with `collect`
        args = []
        kwargs = {}
        for k, v in inputs.items():
            if isinstance(k, int | None):
                args.append((k, v))
            else:
                kwargs[k] = v
        args = [item[1] for item in sorted(args, key=lambda x: x[0])]
        return args, kwargs
