from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from functools import partial
from logging import Logger
from typing import Any, TypeVar

from noob import Tube, init_logger
from noob.event import Event
from noob.node import Node, Return
from noob.store import EventStore
from noob.types import PythonIdentifier, ReturnNodeType, RunnerContext

TInit = TypeVar("TInit", bound=Callable)


@dataclass
class TubeRunner(ABC):
    """
    Abstract parent class for tube runners.

    Tube runners handle calling the nodes and passing the
    events returned by them to each other. Each runner may do so
    however it needs to (synchronously, asynchronously, alone or as part of a cluster, etc.)
    as long as it satisfies this abstract interface.
    """

    tube: Tube
    store: EventStore = field(default_factory=EventStore)

    _callbacks: list[Callable[[Event], None]] = field(default_factory=list)

    _logger: Logger = field(default_factory=lambda: init_logger("tube.runner"))

    @abstractmethod
    def process(self, **kwargs: Any) -> ReturnNodeType:
        """
        Process one step of data from each of the sources,
        passing intermediate data to any subscribed nodes in a chain.

        The `process` method normally does not return anything,
        except when using the special :class:`.Return` node

        Process-scoped ``input`` s can be passed as kwargs.
        """

    @abstractmethod
    def init(self) -> None:
        """
        Start processing data with the tube graph.

        Implementations of this method must raise a :class:`.TubeRunningError`
        if the tube has already been started and is running,
        (i.e. :meth:`.deinit` has not been called,
        or the tube has not exhausted itself)
        """

    @abstractmethod
    def deinit(self) -> None:
        """
        Stop processing data with the tube graph
        """

    def run(self, n: int | None = None) -> None | list[ReturnNodeType]:
        """Run the tube either indefinitely or for a fixed number of complete iterations!"""
        raise NotImplementedError()

    def iter(self, n: int | None = None) -> Generator[ReturnNodeType, None, None]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def running(self) -> bool:
        """
        Whether the tube is currently running
        """
        pass

    def collect_input(
        self, node: Node, epoch: int, input: dict | None = None
    ) -> tuple[list[Any] | None, dict[PythonIdentifier, Any] | None] | None:
        """
        Gather input to give to the passed Node from the :attr:`.TubeRunner.store`

        Returns:
            dict: kwargs to pass to :meth:`.Node.process` if matching events are present
            dict: empty dict if Node is a :class:`.Source`
            None: if no input is available
        """
        if not node.spec.depends:
            return None, None
        if input is None:
            input = {}

        edges = self.tube.in_edges(node)

        inputs = {}

        state_inputs = self.tube.state.collect(edges)
        inputs |= state_inputs if state_inputs else inputs

        event_inputs = self.store.collect(edges, epoch)
        inputs |= event_inputs if event_inputs else inputs

        input_inputs = self.tube.input_collection.collect(edges, input)
        inputs |= input_inputs if input_inputs else inputs

        args = []
        kwargs = {}
        for k, v in inputs.items():
            # Handle positional arguments (int keys) and keyword arguments (string keys)
            # Also handle None keys which indicate scalar positional arguments
            if isinstance(k, int):
                args.append((k, v))
            elif k is None:
                # None key means scalar positional argument (first position)
                args.append((0, v))
            else:
                kwargs[k] = v
        
        # Sort args by position index and extract values
        args = [item[1] for item in sorted(args, key=lambda x: x[0])]

        return args, kwargs

    def collect_return(self) -> ReturnNodeType:
        """
        If any :class:`.Return` nodes are in the tube,
        gather their return values to return from :meth:`.TubeRunner.process`

        Returns:
            dict: of the Return sink's key mapped to the returned value,
            None: if there are no :class:`.Return` sinks in the tube
        """
        ret_nodes = [n for n in self.tube.enabled_nodes.values() if isinstance(n, Return)]
        if not ret_nodes:
            return None
        ret_node = ret_nodes[0]
        return ret_node.get(keep=False)

    def add_callback(self, callback: Callable[[Event], None]) -> None:
        self._callbacks.append(callback)

    def _call_callbacks(self, events: list[Event] | None) -> None:
        if not events:
            return
        for event in events:
            for callback in self._callbacks:
                callback(event)

    @abstractmethod
    def enable_node(self, node_id: str) -> None:
        """
        A method for enabling a node during runtime
        """
        pass

    @abstractmethod
    def disable_node(self, node_id: str) -> None:
        """
        A method for disabling a node during runtime
        """
        pass

    def get_context(self) -> RunnerContext:
        return RunnerContext(runner=self, tube=self.tube)

    def inject_context(self, fn: TInit) -> TInit:
        """Wrap function in a partial with the runner context injected, if requested"""
        sig = inspect.signature(fn)
        ctx_key = [
            k for k, v in sig.parameters.items() if v.annotation and v.annotation is RunnerContext
        ]
        if ctx_key:
            return partial(fn, **{ctx_key[0]: self.get_context()})
        else:
            return fn
