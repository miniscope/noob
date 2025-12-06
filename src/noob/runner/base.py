from __future__ import annotations

import hashlib
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from logging import Logger
from typing import Any, Self

from noob import Tube, init_logger
from noob.event import Event, MetaEvent
from noob.node import Node
from noob.store import EventStore
from noob.types import PythonIdentifier, ReturnNodeType, RunnerContext


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
    max_iter_loops: int = 100
    """The max number of times that `iter` will call `process` to try and get a result"""

    _callbacks: list[Callable[[Event | MetaEvent], None]] = field(default_factory=list)

    _logger: Logger | None = None
    _runner_id: str | None = None

    def __post_init__(self):
        self._logger = init_logger(f"noob.runner.{self.runner_id}")

    @property
    def runner_id(self) -> str:
        if self._runner_id is None:
            hasher = hashlib.blake2b(digest_size=4)
            hasher.update(str(datetime.now(UTC).timestamp()).encode("utf-8"))
            self._runner_id = f"{hasher.hexdigest()}.{self.tube.tube_id}"
        return self._runner_id

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

    def iter(self, n: int | None = None) -> Generator[ReturnNodeType, None, None]:
        """
        Treat the runner as an iterable.

        Calls :meth:`.TubeRunner.process` until it yields a result
        (e.g. multiple times in the case of any ``gather`` s
        that change the cardinality of the graph.)
        """

        self.init()
        current_iter = 0
        try:
            while n is None or current_iter < n:
                ret = None
                loop = 0
                while ret is None:
                    ret = self.process()
                    loop += 1
                    if loop > self.max_iter_loops:
                        raise RuntimeError("Reached maximum process calls per iteration")

                yield ret
                current_iter += 1
        finally:
            self.deinit()

    def run(self, n: int | None = None) -> None | list[ReturnNodeType]:
        outputs = []
        current_iter = 0
        if not self.running:
            self.init()
        try:
            while n is None or current_iter < n:
                out = self.process()
                if out is not None:
                    outputs.append(out)
                current_iter += 1
        except (KeyboardInterrupt, StopIteration):
            # fine, just return
            pass
        finally:
            self.deinit()

        return outputs if outputs else None

    @property
    @abstractmethod
    def running(self) -> bool:
        """
        Whether the tube is currently running
        """
        pass

    def collect_input(
        self, node: Node, epoch: int, input: dict | None = None
    ) -> tuple[list[Any] | None, dict[PythonIdentifier, Any] | None]:
        """
        Gather input to give to the passed Node from the :attr:`.TubeRunner.store`

        Returns:
            dict: kwargs to pass to :meth:`.Node.process` if matching events are present
            dict: empty dict if Node is a :class:`.Source`
            None: if no input is available
        """
        if not node.spec or not node.spec.depends:
            return None, None
        if input is None:
            input = {}

        edges = self.tube.in_edges(node)

        inputs: dict[PythonIdentifier, Any] = {}

        state_inputs = self.tube.state.collect(edges)
        inputs |= state_inputs if state_inputs else inputs

        event_inputs = self.store.collect(edges, epoch)
        inputs |= event_inputs if event_inputs else inputs

        input_inputs = self.tube.input_collection.collect(edges, input)
        inputs |= input_inputs if input_inputs else inputs

        args, kwargs = self.store.split_args_kwargs(inputs)

        return args, kwargs

    @abstractmethod
    def collect_return(self, epoch: int | None = None) -> ReturnNodeType:
        """
        If any :class:`.Return` nodes are in the tube,
        gather their return values to return from :meth:`.TubeRunner.process`

        Returns:
            dict: of the Return sink's key mapped to the returned value,
            None: if there are no :class:`.Return` sinks in the tube
        """

    def add_callback(self, callback: Callable[[Event | MetaEvent], None]) -> None:
        self._callbacks.append(callback)

    def _call_callbacks(self, events: Sequence[Event | MetaEvent] | None) -> None:
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

    def inject_context(self, fn: Callable) -> Callable:
        """Wrap function in a partial with the runner context injected, if requested"""
        sig = inspect.signature(fn)
        ctx_key = [
            k for k, v in sig.parameters.items() if v.annotation and v.annotation is RunnerContext
        ]
        if ctx_key:
            return partial(fn, **{ctx_key[0]: self.get_context()})
        else:
            return fn

    def __enter__(self) -> Self:
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        self.deinit()
