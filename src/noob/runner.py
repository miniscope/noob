from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass, field
from logging import Logger
from threading import Event as ThreadEvent
from typing import Any, Self

from noob import init_logger
from noob.exceptions import AlreadyRunningError
from noob.node import Node
from noob.node.return_ import Return
from noob.store import EventStore
from noob.tube import Tube
from noob.types import ReturnNodeType


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

    _logger: Logger = field(default_factory=lambda: init_logger("tube.runner"))

    @abstractmethod
    def process(self) -> ReturnNodeType:
        """
        Process one step of data from each of the sources,
        passing intermediate data to any subscribed nodes in a chain.

        The `process` method normally does not return anything,
        except when using the special :class:`.Return` node
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

    def gather_input(self, node: Node) -> tuple[list[Any] | None, dict[str, Any] | None]:
        """
        Gather input to give to the passed Node from the :attr:`.TubeRunner.store`

        Returns:
            dict: kwargs to pass to :meth:`.Node.process` if matching events are present
            dict: empty dict if Node is a :class:`.Source`
            None: if no input is available
        """
        if not node.spec.depends:
            return None, None

        edges = self.tube.in_edges(node)
        return self.store.gather(edges)

    def gather_return(self) -> ReturnNodeType:
        """
        If any :class:`.Return` nodes are in the tube,
        gather their return values to return from :meth:`.TubeRunner.process`

        Returns:
            dict: of the Return sink's key mapped to the returned value,
            None: if there are no :class:`.Return` sinks in the tube
        """
        ret_nodes = [n for n in self.tube.nodes.values() if isinstance(n, Return)]
        if not ret_nodes:
            return None
        ret_node = ret_nodes[0]
        return ret_node.get(keep=False)


@dataclass
class SynchronousRunner(TubeRunner):
    """
    Simple, synchronous tube runner.

    Just run the nodes in topological order and return from return nodes.
    """

    def __post_init__(self):
        self._running = ThreadEvent()

    def __enter__(self) -> Self:
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        self.deinit()

    def init(self) -> Self:
        """
        Start processing data with the tube graph.
        """
        # TODO: lock for re-entry
        if self._running.is_set():
            raise AlreadyRunningError("Tube is already running!")

        self._running.set()
        for node in self.tube.nodes.values():
            node.init()
        return self

    def deinit(self) -> None:
        """Stop all nodes processing"""
        # TODO: lock to ensure we've been started
        for node in self.tube.nodes.values():
            node.deinit()
        self._running.clear()

    @property
    def running(self) -> bool:
        """Whether the tube is currently running"""
        return self._running.is_set()

    def process(self) -> ReturnNodeType:
        """
        Iterate through nodes in topological order,
        calling their process method and passing events as they are emitted.
        """
        self.store.clear()

        graph = self.tube.graph()
        graph.prepare()

        while graph.is_active():
            for node_id in graph.get_ready():
                node = self.tube.nodes[node_id]
                args, kwargs = self.gather_input(node)

                # need to eventually distinguish "still waiting" vs "there is none"
                args = [] if args is None else args
                kwargs = {} if kwargs is None else kwargs

                value = node.process(*args, **kwargs)
                self.store.add(value, node_id)
                graph.done(node_id)
                self._logger.debug(f"Node {node_id} emitted %s", value)

        return self.gather_return()

    def iter(self, n: int | None = None) -> Generator[ReturnNodeType, None, None]:
        """Treat the runner as an iterable"""
        self.init()
        current_iter = 0
        try:
            while n is None or current_iter < n:
                yield self.process()
                current_iter += 1
        finally:
            self.deinit()

    def run(self, n: int | None = None) -> None | list[ReturnNodeType]:
        outputs = []
        current_iter = 0
        self.init()
        try:
            while n is None or current_iter < n:
                out = self.process()
                if out is not None:
                    outputs.append(out)
                current_iter += 1
        except KeyboardInterrupt:
            # fine, just return
            pass
        finally:
            self.deinit()

        return outputs if outputs else None
