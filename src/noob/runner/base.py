from __future__ import annotations

import asyncio
import hashlib
import inspect
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine, Generator, Iterator, Sequence
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from logging import Logger
from typing import TYPE_CHECKING, Any, ParamSpec, Self, TypeVar, overload

from noob import Tube, init_logger
from noob.asset import AssetScope
from noob.event import Event, MetaEvent
from noob.exceptions import InputMissingError
from noob.input import InputScope
from noob.node import Edge, Node
from noob.store import EventStore
from noob.types import PythonIdentifier, ReturnNodeType, RunnerContext
from noob.utils import iscoroutinefunction_partial

if TYPE_CHECKING:
    from noob.scheduler import Scheduler
    from noob.types import NodeID

_TReturn = TypeVar("_TReturn")
_PProcess = ParamSpec("_PProcess")


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

    _logger: Logger = None  # type: ignore[assignment]
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

    def process(self, **kwargs: Any) -> ReturnNodeType:
        """
        Process one step of data from each of the sources,
        passing intermediate data to any subscribed nodes in a chain.

        The `process` method normally does not return anything,
        except when using the special :class:`.Return` node

        Process-scoped ``input`` s can be passed as kwargs.

        The Base process method is implemented as a series of lifecycle methods and hooks
        corresponding to different stages of a process call.
        Subclasses may override each of these methods to customize runner behavior.
        Subclasses may also override the :meth:`.TubeRunner.process` method itself,
        but must ensure that the phases of the base process method are executed.

        The methods invoked, in order (see docstrings for each for further explanation)

        * :meth:`._validate_input`
        * :meth:`._before_process`
        * :meth:`._filter_ready`
        * :meth:`._get_node`
        * :meth:`._collect_input`
        * :meth:`._before_call_node`
        * :meth:`._call_node`
        * :meth:`._after_call_node`
        * :meth:`._handle_events`
        * :meth:`._after_process`
        * :meth:`.collect_return`

        Process methods are also wrapped by :meth:`._asset_context` at two levels:

        * Process scope: as a contextmanager wrapping from ``_before_process`` to ``_after_process``
        * Node scope: around the ``_process_node`` method

        Runner-scoped assets are initialized and deinitialized in
        :meth:`.TubeRunner.init` and :meth:`.TubeRunner.deinit`
        """
        input = self._validate_input(**kwargs)
        with self._asset_context(AssetScope.process):
            self._before_process()

            while self.tube.scheduler.is_active():
                ready = self._get_ready()
                ready = self._filter_ready(ready, self.tube.scheduler)
                for node_info in ready:
                    self._process_node(node_info=node_info, input=input)

            self._after_process()
            result = self.collect_return()

        return result

    @abstractmethod
    def init(self) -> None | Coroutine:
        """
        Start processing data with the tube graph.

        Implementations of this method must

        * Initialize nodes
        * Initialize runner-scoped assets
        * raise a :class:`.TubeRunningError`
          if the tube has already been started and is running,
          (i.e. :meth:`.deinit` has not been called,
          or the tube has not exhausted itself)

        """

    @abstractmethod
    def deinit(self) -> None | Coroutine:
        """
        Stop processing data with the tube graph

        Implementations of this method must

        * Deinitialize nodes
        * Deinitialize runner-scoped assets

        """

    def iter(self, n: int | None = None) -> Generator[ReturnNodeType, None, None]:
        """
        Treat the runner as an iterable.

        Calls :meth:`.TubeRunner.process` until it yields a result
        (e.g. multiple times in the case of any ``gather`` s
        that change the cardinality of the graph.)
        """
        try:
            _ = self.tube.input_collection.validate_input(InputScope.process, {})
        except InputMissingError as e:
            raise InputMissingError(
                "Can't use the `iter` method with tubes with process-scoped input "
                "that was not provided when instantiating the tube! "
                "Use `process()` directly, providing required inputs to each call."
            ) from e

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

    @overload
    def run(self, n: int) -> list[ReturnNodeType]: ...

    @overload
    def run(self, n: None) -> None: ...

    def run(self, n: int | None = None) -> None | list[ReturnNodeType]:
        """
        Run the tube infinitely or for a fixed number of iterations in a row.

        Returns results if ``n`` is not ``None`` -
        If ``n`` is ``None`` , we assume we are going to be running for a very long time,
        and don't want to have an infinitely-growing collection in memory.
        """
        try:
            _ = self.tube.input_collection.validate_input(InputScope.process, {})
        except InputMissingError as e:
            raise InputMissingError(
                "Can't use the `run` method with tubes with process-scoped input "
                "that was not provided when instantiating the tube! "
                "Use `process()` directly, providing required inputs to each call."
            ) from e
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

    def _process_node(self, node_info: MetaEvent, input: dict) -> None:
        """
        Find the node, call the node, and handle the outputs of the node.

        Group the methods that apply to a single node
        so that they can be wrapped by a contextmanager
        without needing a million levels of nesting.
        """
        node_id, epoch = node_info["value"], node_info["epoch"]
        node = self._get_node(node_id)

        with self._asset_context(AssetScope.node, node.edges):
            args, kwargs = self._collect_input(node, epoch, input)
            node, args, kwargs = self._before_call_node(node, *args, **kwargs)
            value = self._call_node(node, *args, **kwargs)
            node, value = self._after_call_node(node, value)
            self._handle_events(node, value, epoch)

    def _before_process(self) -> None:
        """
        Hook for subclasses to do some work before the main body of the process method
        """
        return

    def _after_process(self) -> None:
        """
        Hook for subclasses to do some work after the main body of the process method,
        before collecting return values.
        """
        return

    def _get_ready(self, epoch: int | None = None) -> list[MetaEvent]:
        return self.tube.scheduler.get_ready(epoch=epoch)

    def _filter_ready(self, nodes: list[MetaEvent], scheduler: Scheduler) -> list[MetaEvent]:
        """
        Before running, filter or add nodes to run in a sorter generation,
        optionally mutating the scheduler.

        Default is a no-op, subclasses may override to customize behavior.

        Args:
            nodes (Sequence[MetaEvent]): A sequence of ``ReadyNode`` events whose ``value``
                is the node_id and ``epoch`` is the epoch they are ready in.
            scheduler (Scheduler): The Scheduler that yielded the set of ready nodes.
        """
        return nodes

    def _validate_input(self, **kwargs: Any) -> dict:
        """
        Validate input given to the process method, if any is specified
        """
        return self.tube.input_collection.validate_input(InputScope.process, kwargs)

    def _get_node(self, node_id: NodeID) -> Node:
        """
        Get a node.
        Usually from the tube, but separated to allow subclasses to customize behavior
        """
        return self.tube.nodes[node_id]

    def _collect_input(
        self, node: Node, epoch: int, input: dict | None = None
    ) -> tuple[tuple, dict[PythonIdentifier, Any]]:
        """
        Gather input to give to the passed Node from the :attr:`.TubeRunner.store`

        Returns:
            dict: kwargs to pass to :meth:`.Node.process` if matching events are present
            dict: empty dict if Node is a :class:`.Source`
            None: if no input is available
        """
        if not node.spec or not node.spec.depends:
            return tuple(), {}
        if input is None:
            input = {}

        edges = self.tube.in_edges(node)

        inputs: dict[PythonIdentifier, Any] = {}

        self.tube.state.init(AssetScope.node)
        state_inputs = self.tube.state.collect(edges, epoch)
        inputs |= state_inputs if state_inputs else inputs

        event_inputs = self.store.collect(edges, epoch)
        inputs |= event_inputs if event_inputs else inputs

        input_inputs = self.tube.input_collection.collect(edges, input)
        inputs |= input_inputs if input_inputs else inputs

        args, kwargs = self.store.split_args_kwargs(inputs)

        return args, kwargs

    def _before_call_node(self, node: Node, *args: Any, **kwargs: Any) -> tuple[Node, tuple, dict]:
        """
        Hook to modify behavior before calling the node.

        Default is no-op
        """
        return node, args, kwargs

    def _call_node(self, node: Node, *args: Any, **kwargs: Any) -> Any:
        """
        Call a node's process method with provided args and kwargs,
        returning its values.

        By default, try and call sync normally,
        and async with :func:`.call_async_from_sync` .

        Subclasses may override to customize behavior
        """
        if iscoroutinefunction_partial(node.process):
            return call_async_from_sync(node.process, *args, **kwargs)
        else:
            return node.process(*args, **kwargs)

    def _after_call_node(self, node: Node, value: Any) -> tuple[Node, Any]:
        """
        Hook to modify behavior after calling the node
        """
        return node, value

    def _handle_events(self, node: Node, value: Any, epoch: int) -> None:
        """
        After calling a node, handle its return value:

        This method must

        * Convert raw returned values to events
        * Store events to make them available to other nodes, as needed
        * Update the scheduler
        * Update the asset state
        * Call any callbacks with the resultant events

        The base implementation

        * calls :meth:`.EventStore.add_value` to create events
        * calls :meth:`.Scheduler.update` to update the scheduler
        * calls :meth:`._call_callbacks` to emit events to callbacks

        However other implementations may perform the responsibilities asynchronously
        e.g. via futures, see :class:`.AsyncIORunner` for an example.
        """
        events = self.store.add_value(node.signals, value, node.id, epoch)
        if node.id in self.tube.state.dependencies:
            self.tube.state.update(events)
        events_and_metaevents = self.tube.scheduler.update(events)
        self._call_callbacks(events_and_metaevents)
        self._logger.debug("Node %s emitted %s in epoch %s", node.id, value, epoch)

    @contextmanager
    def _asset_context(self, scope: AssetScope, edges: list[Edge] | None = None) -> Iterator[None]:
        """
        Init and deinit assets for a given scope.

        Wraps :meth:`.State.init_context` by default,
        subclasses that override must be sure to handle all asset scopes.
        """
        with self.tube.state.init_context(scope, edges) as context:
            yield context

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


def call_async_from_sync(
    fn: Callable[_PProcess, Coroutine[Any, Any, _TReturn]],
    executor: ThreadPoolExecutor | None = None,
    *args: _PProcess.args,
    **kwargs: _PProcess.kwargs,
) -> _TReturn:
    """
    Call an async function synchronously, either in this thread or a subthread.

    So here's the deal with this nonsense:

    * Calling async from sync is easy when there is no running eventloop in the thread
    * Calling async from sync is **almost comically hard** when there is a running eventloop.

    We are likely to encounter the second case where, e.g.,
    some async application calls some other code that uses a :class:`.SyncRunner`
    to run a tube that has an async node.

    :func:`asyncio.run` and :class:`asyncio.Runner` refuse to run when there is live eventloop,
    and attempting to use any of the :class:`~asyncio.Task` or :class:`~asyncio.Future`
    spawning methods from the running eventloop like :meth:`~asyncio.AbstractEventLoop.call_soon`
    or :func:`asyncio.run_coroutine_threadsafe`
    and then polling for the result with :meth:`asyncio.Future.result`
    causes a deadlock between the outer sync thread and the eventloop.
    The basic problem is that there is no way to wait in the thread (synchronously)
    that yields the thread to the eventloop (which is what async functions are for).

    We need to make a new thread in **some** way,
    Django's ``asgiref`` has a mindblowingly complicated
    `async_to_sync <https://github.com/django/asgiref/blob/2b28409ab83b3e4cf6fed9019403b71f8d7d1c51/asgiref/sync.py#L585>`_
    function that works **roughly** by creating a new thread
    and then calling :func:`asyncio.run` from *within that*
    (plus about a thousand other things to manage all the edge cases).
    That's more than a little bit cursed, because ideally,
    since the hard case here is where there is already an eventloop in the outer thread,
    we would be able to just *use that eventloop*.
    Normally one would just ``await`` the coro directly,
    which is what :class:`.AsyncRunner` does,
    but the :class:`.SyncRunner` can't do that because :meth:`.SyncRunner.process` is sync.

    However if one creates a new thread with a new eventloop,
    that will break any stateful nodes that e.g. have objects like :class:`asyncio.Event`
    that are bound to the first eventloop.

    Until we can figure out how to reuse the outer eventloop,
    we do the best we can with a modified version of ``asgiref`` 's approach.

    * Create a :class:`asyncio.Future` to store the eventual result we will return
      (the "result future")
    * Wrap the coroutine to call in another coroutine that calls :meth:`~asyncio.Future.set_result`
      or :meth:`~asyncio.Future.set_exception` on some passed future
      rather than returning the result directly
    * Use a :class:`~concurrent.futures.ThreadPoolExecutor` to run the wrapped coroutine
      in a **new** :class:`asyncio.AbstractEventLoop` in a separate thread,
      returning a second :class:`~concurrent.futures.Future` (the "completion future")
    * Add a callback to the completion future that notifies a :class:`~threading.Condition`
    * Wait in the main thread for the :class:`~threading.Condition` to be notified
    * Return the result from the result future.

    The reason we don't just directly return the value of the process coroutine
    in the inner wrapper coroutine
    and then return the result of the completion future is error handling -
    Errors raised in the wrapping coroutine have a large amount of noise in the traceback,
    so instead we use :meth:`~asyncio.Future.set_exception` to propagate the raised error
    up to the main thread and raise it there.

    Args:
        fn: The callable that returns a coroutine to run
        executor (concurrent.futures.ThreadPoolExecutor | None): Provide an already-created
            thread pool executor. If ``None`` , creates one and shuts it down before returning
        *args: Passed to ``fn``
        **kwargs: Passed to ``fn``

    Returns: The result of the called function

    References:
        * https://github.com/django/asgiref/blob/2b28409ab83b3e4cf6fed9019403b71f8d7d1c51/asgiref/sync.py#L152
        * https://stackoverflow.com/questions/79663750/call-async-code-inside-sync-code-inside-async-code
        * https://github.com/python/cpython/issues/66435
        * https://github.com/python/cpython/issues/93462
        * https://discuss.python.org/t/support-for-running-async-functions-in-sync-functions/16220/3
        * https://github.com/fsspec/filesystem_spec/blob/2576617e5cbe441bcc53b021bccd85ff3489fde7/fsspec/asyn.py#L63
    """
    if not iscoroutinefunction_partial(fn):
        raise RuntimeError(
            "Called a synchronous function from call_async_from_sync, "
            "something has gone wrong in however this runner is implemented"
        )

    coro = fn(*args, **kwargs)

    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # async coroutine called in a sync runner context
            # while the runner is inside another asyncio eventloop
            # continue to this evil motherfucker of a fallback
            pass
        else:
            raise e

    created = False
    if executor is None:
        created = True
        executor = ThreadPoolExecutor(1)

    result_future: asyncio.Future[_TReturn] = asyncio.Future()
    work_ready = threading.Condition()
    finished = False

    # Closures because this code should never escape the containment tomb of this crime against god
    async def _wrap(call_result: asyncio.Future[_TReturn], fn: Coroutine) -> None:
        nonlocal finished
        try:
            result = await fn
            call_result.set_result(result)
        except Exception as e:
            call_result.set_exception(e)
        finally:
            finished = True

    def _done(_: ConcurrentFuture) -> None:
        nonlocal finished

        finished = True
        with work_ready:
            work_ready.notify_all()

    future_inner = executor.submit(asyncio.run, _wrap(result_future, coro))
    future_inner.add_done_callback(_done)

    try:
        while not finished and not future_inner.done():
            with work_ready:
                work_ready.wait(timeout=1)
        res = result_future.result()
        return res
    finally:
        if created:
            executor.shutdown(wait=False)
