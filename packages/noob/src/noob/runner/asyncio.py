import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from noob.asset import AssetScope
from noob.event import MetaEvent
from noob.exceptions import InputMissingError
from noob.input import InputScope
from noob.node import Node, Return
from noob.runner.base import TubeRunner
from noob.scheduler import Scheduler
from noob.types import Epoch, ReturnNodeType


@dataclass
class AsyncRunner(TubeRunner):
    """
    Run nodes in an asyncio eventloop as soon as they are ready.

    .. important::

        The AsyncIO runner respects stateful nodes across top-level epochs,
        but currently runs subepochs induced by a `map` operation out of order!
        Either make your nodes stateless or use another runner until total ordering is implemented!

    """

    eventloop: asyncio.AbstractEventLoop = field(default_factory=asyncio.get_running_loop)
    exception_timeout: float = 10
    """
    When a node raises an error, wait this long (in seconds)
    before cancelling the other currently running nodes. 
    """
    max_pending_tasks: int = 128
    """
    Only add this many tasks to the eventloop at a time
    """

    def __post_init__(self):
        super().__post_init__()
        self._running = asyncio.Event()
        self._node_ready = asyncio.Event()
        self._init_lock = asyncio.Lock()
        self._scheduler_lock = asyncio.Lock()
        self._task_sem = asyncio.Semaphore(self.max_pending_tasks)
        self._pending_futures = set()
        self._exception: BaseException | None = None

    async def process(self, **kwargs: Any) -> ReturnNodeType:
        """
        Iterate through nodes in topological order,
        calling their process method and passing events as they are emitted.

        Process-scoped ``input`` s can be passed as kwargs.

        We don't want to cancel nodes on the first error,
        as would happen with a task group.
        e.g. some nodes might be writing data, and we want that to succeed
        even if some other node in this generation fails.
        We also want to schedule nodes opportunistically,
        as soon as their dependencies are satisfied,
        without needing to wait for the entire graph generation to finish
        so we queue as many as we can and use futures/callbacks to signal completion.
        If a node raises an error, we allow the other running nodes to complete
        (with some timeout)
        """
        input = self._validate_input(**kwargs)
        with self._asset_context(AssetScope.process):
            await self._before_process()

            while self.tube.scheduler.is_active():
                ready = await self._get_ready()
                ready = self._filter_ready(ready, self.tube.scheduler)
                for node_info in ready:
                    await self._task_sem.acquire()
                    self._process_node(node_info=node_info, input=input)

            self._after_process()
            result = self.collect_return()
        return result

    async def iter(self, n: int | None = None) -> AsyncGenerator[ReturnNodeType]:
        try:
            _ = self.tube.input_collection.validate_input(InputScope.process, {})
        except InputMissingError as e:
            raise InputMissingError(
                "Can't use the `iter` method with tubes with process-scoped input "
                "that was not provided when instantiating the tube! "
                "Use `process()` directly, providing required inputs to each call."
            ) from e

        await self.init()
        current_iter = 0
        has_return = any(isinstance(node, Return) for node in self.tube.nodes.values())
        try:
            while n is None or current_iter < n:
                ret = None
                loop = 0
                if not has_return:
                    ret = await self.process()
                else:
                    while ret is None:
                        ret = await self.process()
                        loop += 1
                        if loop > self.max_iter_loops:
                            raise RuntimeError("Reached maximum process calls per iteration")

                yield ret
                current_iter += 1
        finally:
            await self.deinit()

    async def init(self) -> None:
        """
        Start processing data with the tube graph.
        """
        async with self._init_lock:
            if self._running.is_set():
                # fine!
                return

            self._running.set()
            for node in self.tube.enabled_nodes.values():
                self.inject_context(node.init)()

            self.inject_context(self.tube.state.init)(AssetScope.runner)

    async def deinit(self) -> None:
        """Stop all nodes processing"""
        async with self._init_lock:
            for node in self.tube.enabled_nodes.values():
                self.inject_context(node.deinit)()

            self.inject_context(self.tube.state.deinit)(AssetScope.runner)

        self._running.clear()

    @property
    def running(self) -> bool:
        """Whether the tube is currently running"""
        return self._running.is_set()

    def _process_node(self, node_info: MetaEvent, input: dict) -> None:
        node_id, epoch = node_info["value"], node_info["epoch"]
        node = self._get_node(node_id)

        # FIXME: since nodes can run quasiconcurrently, need to ensure unique assets per node
        self.tube.state.init(AssetScope.node, node.edges)
        args, kwargs = self._collect_input(node, epoch, input)
        node, args, kwargs = self._before_call_node(node, *args, **kwargs)
        value = self._call_node(node, *args, **kwargs)
        node, value = self._after_call_node(node, value)
        self._handle_events(node, value, epoch)

    async def _before_process(self) -> None:  # type: ignore[override]
        if not self._running.is_set():
            await self.init()
        self.store.clear()
        async with self._scheduler_lock:
            self.tube.scheduler.add_epoch()

    async def _get_ready(self, epoch: Epoch | None = None) -> list[MetaEvent]:  # type: ignore[override]
        if self._exception:
            await self._raise_exception()
        async with self._scheduler_lock:
            ready = self.tube.scheduler.get_ready()
        if not ready:
            # if none are ready, wait until another node is complete and check again
            self._node_ready.clear()
            await self._node_ready.wait()
            async with self._scheduler_lock:
                return self.tube.scheduler.get_ready()
        else:
            return ready

    def _call_node(self, node: Node, *args: Any, **kwargs: Any) -> Any:
        future: asyncio.Task | asyncio.Future
        self._logger.debug("Running node %s with args: %s, kwargs: %s", node.id, args, kwargs)
        if node.is_coroutine:
            # mypy can't propagate type guard in cached is_coroutine property
            future = self.eventloop.create_task(node.process(*args, **kwargs))  # type: ignore[arg-type]
        else:
            part = partial(node.process, *args, **kwargs)
            future = self.eventloop.run_in_executor(None, part)
        self._pending_futures.add(future)
        return future

    def _handle_events(
        self, node: Node, value: asyncio.Future | asyncio.Task, epoch: Epoch
    ) -> None:
        value.add_done_callback(partial(self._node_complete, node=node, epoch=epoch))

    def _filter_ready(self, nodes: list[MetaEvent], scheduler: Scheduler) -> list[MetaEvent]:
        # graph autogenerates "assets" and "inputs" nodes if something depends on it
        # but in the sync runner we always have assets and inputs handy
        evts = []
        for node in nodes:
            if node["value"] in ("assets", "input"):
                scheduler.done(node["epoch"], node["value"])
            else:
                evts.append(node)
        return evts

    def _node_complete(self, future: asyncio.Future, node: Node, epoch: Epoch) -> None:
        self._pending_futures.remove(future)
        if future.exception():
            self._logger.debug("Node %s raised exception, re-raising outside of callback")
            self._task_sem.release()
            self._exception = future.exception()
            self.tube.state.deinit(AssetScope.node, node.edges)
            self._node_ready.set()
            return

        value = future.result()
        events = self.store.add_value(node.signals, value, node.id, epoch)
        if events is not None:
            all_events = self.tube.scheduler.update(events)
            if node.id in self.tube.state.dependencies:
                self.tube.state.update(events)
            self.tube.state.deinit(AssetScope.node, node.edges)
            self._call_callbacks(all_events)
        self._task_sem.release()
        self._node_ready.set()
        self._logger.debug("Node %s emitted %s in epoch %s", node.id, value, epoch)

    async def _raise_exception(self) -> None:
        if not self._exception:
            raise RuntimeError("Told to raise an exception, but no exception was found!")

        if not self._pending_futures:
            raise self._exception

        try:
            async for completed in asyncio.as_completed(
                self._pending_futures, timeout=self.exception_timeout
            ):
                # do nothing with the completed results here, we just want them to complete
                await completed
        except TimeoutError:
            self._logger.warning(
                "Nodes still running after timeout while waiting to raise exception: %s",
                self._pending_futures,
            )
        except Exception as e:
            # another node raised an exception, now we're really quitting
            raise self._exception from e
        else:
            raise self._exception

    def enable_node(self, node_id: str) -> None:
        self.tube.nodes[node_id].init()
        self.tube.enable_node(node_id)

    def disable_node(self, node_id: str) -> None:
        self.tube.nodes[node_id].deinit()
        self.tube.disable_node(node_id)

    def collect_return(self, epoch: Epoch | None = None) -> ReturnNodeType:
        """The return node holds values from a single epoch, get and transform them"""
        if epoch is not None:
            raise ValueError("Sync runner only stores a single epoch at a time")
        ret_nodes = [n for n in self.tube.enabled_nodes.values() if isinstance(n, Return)]
        if not ret_nodes:
            return None
        ret_node = ret_nodes[0]
        return ret_node.get(keep=False)

    async def __aenter__(self):
        await self.init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        await self.deinit()
