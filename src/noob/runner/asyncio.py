import asyncio
import sys
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

from noob.input import InputScope
from noob.node import Node, Return
from noob.runner.base import TubeRunner
from noob.types import ReturnNodeType
from noob.utils import iscoroutinefunction_partial

if sys.version_info < (3, 14):
    pass
else:
    pass


@dataclass
class AsyncRunner(TubeRunner):

    eventloop: asyncio.AbstractEventLoop | None = None

    def __post_init__(self):
        super().__post_init__()
        self._running = asyncio.Event()
        self._node_ready = asyncio.Event()
        self._init_lock = asyncio.Lock()
        self._pending_futures = set()
        if not self.eventloop or self.eventloop.is_closed():
            self.eventloop = asyncio.get_running_loop()

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

            for asset in self.tube.state.assets.values():
                self.inject_context(asset.init)()

    async def deinit(self) -> None:
        """Stop all nodes processing"""
        async with self._init_lock:
            for node in self.tube.enabled_nodes.values():
                node.deinit()

            for asset in self.tube.state.assets.values():
                asset.deinit()

        self._running.clear()

    @property
    def running(self) -> bool:
        """Whether the tube is currently running"""
        return self._running.is_set()

    async def process(self, **kwargs: Any) -> ReturnNodeType:
        """
        Iterate through nodes in topological order,
        calling their process method and passing events as they are emitted.

        Process-scoped ``input`` s can be passed as kwargs.
        """
        self.eventloop = cast(asyncio.AbstractEventLoop, self.eventloop)
        if not self._running.is_set():
            await self.init()

        input = self.tube.input_collection.validate_input(InputScope.process, kwargs)
        self.store.clear()

        scheduler = self.tube.scheduler
        scheduler.add_epoch()

        while scheduler.is_active():
            ready = scheduler.get_ready()
            if not ready:
                # if none are ready, wait until another node is complete and check again
                self._node_ready.clear()
                await self._node_ready.wait()
                continue

            # we don't want to cancel nodes on the first error,
            # as would happen with a task group.
            # e.g. some nodes might be writing data, and we want that to succeed
            # even if some other node in this generation fails.
            # We also want to schedule nodes opportunistically,
            # as soon as their dependencies are satisfied,
            # without needing to wait for the entire graph generation to finish
            # so we queue as many as we can and use futures/callbacks to signal completion

            for node_info in ready:
                node_id, epoch = node_info["value"], node_info["epoch"]

                if node_id in ("assets", "input"):
                    # graph autogenerates "assets" node if something depends on it
                    scheduler.done(epoch, node_id)
                    continue
                node = self.tube.nodes[node_id]
                if not node.enabled:
                    # nodes can be in the graph while disabled if something else depends on them
                    # FIXME when we stop using the builtin graphlib
                    continue

                maybe_args, maybe_kwargs = self.collect_input(node, epoch, input)

                # need to eventually distinguish "still waiting" vs "there is none"
                args = [] if maybe_args is None else maybe_args
                kwargs = {} if maybe_kwargs is None else maybe_kwargs
                future: asyncio.Future | asyncio.Task
                if iscoroutinefunction_partial(node.process):
                    future = self.eventloop.create_task(node.process(*args, **kwargs))
                else:
                    part = partial(node.process, *args, **kwargs)
                    future = self.eventloop.run_in_executor(None, part)
                future.add_done_callback(partial(self._node_complete, node=node, epoch=epoch))
                self._pending_futures.add(future)
                self._logger.debug("Scheduled node %s in epoch %s in eventloop", node_id, epoch)

        return self.collect_return()

    def _node_complete(self, future: asyncio.Future, node: Node, epoch: int) -> None:
        value = future.result()
        events = self.store.add_value(node.signals, value, node.id, epoch)
        if events is not None:
            all_events = self.tube.scheduler.update(events)
            self._call_callbacks(all_events)
        self._node_ready.set()
        self._logger.debug("Node %s emitted %s in epoch %s", node.id, value, epoch)

    def enable_node(self, node_id: str) -> None:
        self.tube.nodes[node_id].init()
        self.tube.enable_node(node_id)

    def disable_node(self, node_id: str) -> None:
        self.tube.nodes[node_id].deinit()
        self.tube.disable_node(node_id)

    def collect_return(self, epoch: int | None = None) -> ReturnNodeType:
        """The return node holds values from a single epoch, get and transform them"""
        if epoch is not None:
            raise ValueError("Sync runner only stores a single epoch at a time")
        ret_nodes = [n for n in self.tube.enabled_nodes.values() if isinstance(n, Return)]
        if not ret_nodes:
            return None
        ret_node = ret_nodes[0]
        return ret_node.get(keep=False)
