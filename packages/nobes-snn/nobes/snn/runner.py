import asyncio
import contextlib
import time
from collections import defaultdict, deque
from collections.abc import AsyncGenerator, Generator
from typing import Any, ClassVar

from noob.event import MetaSignal
from noob.network.message import EventMsg
from noob.runner.base import TubeRunner
from noob.runner.zmq import NodeRunner, ZMQRunner
from noob.types import Epoch, ReturnNodeType


class NodeFreeRunner(NodeRunner):
    def __init__(self, interval: float = 0.01, *args: Any, **kwargs: Any):
        """
        Args:
            interval (float): minimum time (in seconds) that each iteration should take.
                (i.e., if the iteration takes less time than this, sleep for the difference)
        """
        super().__init__(*args, **kwargs)
        self._interval = interval
        self._signals = defaultdict(deque)
        self._event_lock = asyncio.Lock()

    async def await_inputs(self) -> AsyncGenerator[tuple[tuple[Any], dict[str, Any], Epoch]]:
        """epochs are meaningless to free runner"""
        current_epoch = 0

        while not self._quitting.is_set():
            await self._freerun.wait()
            epoch = Epoch(current_epoch)
            input_events = []
            async with self._event_lock:
                for dep in self.depends:
                    with contextlib.suppress(IndexError):
                        input_events.append(self._signals[dep].popleft())
            inputs = self.store.transform_events(self._node.edges, input_events)
            args, kwargs = self.store.split_args_kwargs(inputs)
            yielded_time = time.time()

            yield args, kwargs, epoch

            runtime = time.time() - yielded_time
            await asyncio.sleep(max(0.0, self._interval - runtime))
            current_epoch += 1

    async def on_event(self, msg: EventMsg) -> None:
        """Just stack the events up in queues, we don't care about epochs here"""
        depended_events = [
            e
            for e in msg.value
            if (e["node_id"], e["signal"]) in self.depends and e["value"] is not MetaSignal.NoEvent
        ]

        async with self._event_lock:
            for e in depended_events:
                self._signals[(e["node_id"], e["signal"])].append(e)


class ZMQFreeRunner(ZMQRunner):
    noderunner_cls: ClassVar[type[TubeRunner]] = NodeFreeRunner

    def run(self, n: int | None = None) -> None | list[ReturnNodeType]:
        if n is not None:
            raise NotImplementedError(
                "What do you mean freerun for only a little bit you are either in or you're out"
            )
        super().run()

    def iter(self, n: int | None = None) -> Generator[ReturnNodeType, None, None]:
        raise NotImplementedError(
            "iteration mode doesn't really make sense for the free runner! use run!"
        )

    def process(self, **kwargs: Any) -> ReturnNodeType:
        raise NotImplementedError(
            "process mode doesn't really make sense for the free runner! use run!"
        )

    async def on_event(self, msg: EventMsg) -> None:
        pass
