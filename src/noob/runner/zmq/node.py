import asyncio
import concurrent.futures
import multiprocessing as mp
import os
import signal
import traceback
from collections import deque
from collections.abc import AsyncGenerator
from functools import partial
from itertools import count
from types import FrameType
from typing import Any, cast

import zmq
from pydantic import ValidationError

from noob import init_logger
from noob.config import config
from noob.event import Event, MetaEvent
from noob.input import InputCollection
from noob.network.loop import EventloopMixin
from noob.network.message import (
    AnnounceMsg,
    DeinitMsg,
    ErrorMsg,
    ErrorValue,
    EventMsg,
    IdentifyMsg,
    IdentifyValue,
    Message,
    MessageType,
    NodeStatus,
    ProcessMsg,
    StartMsg,
    StatusMsg,
    StopMsg,
)
from noob.node import Node, NodeSpecification, Signal
from noob.scheduler import Scheduler
from noob.store import EventStore
from noob.types import Epoch
from noob.utils import iscoroutinefunction_partial


class NodeRunner(EventloopMixin):
    """
    Runner for a single node

    - DEALER to communicate with command inbox
    - PUB (outbox) to publish events
    - SUB (inbox) to subscribe to events from other nodes.
    """

    def __init__(
        self,
        spec: NodeSpecification,
        runner_id: str,
        command_outbox: str,
        command_router: str,
        input_collection: InputCollection,
        protocol: str = "ipc",
    ):
        self.spec = spec
        self.runner_id = runner_id
        self.input_collection = input_collection
        self.command_outbox = command_outbox
        self.command_router = command_router
        self.protocol = protocol
        self.store = EventStore()
        self.scheduler: Scheduler = None  # type: ignore[assignment]
        self.logger = init_logger(f"runner.node.{runner_id}.{self.spec.id}")

        self._node: Node | None = None
        self._depends: tuple[tuple[str, str], ...] | None = None
        self._has_input: bool | None = None
        self._nodes: dict[str, IdentifyValue] = {}
        self._counter = count()
        self._epochs_todo: deque[Epoch] = deque()
        self._freerun = asyncio.Event()
        self._process_one = asyncio.Event()
        self._status: NodeStatus = NodeStatus.stopped
        self._status_lock = asyncio.Lock()
        self._ready_condition = asyncio.Condition()
        super().__init__()
        self._quitting = asyncio.Event()

    @property
    def outbox_address(self) -> str:
        if self.protocol == "ipc":
            path = config.tmp_dir / f"{self.runner_id}/nodes/{self.spec.id}/outbox"
            path.parent.mkdir(parents=True, exist_ok=True)
            return f"{self.protocol}://{str(path)}"
        else:
            raise NotImplementedError()

    @property
    def depends(self) -> tuple[tuple[str, str], ...] | None:
        """(node, signal) tuples of the wrapped node's dependencies"""
        if self._node is None:
            return None
        elif self._depends is None:
            self._depends = tuple(
                (edge.source_node, edge.source_signal) for edge in self._node.edges
            )
            self.logger.debug("Depends on: %s", self._depends)
        return self._depends

    @property
    def has_input(self) -> bool:
        if self._has_input is None:
            self._has_input = (
                False if not self.depends else any(d[0] == "input" for d in self.depends)
            )
        return self._has_input

    @property
    def status(self) -> NodeStatus:
        return self._status

    @status.setter
    def status(self, status: NodeStatus) -> None:
        self._status = status

    @classmethod
    def run(cls, spec: NodeSpecification, **kwargs: Any) -> None:
        """
        Target for multiprocessing.run,
        init the class and start it!
        """

        # ensure that events and conditions are bound to the eventloop created in the process
        async def _run_inner() -> None:
            nonlocal spec, kwargs
            runner = NodeRunner(spec=spec, **kwargs)
            await runner._run()

        asyncio.run(_run_inner())

    async def _run(self) -> None:
        try:

            def _handler(sig: int, frame: FrameType | None = None) -> None:
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                raise KeyboardInterrupt()

            signal.signal(signal.SIGTERM, _handler)
            await self.init()
            self._node = cast(Node, self._node)
            self._freerun.clear()
            self._process_one.clear()
            await asyncio.gather(self._poll_receivers(), self._process_loop())
        except KeyboardInterrupt:
            self.logger.debug("Got keyboard interrupt, quitting")
        except Exception as e:
            await self.error(e)
        finally:
            await self.deinit()

    async def _process_loop(self) -> None:
        self._node = cast(Node, self._node)
        is_async = iscoroutinefunction_partial(self._node.process)
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            async for args, kwargs, epoch in self.await_inputs():
                self.logger.debug(
                    "Running with args: %s, kwargs: %s, epoch: %s", args, kwargs, epoch
                )
                if is_async:
                    # mypy fails here because it can't propagate the type guard above
                    value = await self._node.process(*args, **kwargs)  # type: ignore[misc]
                else:
                    part = partial(self._node.process, *args, **kwargs)
                    value = await loop.run_in_executor(executor, part)
                events = self.store.add_value(self._node.signals, value, self._node.id, epoch)
                async with self._ready_condition:
                    self.scheduler.add_epoch()

                    # nodes don't report epoch endings since they don't know about the full tube
                    events = [e for e in events if e["node_id"] != "meta"]
                    if events:
                        self.scheduler.update(events)
                        await self.publish_events(events)
                    self._ready_condition.notify_all()

    async def await_inputs(self) -> AsyncGenerator[tuple[tuple[Any], dict[str, Any], Epoch]]:
        """
        Iterate inputs as they are ready

        Handle multiple types of running
        - `process` based: run a single epoch, or set of epochs at a time
        - free`run` based: run until told to stop

        And multiple types of nodes
        - stateful: must call epochs and subepochs in order
        - stateless: call whichever epoch whenever the dependencies are met
        """
        self._node = cast(Node, self._node)
        epoch = None
        # FIXME: This handling of statefulness is shamefully awful and if you see this
        # YELL AT JONNY TO FIX IT.
        # This logic should all be consolidated into the scheduler, see issue #144
        expected_epoch = Epoch(0) if self._node.stateful else None
        while not self._quitting.is_set():
            if (
                not self._freerun.is_set()
                and epoch is None
                and (
                    len(self._epochs_todo) == 0
                    or (
                        len(self._epochs_todo) > 0
                        and self._node.stateful
                        and self._epochs_todo[0] != expected_epoch
                    )
                )
            ):
                self._process_one.clear()
                await self._process_one.wait()
                if (
                    len(self._epochs_todo) > 0
                    and self._node.stateful
                    and self._epochs_todo[0] != expected_epoch
                ):
                    continue

            # if we don't have a current epoch that we're working on...
            if epoch is None:
                if len(self._epochs_todo) > 0:
                    # given to us explicitly by a `process` call
                    epoch = self._epochs_todo.popleft()
                else:
                    # infer while freerunning
                    epoch = Epoch(next(self._counter)) if self._node.stateful else None
                epoch = cast(Epoch, epoch)
                if self._node.stateful:
                    expected_epoch = Epoch(epoch[0].epoch + 1)
            readies = await self.await_node(epoch=epoch)

            if epoch is None:
                # stateless nodes - run the given epoch to completion
                epoch = list(dict.fromkeys([r["epoch"].root for r in readies]))[0]
                self._counter = count(max(next(self._counter), epoch.root[0].epoch + 1))

            for ready in readies:
                edges = self._node.edges

                inputs = self.store.collect(
                    edges, ready["epoch"], eventmap=self._node.injections.get("events")
                )
                if inputs is None:
                    inputs = {}
                if self._node.injections.get("epoch"):
                    inputs[self._node.injections["epoch"]] = ready["epoch"]

                args, kwargs = self.store.split_args_kwargs(inputs)
                yield args, kwargs, ready["epoch"]

            if self.scheduler.epoch_completed(epoch):
                self.logger.debug("Epoch completed: %s", epoch)
                self.store.clear(epoch)
                epoch = None

    async def publish_events(self, events: list[Event]) -> None:
        msg = EventMsg(node_id=self.spec.id, value=events)
        await self.sockets["outbox"].send_multipart([b"event", msg.to_bytes()])

    async def init(self) -> None:
        self.logger.debug("Initializing")
        await self.init_node()
        self._init_sockets()
        self._quitting.clear()
        self.status = (
            NodeStatus.waiting
            if self.depends and [d for d in self.depends if d[0] != "input"]
            else NodeStatus.ready
        )
        await self.identify()
        self.logger.debug("Initialization finished")

    async def deinit(self) -> None:
        """
        Deinitialize the node class after receiving on_deinit message and
        draining out the end of the _process_loop.
        """
        self.logger.debug("Deinitializing")
        if self._node is not None:
            self._node.deinit()

        # should have already been called in on_deinit, but just to make sure we're killed dead...
        self._quitting.set()

        self.logger.debug("Deinitialization finished")

    async def identify(self) -> None:
        """
        Send the command node an announce to say we're alive
        """
        if self._node is None:
            raise RuntimeError(
                "Node was not initialized by the time we tried to "
                "identify ourselves to the command node."
            )

        self.logger.debug("Identifying")
        async with self._status_lock:
            ann = IdentifyMsg(
                node_id=self.spec.id,
                value=IdentifyValue(
                    node_id=self.spec.id,
                    status=self.status,
                    outbox=self.outbox_address,
                    signals=[s.name for s in self._node.signals] if self._node.signals else None,
                    slots=(
                        [slot_name for slot_name in self._node.slots] if self._node.slots else None
                    ),
                ),
            )
            await self.sockets["dealer"].send_multipart([ann.to_bytes()])
        self.logger.debug("Sent identification message: %s", ann)

    async def update_status(self, status: NodeStatus) -> None:
        """Update our internal status and announce it to the command node"""
        self.logger.debug("Updating status as %s", status)
        async with self._status_lock:
            self.status = status
            msg = StatusMsg(node_id=self.spec.id, value=status)
            await self.sockets["dealer"].send_multipart([msg.to_bytes()])
            self.logger.debug("Updated status")

    async def init_node(self) -> None:
        self._node = Node.from_specification(self.spec, self.input_collection)
        self._node.init()
        self.scheduler = Scheduler(
            nodes={self.spec.id: self.spec},
            edges=self._node.edges,
            logger=init_logger(f"noob.scheduler.{self.spec.id}"),
        )
        async with self._ready_condition:
            self.scheduler.add_epoch()
            self._ready_condition.notify_all()

    def _init_sockets(self) -> None:
        self._init_loop()
        self._init_dealer()
        self._init_outbox()
        self._init_inbox()

    def _init_dealer(self) -> None:
        dealer = self.context.socket(zmq.DEALER)
        dealer.setsockopt_string(zmq.IDENTITY, self.spec.id)
        dealer.connect(self.command_router)
        self.register_socket("dealer", dealer)
        self.logger.debug("Connected to command node at %s", self.command_router)

    def _init_outbox(self) -> None:
        pub = self.context.socket(zmq.PUB)
        pub.setsockopt_string(zmq.IDENTITY, self.spec.id)
        if self.protocol == "ipc":
            pub.bind(self.outbox_address)
        else:
            raise NotImplementedError()
            # something like:
            # port = pub.bind_to_random_port(self.protocol)
        self.register_socket("outbox", pub)

    def _init_inbox(self) -> None:
        """
        Init the subscriber, but don't attempt to subscribe to anything but the command yet!
        we do that when we get node Announces
        """
        sub = self.context.socket(zmq.SUB)
        sub.setsockopt_string(zmq.IDENTITY, self.spec.id)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        sub.connect(self.command_outbox)
        self.register_socket("inbox", sub, receiver=True)
        self.add_callback("inbox", self.on_inbox)
        self.logger.debug("Subscribed to command outbox %s", self.command_outbox)

    async def on_inbox(self, message: Message) -> None:
        # FIXME: all this switching sux,
        # just have a decorator to register a handler for a given message type
        if message.type_ == MessageType.announce:
            message = cast(AnnounceMsg, message)
            await self.on_announce(message)
        elif message.type_ == MessageType.event:
            message = cast(EventMsg, message)
            await self.on_event(message)
        elif message.type_ == MessageType.process:
            message = cast(ProcessMsg, message)
            await self.on_process(message)
        elif message.type_ == MessageType.start:
            message = cast(StartMsg, message)
            await self.on_start(message)
        elif message.type_ == MessageType.stop:
            message = cast(StopMsg, message)
            await self.on_stop(message)
        elif message.type_ == MessageType.deinit:
            message = cast(DeinitMsg, message)
            await self.on_deinit(message)
        elif message.type_ == MessageType.ping:
            await self.identify()
        else:
            # log but don't throw - other nodes shouldn't be able to crash us
            self.logger.error(f"{message.type_} not implemented!")
            self.logger.debug("%s", message)

    async def on_announce(self, msg: AnnounceMsg) -> None:
        """
        Store map, connect to the nodes we depend on
        """
        self._node = cast(Node, self._node)
        self.logger.debug("Processing announce")

        depended_nodes = {edge.source_node for edge in self._node.edges}
        if depended_nodes:
            self.logger.debug("Should subscribe to %s", depended_nodes)
        for node_id in msg.value["nodes"]:
            if node_id in depended_nodes and node_id not in self._nodes:
                # TODO: a way to check if we're already connected, without storing it locally?
                outbox = msg.value["nodes"][node_id]["outbox"]
                self.logger.debug("Subscribing to %s at %s", node_id, outbox)
                self.sockets["inbox"].connect(outbox)
                self.logger.debug("Subscribed to %s at %s", node_id, outbox)
        self._nodes = msg.value["nodes"]
        if set(self._nodes) >= depended_nodes - {"input"} and self.status == NodeStatus.waiting:
            await self.update_status(NodeStatus.ready)
        # status and announce messages can be received out of order,
        # so if we observe the command node being out of sync, we update it.
        elif (
            self._node.id in msg.value["nodes"]
            and msg.value["nodes"][self._node.id]["status"] != self.status.value
        ):
            await self.update_status(self.status)

    async def on_event(self, msg: EventMsg) -> None:
        self.logger.debug("RECEIVED EVENTS: %s", msg.value)
        events = msg.value
        if not self.depends:
            self.logger.debug("No dependencies, not storing events")
            return

        to_add = [e for e in events if (e["node_id"], e["signal"]) in self.depends]
        for event in to_add:
            self.store.add(event)
        async with self._ready_condition:
            self.scheduler.update(events)
            self._ready_condition.notify_all()

    async def on_start(self, msg: StartMsg) -> None:
        """
        Start running in free mode
        """
        await self.update_status(NodeStatus.running)
        if msg.value is None:
            self._freerun.set()
        else:
            self._epochs_todo.extend(Epoch(next(self._counter)) for _ in range(msg.value))
        self._process_one.set()

    async def on_process(self, msg: ProcessMsg) -> None:
        """
        Process a single graph iteration
        """
        self._node = cast(Node, self._node)
        self.logger.debug("Received Process message: %s", msg)
        self._epochs_todo.append(msg.value["epoch"])
        if self._node.stateful:
            self._epochs_todo = deque(sorted(self._epochs_todo))
        self._counter = count(max(next(self._counter), msg.value["epoch"].root[0].epoch + 1))
        if self.has_input and self.depends:  # for mypy - depends is always true if has_input is
            # combine with any tube-scoped input and store as events
            # when calling the node, we get inputs from the eventstore rather than input collection
            process_input = msg.value["input"] if msg.value["input"] else {}
            # filter to only the input that we depend on
            combined = {
                dep[1]: self.input_collection.get(dep[1], process_input)
                for dep in self.depends
                if dep[0] == "input"
            }
            if len(combined) == 1:
                value = combined[next(iter(combined.keys()))]
            else:
                value = list(combined.values())
            async with self._ready_condition:
                events = self.store.add_value(
                    [Signal(name=k, type_=None) for k in combined],
                    value,
                    node_id="input",
                    epoch=msg.value["epoch"],
                )
                scheduler_events = self.scheduler.update(events)
                self._ready_condition.notify_all()
                self.logger.debug("Updated scheduler with process events: %s", scheduler_events)

        self._process_one.set()

    async def on_stop(self, msg: StopMsg) -> None:
        """Stop processing (but stay responsive)"""
        self._process_one.clear()
        self._freerun.clear()
        await self.update_status(NodeStatus.stopped)
        self.logger.debug("Stopped")

    async def on_deinit(self, msg: DeinitMsg) -> None:
        """
        Deinitialize the node, close networking thread.

        Cause the main loop to end, which calls deinit
        """
        await self.update_status(NodeStatus.closed)
        self._quitting.set()

        pid = mp.current_process().pid
        if pid is None:
            return
        self.logger.debug("Emitting sigterm to self %s", msg)
        os.kill(pid, signal.SIGTERM)
        raise asyncio.CancelledError()

    async def error(self, err: Exception) -> None:
        """
        Capture the error and traceback context from an exception using
        :class:`traceback.TracebackException` and send to command node to re-raise
        """
        tbexception = "\n".join(traceback.format_tb(err.__traceback__))
        self.logger.debug("Throwing error in main runner: %s", tbexception)
        args = (err.title, err.errors()) if isinstance(err, ValidationError) else err.args  # type: ignore[attr-defined,unused-ignore]

        msg = ErrorMsg(
            node_id=self.spec.id,
            value=ErrorValue(
                err_type=type(err),
                err_args=args,
                traceback=tbexception,
            ),
        )
        await self.sockets["dealer"].send_multipart([msg.to_bytes()])

    async def await_node(self, epoch: Epoch | None = None) -> list[MetaEvent]:
        """
        Block until a node is ready

        Args:
            epoch (Epoch, None): if `int` , wait until the node is ready in the given epoch,
                otherwise wait until the node is ready in any epoch

        """
        async with self._ready_condition:
            await self._ready_condition.wait_for(
                lambda: self.scheduler.node_is_ready(self.spec.id, epoch)
            )

            # be FIFO-like and get the earliest epoch the node is ready in
            if epoch is None:
                for ep, graph in self.scheduler._epochs.items():
                    if self.spec.id in graph.ready_nodes:
                        epoch = ep
                        break

            if epoch is None:
                raise RuntimeError(
                    "Could not find ready epoch even though node ready condition passed, "
                    "something is wrong with the way node status checking is "
                    "locked between threads."
                )

            # mark just one event as "out."
            # threadsafe because we are holding the lock that protects graph mutation
            ready = self.scheduler.get_ready(epoch, node_id=self.spec.id)
            ready = [r for r in ready if r["value"] == self.spec.id]

        return ready
