"""
- Central command pub/sub
- each sub-runner has its own set of sockets for publishing and consuming events
- use the `node_id.signal` etc. as basically a feed address

.. todo::

    Currently only IPC is supported, and thus the zmq runner can't run across machines.
    Supporting TCP is WIP, it will require some degree of authentication
    among nodes to prevent arbitrary code execution,
    since we shouldn't count on users to properly firewall their runners.


.. todo::

    The socket spawning and event handling is awfully manual here.
    Leaving it as is because it's somewhat unlikely we'll need to generalize it,
    but otherwise it would be great to standardize socket names and have
    event handler decorators like:

        @on_router(MessageType.sometype)

"""

import asyncio
import concurrent.futures
import math
import multiprocessing as mp
import os
import signal
import threading
import traceback
from collections.abc import AsyncGenerator, Generator, MutableSequence
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from itertools import count
from multiprocessing.synchronize import Event as EventType
from time import time
from types import FrameType
from typing import TYPE_CHECKING, Any, cast, overload
from uuid import uuid4

try:
    import zmq
except ImportError as e:
    raise ImportError(
        "Attempted to import zmq runner, but zmq deps are not installed. install with `noob[zmq]`",
    ) from e


from noob.config import config
from noob.event import Event, MetaEvent, MetaEventType, MetaSignal
from noob.exceptions import InputMissingError
from noob.input import InputCollection, InputScope
from noob.logging import init_logger
from noob.network.loop import EventloopMixin
from noob.network.message import (
    AnnounceMsg,
    AnnounceValue,
    DeinitMsg,
    ErrorMsg,
    ErrorValue,
    EventMsg,
    IdentifyMsg,
    IdentifyValue,
    Message,
    MessageType,
    NodeStatus,
    PingMsg,
    ProcessMsg,
    StartMsg,
    StatusMsg,
    StopMsg,
)
from noob.node import Node, NodeSpecification, Return, Signal
from noob.runner.base import TubeRunner
from noob.scheduler import Scheduler
from noob.store import EventStore
from noob.types import NodeID, ReturnNodeType
from noob.utils import iscoroutinefunction_partial

if TYPE_CHECKING:
    pass


class CommandNode(EventloopMixin):
    """
    Pub node that controls the state of the other nodes/announces addresses

    - one PUB socket to distribute commands
    - one ROUTER socket to receive return messages from runner nodes
    - one SUB socket to subscribe to all events

    The wrapping runner should register callbacks with `add_callback` to handle incoming messages.

    """

    def __init__(self, runner_id: str, protocol: str = "ipc", port: int | None = None):
        """

        Args:
            runner_id (str): The unique ID for the runner/tube session.
                All nodes within a runner use this to limit communication within a tube
            protocol:
            port:
        """

        self.runner_id = runner_id
        self.port = port
        self.protocol = protocol
        self.logger = init_logger(f"runner.node.{runner_id}.command")
        self._nodes: dict[str, IdentifyValue] = {}
        self._ready_condition: threading.Condition = None  # type: ignore[assignment]
        self._init = threading.Event()
        super().__init__()

    @property
    def pub_address(self) -> str:
        """Address the publisher bound to"""
        if self.protocol == "ipc":
            path = config.tmp_dir / f"{self.runner_id}/command/outbox"
            path.parent.mkdir(parents=True, exist_ok=True)
            return f"{self.protocol}://{str(path)}"
        else:
            raise NotImplementedError()

    @property
    def router_address(self) -> str:
        """Address the return router is bound to"""
        if self.protocol == "ipc":
            path = config.tmp_dir / f"{self.runner_id}/command/inbox"
            path.parent.mkdir(parents=True, exist_ok=True)
            return f"{self.protocol}://{str(path)}"
        else:
            raise NotImplementedError()

    def run(self) -> None:
        """
        Target for :class:`threading.Thread`
        """
        asyncio.run(self._run())

    async def _run(self) -> None:
        self.init()
        await self._poll_receivers()

    def init(self) -> None:
        self.logger.debug("Starting command runner")
        self._init.clear()
        self._init_loop()
        self._ready_condition = threading.Condition()
        self._init_sockets()
        self._init.set()
        self.logger.debug("Command runner started")

    def deinit(self) -> None:
        """Close the eventloop, stop processing messages, reset state"""
        self.logger.debug("Deinitializing")

        async def _deinit() -> None:
            msg = DeinitMsg(node_id="command")
            await self.sockets["outbox"].send_multipart([b"deinit", msg.to_bytes()])
            self._quitting.set()

        self.loop.create_task(_deinit())
        self.logger.debug("Queued loop for deinitialization")

    def stop(self) -> None:
        self.logger.debug("Stopping command runner")
        msg = StopMsg(node_id="command")
        self.loop.call_soon_threadsafe(
            self.sockets["outbox"].send_multipart, [b"stop", msg.to_bytes()]
        )
        self.logger.debug("Command runner stopped")

    def _init_sockets(self) -> None:
        self._init_outbox()
        self._init_router()
        self._init_inbox()

    def _init_outbox(self) -> None:
        """Create the main control publisher"""
        pub = self.context.socket(zmq.PUB)
        pub.bind(self.pub_address)
        pub.setsockopt_string(zmq.IDENTITY, "command.outbox")
        self.register_socket("outbox", pub)

    def _init_router(self) -> None:
        """Create the inbox router"""
        router = self.context.socket(zmq.ROUTER)
        router.bind(self.router_address)
        router.setsockopt_string(zmq.IDENTITY, "command.router")
        self.register_socket("router", router, receiver=True)
        self.add_callback("router", self.on_router)
        self.logger.debug("Router bound to %s", self.router_address)

    def _init_inbox(self) -> None:
        """Subscriber that receives all events from running nodes"""
        sub = self.context.socket(zmq.SUB)
        sub.setsockopt_string(zmq.IDENTITY, "command.inbox")
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.register_socket("inbox", sub, receiver=True)

    async def announce(self) -> None:
        msg = AnnounceMsg(
            node_id="command", value=AnnounceValue(inbox=self.router_address, nodes=self._nodes)
        )
        await self.sockets["outbox"].send_multipart([b"announce", msg.to_bytes()])

    async def ping(self) -> None:
        """Send a ping message asking everyone to identify themselves"""
        msg = PingMsg(node_id="command")
        await self.sockets["outbox"].send_multipart([b"ping", msg.to_bytes()])

    def start(self, n: int | None = None) -> None:
        """
        Start running in free-run mode
        """
        self.loop.call_soon_threadsafe(
            self.sockets["outbox"].send_multipart,
            [b"start", StartMsg(node_id="command", value=n).to_bytes()],
        )
        self.logger.debug("Sent start message")

    def process(self, epoch: int, input: dict | None = None) -> None:
        """Emit a ProcessMsg to process a single round through the graph"""
        # no empty dicts
        input = input if input else None
        self.loop.call_soon_threadsafe(
            self.sockets["outbox"].send_multipart,
            [
                b"process",
                ProcessMsg(node_id="command", value={"input": input, "epoch": epoch}).to_bytes(),
            ],
        )
        self.logger.debug("Sent process message")

    def await_ready(self, node_ids: list[NodeID], timeout: float = 10) -> None:
        """
        Wait until all the node_ids have announced themselves
        """

        def _ready_nodes() -> set[str]:
            return {node_id for node_id, state in self._nodes.items() if state["status"] == "ready"}

        def _is_ready() -> bool:
            ready_nodes = _ready_nodes()
            waiting_for = set(node_ids)
            self.logger.debug(
                "Checking if ready, ready nodes are: %s, waiting for %s",
                ready_nodes,
                waiting_for,
            )
            return waiting_for.issubset(ready_nodes)

        with self._ready_condition:
            # ping periodically for identifications in case we have slow subscribers
            start_time = time()
            ready = False
            while time() < start_time + timeout and not ready:
                ready = self._ready_condition.wait_for(_is_ready, timeout=1)
                if not ready:
                    self.loop.call_soon_threadsafe(self.loop.create_task, self.ping())

        # if still not ready, timeout
        if not ready:
            raise TimeoutError(
                f"Nodes were not ready after the timeout. "
                f"Waiting for: {set(node_ids)}, "
                f"ready: {_ready_nodes()}"
            )

    async def on_router(self, message: Message) -> None:
        self.logger.debug("Received ROUTER message %s", message)

        if message.type_ == MessageType.identify:
            message = cast(IdentifyMsg, message)
            await self.on_identify(message)
        elif message.type_ == MessageType.status:
            message = cast(StatusMsg, message)
            await self.on_status(message)

    async def on_identify(self, msg: IdentifyMsg) -> None:
        self._nodes[msg.node_id] = msg.value
        self.sockets["inbox"].connect(msg.value["outbox"])

        try:
            await self.announce()
            self.logger.debug("Announced")
        except Exception as e:
            self.logger.exception("Exception announced: %s", e)

        with self._ready_condition:
            self._ready_condition.notify_all()

    async def on_status(self, msg: StatusMsg) -> None:
        if msg.node_id not in self._nodes:
            self.logger.warning(
                "Node %s sent us a status before sending its full identify message, ignoring",
                msg.node_id,
            )
            return
        self._nodes[msg.node_id]["status"] = msg.value

        with self._ready_condition:
            self._ready_condition.notify_all()


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
        self._freerun = asyncio.Event()
        self._process_one = asyncio.Event()
        self._status: NodeStatus = NodeStatus.stopped
        self._status_lock = asyncio.Lock()
        self._ready_condition = asyncio.Condition()
        self._to_process = 0
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
                    self._ready_condition.notify_all()

                # nodes should not report epoch endings since they don't know about the full tube
                events = [e for e in events if e["node_id"] != "meta"]
                if events:
                    await self.update_graph(events)
                    await self.publish_events(events)

    async def await_inputs(self) -> AsyncGenerator[tuple[tuple[Any], dict[str, Any], int]]:
        self._node = cast(Node, self._node)
        while not self._quitting.is_set():
            # if we are not freerunning, keep track of how many times we are supposed to run,
            # and run until we aren't supposed to anymore!
            if not self._freerun.is_set():
                if self._to_process <= 0:
                    self._to_process = 0
                    await self._process_one.wait()
                self._to_process -= 1
                if self._to_process <= 0:
                    self._to_process = 0
                    self._process_one.clear()

            epoch = next(self._counter) if self._node.stateful else None

            ready = await self.await_node(epoch=epoch)
            edges = self._node.edges
            inputs = self.store.collect(edges, ready["epoch"])
            if inputs is None:
                inputs = {}
            args, kwargs = self.store.split_args_kwargs(inputs)
            # clear events for this epoch, since we have consumed what we need here.
            self.store.clear(ready["epoch"])
            yield args, kwargs, ready["epoch"]

    async def update_graph(self, events: list[Event]) -> None:
        async with self._ready_condition:
            self.scheduler.update(events)
            self._ready_condition.notify_all()

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
        self.scheduler = Scheduler(nodes={self.spec.id: self.spec}, edges=self._node.edges)
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
        events = msg.value
        if not self.depends:
            self.logger.debug("No dependencies, not storing events")
            return

        to_add = [e for e in events if (e["node_id"], e["signal"]) in self.depends]
        for event in to_add:
            self.store.add(event)
        self.logger.debug("scheduler updating")
        async with self._ready_condition:
            self.scheduler.update(events)
            self._ready_condition.notify_all()
        self.logger.debug("scheduler updated")

    async def on_start(self, msg: StartMsg) -> None:
        """
        Start running in free mode
        """
        await self.update_status(NodeStatus.running)
        if msg.value is None:
            self._freerun.set()
        else:
            self._to_process += msg.value
        self._process_one.set()

    async def on_process(self, msg: ProcessMsg) -> None:
        """
        Process a single graph iteration
        """
        self.logger.debug("Received Process message: %s", msg)
        self._to_process += 1
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
        self._to_process = 0
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
        msg = ErrorMsg(
            node_id=self.spec.id,
            value=ErrorValue(
                err_type=type(err),
                err_args=err.args,
                traceback=tbexception,
            ),
        )
        await self.sockets["dealer"].send_multipart([msg.to_bytes()])

    async def await_node(self, epoch: int | None = None) -> MetaEvent:
        """
        Block until a node is ready

        Args:
            node_id:
            epoch (int, None): if `int` , wait until the node is ready in the given epoch,
                otherwise wait until the node is ready in any epoch

        Returns:

        """
        async with self._ready_condition:
            await self._ready_condition.wait_for(
                lambda: self.scheduler.node_is_ready(self.spec.id, epoch)
            )

            # be FIFO-like and get the earliest epoch the node is ready in
            if epoch is None:
                for ep in self.scheduler._epochs:
                    if self.scheduler.node_is_ready(self.spec.id, ep):
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
            self.scheduler[epoch].mark_out(self.spec.id)

        return MetaEvent(
            id=uuid4().int,
            timestamp=datetime.now(),
            node_id="meta",
            signal=MetaEventType.NodeReady,
            epoch=epoch,
            value=self.spec.id,
        )


@dataclass
class ZMQRunner(TubeRunner):
    """
    A concurrent runner that uses zmq to broker events between nodes running in separate processes
    """

    node_procs: dict[NodeID, mp.Process] = field(default_factory=dict)
    command: CommandNode | None = None
    quit_timeout: float = 10
    """time in seconds to wait after calling deinit to wait before killing runner processes"""
    store: EventStore = field(default_factory=EventStore)
    autoclear_store: bool = True
    """
    If ``True`` (default), clear the event store after events are processed and returned.
    If ``False`` , don't clear events from the event store
    """

    _initialized: EventType = field(default_factory=mp.Event)
    _running: EventType = field(default_factory=mp.Event)
    _init_lock: threading.RLock = field(default_factory=threading.RLock)
    _running_lock: threading.Lock = field(default_factory=threading.Lock)
    _ignore_events: bool = False
    _return_node: Return | None = None
    _to_throw: ErrorValue | None = None
    _current_epoch: int = 0
    _epoch_futures: dict[int, concurrent.futures.Future] = field(default_factory=dict)

    @property
    def running(self) -> bool:
        with self._running_lock:
            return self._running.is_set()

    @property
    def initialized(self) -> bool:
        with self._init_lock:
            return self._initialized.is_set()

    def init(self) -> None:
        if self.running:
            return
        with self._init_lock:
            self._logger.debug("Initializing ZMQ runner")
            self.command = CommandNode(runner_id=self.runner_id)
            threading.Thread(target=self.command.run, daemon=True).start()
            self.command._init.wait()
            self.command.add_callback("inbox", self.on_event)
            self.command.add_callback("router", self.on_router)
            self._logger.debug("Command node initialized")

            for node_id, node in self.tube.nodes.items():
                if isinstance(node, Return):
                    self._return_node = node
                    continue
                self.node_procs[node_id] = mp.Process(
                    target=NodeRunner.run,
                    args=(node.spec,),
                    kwargs={
                        "runner_id": self.runner_id,
                        "command_outbox": self.command.pub_address,
                        "command_router": self.command.router_address,
                        "input_collection": self.tube.input_collection,
                    },
                    name=".".join([self.runner_id, node_id]),
                    daemon=True,
                )
                self.node_procs[node_id].start()
            self._logger.debug("Started node processes, awaiting ready")
            try:
                self.command.await_ready(
                    [k for k, v in self.tube.nodes.items() if not isinstance(v, Return)]
                )
            except TimeoutError as e:
                self._logger.debug("Timeouterror, deinitializing before throwing")
                self._initialized.set()
                self.deinit()
                self._logger.exception(e)
                raise

            self._logger.debug("Nodes ready")
            self._initialized.set()

    def deinit(self) -> None:
        if not self.initialized:
            return

        with self._init_lock:
            self.command = cast(CommandNode, self.command)
            self.command.stop()
            # wait for nodes to finish, if they don't finish in the timeout, kill them
            started_waiting = time()
            waiting_on = set(self.node_procs.values())
            while time() < started_waiting + self.quit_timeout and len(waiting_on) > 0:
                _waiting = waiting_on.copy()
                for proc in _waiting:
                    if not proc.is_alive():
                        waiting_on.remove(proc)
                    else:
                        proc.terminate()

            for proc in waiting_on:
                self._logger.info(
                    f"NodeRunner {proc.name} was still alive after timeout expired, killing it"
                )
                proc.kill()
                try:
                    proc.close()
                except ValueError:
                    self._logger.info(
                        f"NodeRunner {proc.name} still not closed! making an unclean exit."
                    )

            self.command.clear_callbacks()
            self.command.deinit()
            self.tube.scheduler.clear()
            self._initialized.clear()

    def process(self, **kwargs: Any) -> ReturnNodeType:
        if not self.initialized:
            self._logger.info("Runner called process without calling `init`, initializing now.")
            self.init()
        if self.running:
            raise RuntimeError(
                "Runner is already running in free run mode! use iter to gather results"
            )
        input = self.tube.input_collection.validate_input(InputScope.process, kwargs)
        self._running.set()
        try:
            self._current_epoch = self.tube.scheduler.add_epoch()
            # we want to mark 'input' as done if it's in the topo graph,
            # but input can be present and only used as a param,
            # so we can't check presence of inputs in the input collection
            if "input" in self.tube.scheduler._epochs[self._current_epoch].ready_nodes:
                self.tube.scheduler.done(self._current_epoch, "input")
            self.command = cast(CommandNode, self.command)
            self.command.process(self._current_epoch, input)
            self._logger.debug("awaiting epoch %s", self._current_epoch)
            self.await_epoch(self._current_epoch)
            self._logger.debug("collecting return")

            return self.collect_return(self._current_epoch)
        finally:
            self._running.clear()

    def iter(self, n: int | None = None) -> Generator[ReturnNodeType, None, None]:
        """
        Iterate over results as they are available.

        Tube runs in free-run mode for n iterations,
        This method is usually only useful for tubes with :class:`.Return` nodes.
        This method yields only when return is available:
        the tube will run more than n ``process`` calls if there are e.g. gather nodes
        that cause the return value to be empty.

        To call the tube a specific number of times and do something with the events
        other than returning a value, use callbacks and :meth:`.run` !

        Note that backpressure control is not yet implemented!!!
        If the outer iter method is slow, or there is a bottleneck in your tube,
        you might incur some serious memory usage!
        Backpressure and observability is a WIP!

        If you need a version of this method that *always* makes a fixed number of process calls,
        raise an issue!
        """
        if not self.initialized:
            raise RuntimeError(
                "ZMQRunner must be explicitly initialized and deinitialized, "
                "use the runner as a contextmanager or call `init()` and `deinit()`"
            )
        try:
            _ = self.tube.input_collection.validate_input(InputScope.process, {})
        except InputMissingError as e:
            raise InputMissingError(
                "Can't use the `iter` method with tubes with process-scoped input "
                "that was not provided when instantiating the tube! "
                "Use `process()` directly, providing required inputs to each call."
            ) from e
        if self.running:
            raise RuntimeError("Already Running!")
        self.command = cast(CommandNode, self.command)

        epoch = self._current_epoch
        start_epoch = epoch
        stop_epoch = epoch + n if n is not None else epoch
        # start running without a limit - we'll check as we go.
        self.command.start(n)
        self._running.set()
        current_iter = 0
        try:
            while n is None or current_iter < n:
                ret = MetaSignal.NoEvent
                loop = 0
                while ret is MetaSignal.NoEvent:
                    self._logger.debug("Awaiting epoch %s", epoch)
                    self.await_epoch(epoch)
                    ret = self.collect_return(epoch)
                    epoch += 1
                    self._current_epoch = epoch
                    if loop > self.max_iter_loops:
                        raise RuntimeError("Reached maximum process calls per iteration")
                    # if we have run out of epochs to run, request some more with a cheap heuristic
                    if n is not None and epoch >= stop_epoch:
                        stop_epoch += self._request_more(
                            n=n, current_iter=current_iter, n_epochs=stop_epoch - start_epoch
                        )

                current_iter += 1
                yield ret

        finally:
            self.stop()

    @overload
    def run(self, n: int) -> list[ReturnNodeType]: ...

    @overload
    def run(self, n: None = None) -> None: ...

    def run(self, n: int | None = None) -> None | list[ReturnNodeType]:
        """
        Run the tube in freerun mode - every node runs as soon as its dependencies are satisfied,
        not waiting for epochs to complete before starting the next epoch.

        Blocks when ``n`` is not None -
        This is for consistency with the synchronous/asyncio runners,
        but may change in the future.

        If ``n`` is None, does not block.
        stop processing by calling :meth:`.stop` or deinitializing
        (exiting the contextmanager, or calling :meth:`.deinit`)
        """
        if not self.initialized:
            raise RuntimeError(
                "ZMQRunner must be explicitly initialized and deinitialized, "
                "use the runner as a contextmanager or call `init()` and `deinit()`"
            )
        if self.running:
            raise RuntimeError("Already Running!")
        try:
            _ = self.tube.input_collection.validate_input(InputScope.process, {})
        except InputMissingError as e:
            raise InputMissingError(
                "Can't use the `iter` method with tubes with process-scoped input "
                "that was not provided when instantiating the tube! "
                "Use `process()` directly, providing required inputs to each call."
            ) from e
        self.command = cast(CommandNode, self.command)

        if n is None:
            if self.autoclear_store:
                self._ignore_events = True
            self.command.start()
            self._running.set()
            return None

        elif self.tube.has_return:
            # run until n return values
            results = []
            for res in self.iter(n):
                results.append(res)
            return results

        else:
            # run n epochs
            self.command.start(n)
            self._running.set()
            self._current_epoch = self.await_epoch(self._current_epoch + n)
            return None

    def stop(self) -> None:
        """
        Stop running the tube.
        """
        self.command = cast(CommandNode, self.command)
        self._ignore_events = False
        self.command.stop()
        self._running.clear()

    def on_event(self, msg: Message) -> None:
        self._logger.debug("EVENT received: %s", msg)
        if msg.type_ != MessageType.event:
            self._logger.debug(f"Ignoring message type {msg.type_}")
            return

        msg = cast(EventMsg, msg)
        # store events (if we are not in freerun mode, where we don't want to store infinite events)
        if not self._ignore_events:
            for event in msg.value:
                self.store.add(event)
        events = self.tube.scheduler.update(msg.value)
        events = cast(MutableSequence[Event | MetaEvent], events)
        if self._return_node is not None:
            # mark the return node done if we've received the expected events for an epoch
            # do it here since we don't really run the return node like a real node
            # to avoid an unnecessary pickling/unpickling across the network
            epochs = set(e["epoch"] for e in msg.value)
            for epoch in epochs:
                if (
                    self.tube.scheduler.node_is_ready(self._return_node.id, epoch)
                    and epoch in self.tube.scheduler._epochs
                ):
                    self._logger.debug("Marking return node ready in epoch %s", epoch)
                    ep_ended = self.tube.scheduler.done(epoch, self._return_node.id)
                    if ep_ended is not None:
                        events.append(ep_ended)
        for e in events:
            if (
                e["node_id"] == "meta"
                and e["signal"] == MetaEventType.EpochEnded
                and e["value"] in self._epoch_futures
            ):
                self._epoch_futures[e["value"]].set_result(e["value"])
                del self._epoch_futures[e["value"]]

    def on_router(self, msg: Message) -> None:
        if isinstance(msg, ErrorMsg):
            self._handle_error(msg)

    def collect_return(self, epoch: int | None = None) -> Any:
        if epoch is None:
            raise ValueError("Must specify epoch in concurrent runners")
        if self._return_node is None:
            return None
        else:
            events = self.store.collect(self._return_node.edges, epoch)
            if events is None:
                return MetaSignal.NoEvent
            args, kwargs = self.store.split_args_kwargs(events)
            self._return_node.process(*args, **kwargs)
            ret = self._return_node.get(keep=False)
            if self.autoclear_store:
                self.store.clear(epoch)
            return ret

    def _handle_error(self, msg: ErrorMsg) -> None:
        """Cancel current epoch, stash error for process method to throw"""
        self._logger.error("Received error from node: %s", msg)
        exception = msg.to_exception()
        self._to_throw = msg.value
        if self._current_epoch is not None:
            # if we're waiting in the process method,
            # end epoch and raise error there
            self.tube.scheduler.end_epoch(self._current_epoch)
            self.deinit()
            if self._current_epoch in self._epoch_futures:
                self._epoch_futures[self._current_epoch].set_exception(exception)
                del self._epoch_futures[self._current_epoch]
            else:
                raise exception
        else:
            # e.g. errors during init, raise here.
            raise exception

    def _request_more(self, n: int, current_iter: int, n_epochs: int) -> int:
        """
        During iteration with cardinality-reducing nodes,
        if we haven't gotten the requested n return values in n epochs,
        request more epochs based on how many return values we got for n iterations

        Args:
            n (int): number of requested return values
            current_iter (int): current number of return values that have been collected
            n_epochs (int): number of epochs that have run
        """
        self.command = cast(CommandNode, self.command)
        n_remaining = n - current_iter
        if n_remaining <= 0:
            self._logger.warning(
                "Asked to request more epochs, but already collected enough return values. "
                "Ignoring. "
                "Requested n: %s, collected n: %s",
                n,
                current_iter,
            )
            return 0

        # if we get one return value every 5 epochs,
        # and we ran 5 epochs to get 1 result,
        # then we need to run 20 more to get the other 4,
        # or, (n remaining) * (epochs per result)
        # so...
        divisor = current_iter if current_iter > 0 else 1
        get_more = math.ceil(n_remaining * (n_epochs / divisor))
        self.command.start(get_more)
        return get_more

    def enable_node(self, node_id: str) -> None:
        raise NotImplementedError()

    def disable_node(self, node_id: str) -> None:
        raise NotImplementedError()

    def await_epoch(self, epoch: int) -> int:
        if self.tube.scheduler.epoch_completed(epoch):
            return epoch

        if epoch not in self._epoch_futures:
            self._epoch_futures[epoch] = concurrent.futures.Future()
        return self._epoch_futures[epoch].result()
