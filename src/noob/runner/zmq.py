"""
- Central command pub/sub
- each sub-runner has its own set of sockets for publishing and consuming events
- use the `node_id.signal` etc. as basically a feed address
"""

import base64
import json
import multiprocessing as mp
import pickle
import threading
from collections import defaultdict
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from itertools import count
from time import time
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast
from typing import Annotated as A

import zmq
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    PlainSerializer,
    Tag,
    TypeAdapter,
)
from tornado.ioloop import IOLoop

# from zmq.eventloop.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream

from noob.config import config
from noob.event import Event
from noob.input import InputCollection
from noob.logging import init_logger
from noob.node import Node, NodeSpecification, Return
from noob.runner.base import TubeRunner
from noob.scheduler import Scheduler, SchedulerMode
from noob.store import EventStore
from noob.types import NodeID, ReturnNodeType

if TYPE_CHECKING:
    pass


class Callbacks(TypedDict, total=False):
    on_inbox: Callable[[Event], Any]
    on_command: Callable[[Event], Any]


class MessageType(StrEnum):
    announce = "announce"
    identify = "identify"
    start = "start"
    stop = "stop"
    event = "event"


def _to_json(val: Event) -> str:
    try:
        return json.dumps(val)
    except TypeError:
        # pickle and b64encode
        return "pck__" + base64.b64encode(pickle.dumps(val)).decode("utf-8")


def _from_json(val: Any) -> Event:
    if isinstance(val, str):
        if val.startswith("pck__"):
            return pickle.loads(base64.b64decode(val[5:]))
        else:
            return Event(**json.loads(val))
    else:
        return val


SerializableEvent = A[
    Event, PlainSerializer(_to_json, when_used="json"), BeforeValidator(_from_json)
]


class Message(BaseModel):
    type_: MessageType = Field(..., alias="type")
    node_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    value: dict | str | None = None

    model_config = ConfigDict(use_enum_values=True, validate_by_alias=True, serialize_by_alias=True)

    @classmethod
    def from_bytes(cls, msg: list[bytes]) -> "Message":
        return MessageAdapter.validate_json(msg[-1].decode("utf-8"))

    def to_bytes(self) -> bytes:
        return self.model_dump_json().encode("utf-8")


class NodeIdentifyMessage(TypedDict):
    node_id: str
    outbox: str
    signals: list[str] | None
    slots: list[str] | None


class AnnounceMessage(TypedDict):
    """Command node 'announces' identities of other peers and the events they emit"""

    inbox: str
    nodes: dict[str, NodeIdentifyMessage]


class AnnounceMsg(Message):
    type_: Literal[MessageType.announce] = Field(MessageType.announce, alias="type")
    value: AnnounceMessage


class IdentifyMsg(Message):
    type_: Literal[MessageType.identify] = Field(MessageType.identify, alias="type")
    value: NodeIdentifyMessage


class StartMsg(Message):
    type_: Literal[MessageType.start] = Field(MessageType.start, alias="type")
    value: None = None


class StopMsg(Message):
    type_: Literal[MessageType.stop] = Field(MessageType.stop, alias="type")
    value: None = None


class EventMsg(Message):
    type_: Literal[MessageType.event] = Field(MessageType.event, alias="type")
    value: list[SerializableEvent]


def _type_discriminator(v: dict | Message) -> str:
    typ = v.get("type", "any") if isinstance(v, dict) else v.type_

    if typ in MessageType.__members__:
        return typ
    else:
        return "any"


MessageUnion = A[
    A[AnnounceMessage, Tag("announce")]
    | A[IdentifyMsg, Tag("identify")]
    | A[StartMsg, Tag("start")]
    | A[StopMsg, Tag("stop")]
    | A[EventMsg, Tag("event")]
    | A[Message, Tag("any")],
    Discriminator(_type_discriminator),
]
MessageAdapter = TypeAdapter(MessageUnion)


class NetworkMixin:

    def __init__(self):
        self._context = None
        self._loop = None
        self._quitting = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def context(self) -> zmq.Context:
        if self._context is None:
            self._context = zmq.Context.instance()
        return self._context

    @property
    def loop(self) -> IOLoop:
        if self._loop is None:
            self._loop = IOLoop.current()
        return self._loop

    def start_loop(self) -> None:
        if self._thread is not None:
            raise RuntimeWarning("Command node already started")

        self._quitting.clear()

        _ready = threading.Event()

        def _signal_ready() -> None:
            _ready.set()

        def _run() -> None:
            if hasattr(self, "logger"):
                self.logger.debug("Starting eventloop")
            while not self._quitting.is_set():
                try:
                    self.loop.add_callback(_signal_ready)
                    self.loop.start()

                except RuntimeError:
                    # loop already started
                    if hasattr(self, "logger"):
                        self.logger.debug("Eventloop already started, quitting")
                    break
            if hasattr(self, "logger"):
                self.logger.debug("Stopping eventloop")
            self._thread = None

        self._thread = threading.Thread(target=_run)
        self._thread.start()
        # wait until the loop has started
        _ready.wait(5)
        if hasattr(self, "logger"):
            self.logger.debug("Event loop started")

    def stop_loop(self) -> None:
        if self._thread is None:
            return
        self._quitting.set()
        self.loop.stop()
        self._thread.join()


class CommandNode(NetworkMixin):
    """
    Pub node that controls the state of the other nodes/announces addresses

    - one PUB socket to distribute commands
    - one ROUTER socket to receive return messages from runner nodes
    """

    def __init__(self, runner_id: str, protocol: str = "ipc", port: int | None = None):
        """

        Args:
            runner_id (str): The unique ID for the runner/tube session.
                All nodes within a runner use this to limit communication within a tube
            protocol:
            port:
        """
        super().__init__()
        self.runner_id = runner_id
        self.port = port
        self.protocol = protocol
        self.logger = init_logger(f"runner.command.{runner_id}")
        self._pub = None
        self._sub = None
        self._router = None
        self._context = None
        self._loop = None
        self._thread: threading.Thread | None = None
        self._quitting = threading.Event()
        self._nodes: dict[str, NodeIdentifyMessage] = {}
        self._ready_condition = threading.Condition()
        self._callbacks: dict[str, list[Callable[[Message], ...]]] = defaultdict(list)

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
            # return "tcp://127.0.0.1:8991"
            path = config.tmp_dir / f"{self.runner_id}/command/inbox"
            path.parent.mkdir(parents=True, exist_ok=True)
            return f"{self.protocol}://{str(path)}"
        else:
            raise NotImplementedError()

    def start(self) -> None:
        self.logger.debug("Starting command runner")
        self.init_sockets()
        self.start_loop()
        self.logger.debug("Command runner started")

    def stop(self) -> None:
        self.logger.debug("Stopping command runner")
        msg = StopMsg(node_id="command")
        self._pub.send_multipart([b"stop", msg.to_bytes()])
        self.stop_loop()
        self.logger.debug("Command runner stopped")

    def init_sockets(self) -> None:
        self._pub = self.init_pub()
        self._router = self.init_router()
        self._sub = self.init_subscriber()

    def init_pub(self) -> zmq.Socket:
        """Create the main control publisher"""
        pub = self.context.socket(zmq.PUB)
        pub.bind(self.pub_address)
        pub.setsockopt_string(zmq.IDENTITY, "command.outbox")
        return pub

    def init_router(self) -> ZMQStream:
        """Create the inbox router"""
        router = self.context.socket(zmq.ROUTER)
        router.bind(self.router_address)
        router.setsockopt_string(zmq.IDENTITY, "command.inbox")
        router = ZMQStream(router, self.loop)
        router.on_recv(self.on_inbox)
        self.logger.debug("Inbox bound to %s", self.router_address)
        return router

    def init_subscriber(self) -> ZMQStream:
        """Subscriber that receives all events from running nodes"""
        sub = self.context.socket(zmq.SUB)
        sub.setsockopt_string(zmq.IDENTITY, "command.subscriber")
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        sub = ZMQStream(sub, self.loop)
        sub.on_recv(self.on_sub)
        return sub

    def announce(self) -> None:
        msg = AnnounceMsg(
            node_id="command", value=AnnounceMessage(inbox=self.router_address, nodes=self._nodes)
        )
        self._pub.send_multipart([b"announce", msg.to_bytes()])

    def add_callback(self, type_: Literal["inbox", "event"], cb: Callable[[Message], Any]) -> None:
        self._callbacks[type_].append(cb)

    def await_ready(self, node_ids: list[NodeID]) -> None:
        """
        Wait until all the node_ids have announced themselves
        """
        with self._ready_condition:
            if set(node_ids) == set(self._nodes):
                return
            self._ready_condition.wait_for(lambda: set(node_ids) == set(self._nodes))

    def publish_input(self, **kwargs: Any) -> None:
        """
        Publish process-scoped input to tubes
        """
        raise NotImplementedError()

    def on_inbox(self, msg: list[bytes]) -> None:

        try:
            msg = Message.from_bytes(msg)
            self.logger.debug("Received INBOX message %s", msg)
        except Exception as e:
            self.logger.exception("Exception decoding: %s,  %s", msg, e)
            raise e

        if msg.type_ == MessageType.identify:
            msg = cast(IdentifyMsg, msg)
            self.on_identify(msg)
        else:
            raise NotImplementedError()

        for cb in self._callbacks["inbox"]:
            cb(msg)

    def on_sub(self, msg: list[bytes]) -> None:
        msg = Message.from_bytes(msg)
        self.logger.debug("Received SUBSCRIBER message: %s", msg)
        for cb in self._callbacks["event"]:
            cb(msg)

    def on_identify(self, msg: IdentifyMsg) -> None:
        with self._ready_condition:
            self._nodes[msg.node_id] = msg.value
            self._sub.connect(msg.value["outbox"])
            self._ready_condition.notify_all()

        try:
            self.announce()
            self.logger.debug("Announced")
        except Exception as e:
            self.logger.exception("Exception announced: %s", e)


class NodeRunner(NetworkMixin):
    """
    Runner for a single node

    - DEALER to communicate with command inbox
    - PUB to publish events
    - SUB to subscribe to events
    """

    def __init__(
        self,
        spec: NodeSpecification,
        runner_id: str,
        command_outbox: str,
        command_inbox: str,
        input_collection: InputCollection,
        protocol: str = "ipc",
        callbacks: Callbacks | None = None,
    ):
        super().__init__()
        self.spec = spec
        self.runner_id = runner_id
        self.input_collection = input_collection
        self.command_outbox = command_outbox
        self.command_inbox = command_inbox
        self.protocol = protocol
        self.store = EventStore()
        self.scheduler: Scheduler | None = None
        self.logger = init_logger(f"runner.node.{runner_id}.{self.spec.id}")
        if callbacks is None:
            self.callbacks = Callbacks()
        else:
            self.callbacks = callbacks

        self._dealer: zmq.Socket | ZMQStream | None = None
        self._pub: zmq.Socket | None = None
        self._sub = None
        self._node: Node | None = None
        self._depends: tuple[tuple[str, str], ...] | None = None
        self._nodes: dict[str, NodeIdentifyMessage] = {}
        self._counter = count()
        self._process_quitting = mp.Event()

    @property
    def pub_address(self) -> str:
        if self.protocol == "ipc":
            path = config.tmp_dir / f"{self.runner_id}/nodes/{self.spec.id}/outbox"
            path.parent.mkdir(parents=True, exist_ok=True)
            return f"{self.protocol}://{str(path)}"
        else:
            raise NotImplementedError()

    @property
    def depends(self) -> tuple[tuple[str, str], ...] | None:
        if self._depends is not None:
            return self._depends
        elif self._node is None:
            return None
        else:
            self._depends = tuple(
                (edge.source_node, edge.source_signal) for edge in self._node.edges
            )

    @classmethod
    def run(cls, spec: NodeSpecification, **kwargs: Any) -> None:
        """
        Target for multiprocessing.run,
        init the class and start it!
        """
        runner = NodeRunner(spec=spec, **kwargs)
        runner.init()

        runner._process_quitting.clear()
        for args, kwargs, epoch in runner.await_inputs():
            runner.logger.debug("Running with args: %s, kwargs: %s, epoch: %s", args, kwargs, epoch)
            value = runner._node.process(*args, **kwargs)
            events = runner.store.add_value(runner._node.signals, value, runner._node.id, epoch)
            runner.update_graph(events)
            runner.publish_events(events)

        runner.deinit()

    def await_inputs(self) -> Generator[tuple[list[Any], dict[Any], int]]:

        while not self._process_quitting.is_set():
            ready = self.scheduler.await_node(self.spec.id)
            edges = self._node.edges
            inputs = self.store.collect(edges, ready["epoch"])
            if inputs is None:
                inputs = {}

            # TODO: Move this to `EventStore.transform_events`
            args = []
            kwargs = {}
            for k, v in inputs.items():
                if isinstance(k, int | None):
                    args.append((k, v))
                else:
                    kwargs[k] = v
            args = [item[1] for item in sorted(args, key=lambda x: x[0])]
            yield args, kwargs, ready["epoch"]

    def update_graph(self, events: list[Event]) -> None:
        self.scheduler.update(events)

    def publish_events(self, events: list[Event]) -> None:
        msg = EventMsg(node_id=self.spec.id, value=events)
        self._pub.send_multipart([b"event", msg.to_bytes()])

    def init(self) -> None:
        self.logger.debug("Initializing")

        self.init_node()
        self.start_sockets()
        self.identify()
        self.logger.debug("Initialization finished")

    def deinit(self) -> None:
        self.logger.debug("Deinitializing")
        self._node.deinit()
        self.stop_loop()
        self.context.destroy()
        self.logger.debug("Deinitialization finished")

    def identify(self) -> None:
        """
        Send the command node an announce to say we're alive
        """
        ann = IdentifyMsg(
            node_id=self.spec.id,
            value=NodeIdentifyMessage(
                node_id=self.spec.id,
                outbox=self.pub_address,
                signals=[s.name for s in self._node.signals] if self._node.signals else None,
                slots=[slot_name for slot_name in self._node.slots] if self._node.slots else None,
            ),
        )
        self._dealer.send_multipart([ann.to_bytes()])
        self.logger.debug("Sent identification message: %s", ann.to_bytes())

    def start_sockets(self) -> None:
        self.init_sockets()
        self.start_loop()

    def init_node(self) -> None:
        self._node = Node.from_specification(self.spec, self.input_collection)
        self._node.init()
        self.scheduler = Scheduler(
            nodes={self.spec.id: self.spec}, edges=self._node.edges, mode=SchedulerMode.parallel
        )
        self.scheduler.add_epoch()

    def init_sockets(self) -> None:
        self._dealer = self.init_dealer()
        self._pub = self.init_pub()
        self._sub = self.init_sub()

    def init_dealer(self) -> ZMQStream:
        dealer = self.context.socket(zmq.DEALER)
        dealer.setsockopt_string(zmq.IDENTITY, self.spec.id)
        res = dealer.connect(self.command_inbox)
        self.logger.debug(str(dir(res)))
        dealer = ZMQStream(dealer, self.loop)
        dealer.on_recv(self.on_outbox)
        self.logger.debug("Connected to command node at %s", self.command_inbox)
        return dealer

    def init_pub(self) -> ZMQStream:
        pub = self.context.socket(zmq.PUB)
        pub.setsockopt_string(zmq.IDENTITY, self.spec.id)
        if self.protocol == "ipc":
            pub.bind(self.pub_address)
        else:
            raise NotImplementedError()
            # something like:
            # port = pub.bind_to_random_port(self.protocol)

        pub = ZMQStream(pub, self.loop)
        pub.on_recv(self.on_inbox)
        return pub

    def init_sub(self) -> ZMQStream:
        """
        Init the subscriber, but don't attempt to subscribe to anything but the command yet!
        we do that when we get node Announces
        """
        sub = self.context.socket(zmq.SUB)
        sub.setsockopt_string(zmq.IDENTITY, self.spec.id)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        sub.connect(self.command_outbox)
        sub = ZMQStream(sub, self.loop)
        sub.on_recv(self.on_inbox)
        self.logger.debug("Subscribed to command outbox %s", self.command_outbox)
        return sub

    def on_outbox(self, msg: list[bytes]) -> None:
        self.logger.debug("DEALER received %s", msg)

    def on_inbox(self, msg: list[bytes]) -> None:

        try:
            msg = Message.from_bytes(msg)

            self.logger.debug("INBOX received %s", msg)
        except Exception as e:
            self.logger.exception("Error decoding message %s %s", msg, e)
            raise e

        if msg.type_ == MessageType.announce:
            msg = cast(AnnounceMsg, msg)
            self.on_announce(msg)
        elif msg.type_ == MessageType.event:
            msg = cast(EventMsg, msg)
            self.on_event(msg)
        else:
            raise NotImplementedError()

    def on_announce(self, msg: AnnounceMsg) -> None:
        """
        Store map, connect to the nodes we depend on
        """
        for node_id in msg.value["nodes"]:
            if (
                node_id in {edge.source_node for edge in self._node.edges}
                and node_id not in self._nodes
            ):
                # TODO: a way to check if we're already connected, without storing it locally?
                outbox = msg.value["nodes"][node_id]["outbox"]
                self._sub.connect(outbox)
                self.logger.debug("Subscribed to %s at %s", node_id, outbox)
        self._nodes = msg.value["nodes"]

    def on_event(self, msg: EventMsg) -> None:
        events = msg.value
        to_add = [e for e in events if (e["node_id"], e["signal"]) in self.depends]
        for event in to_add:
            self.store.add(event)
        self.scheduler.update(events)


@dataclass
class ZMQRunner(TubeRunner):
    """
    A concurrent runner that uses zmq to broker events between nodes running in separate processes
    """

    node_procs: dict[NodeID, mp.Process] = field(default_factory=dict)
    command: CommandNode | None = None
    quit_timeout: float = 5
    """time in seconds to wait after calling deinit to wait before killing runner processes"""
    store: EventStore = field(default_factory=EventStore)

    _running: mp.Event = field(default_factory=mp.Event)
    _epoch: count = field(default_factory=count)
    _epoch_condition: threading.Condition = field(default_factory=threading.Condition)
    _return_node: Return | None = None
    _init_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def running(self) -> bool:
        with self._init_lock:
            return self._running.is_set()

    def init(self) -> None:
        if self.running:
            return
        with self._init_lock:
            self._logger.debug("Initializing ZMQ runner")
            self.command = CommandNode(runner_id=self.runner_id)
            self.command.start()
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
                        "command_inbox": self.command.router_address,
                        "input_collection": self.tube.input_collection,
                    },
                    name=".".join([self.runner_id, node_id]),
                    daemon=True,
                )
                self.node_procs[node_id].start()
            self._logger.debug("Started node processes, awaiting ready")
            # self.command.await_ready(list(self.tube.nodes.keys()))
            self._running.set()

    def deinit(self) -> None:
        self.command.stop()
        # wait for nodes to finish, if they don't finish in the timeout, kill them
        started_waiting = time()
        waiting_on = set(self.node_procs.values())
        while time() < started_waiting + self.quit_timeout and len(waiting_on) > 0:
            for proc in waiting_on:
                if not proc.is_alive():
                    waiting_on.remove(proc)

        for proc in waiting_on:
            self._logger.info(
                f"NodeRunner {proc.name} was still alive after timeout expired, killing it"
            )
            proc.kill()

    def process(self, **kwargs: Any) -> ReturnNodeType:
        if not self.running:
            self._logger.info("Runner called process without calling `init`, initializing now.")
            self.init()

        this_epoch = next(self._epoch)
        self._logger.debug("awaiting epoch %s", this_epoch)
        self.tube.scheduler.await_epoch(this_epoch)
        self._logger.debug("collecting return")
        return self.collect_return(this_epoch)

    def on_event(self, msg: EventMsg) -> None:
        self._logger.debug("EVENT received: %s", msg)
        if msg.type_ != MessageType.event:
            self._logger.debug(f"Ignoring message type {msg.type_}")

        with self._epoch_condition:
            for event in msg.value:
                self.store.add(event)
            self.tube.scheduler.update(msg.value)

    def collect_return(self, epoch: int | None = None) -> Any:
        if epoch is None:
            raise ValueError("Must specify epoch in concurrent runners")
        if self._return_node is None:
            return None
        else:
            self._return_node.process(**self.store.collect(self._return_node.edges, epoch))
            return self._return_node.get(keep=False)

    def enable_node(self, node_id: str) -> None:
        raise NotImplementedError()

    def disable_node(self, node_id: str) -> None:
        raise NotImplementedError()
