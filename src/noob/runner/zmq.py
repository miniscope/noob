"""
- Central command pub/sub
- each sub-runner has its own set of sockets for publishing and consuming events
- use the `node_id.signal` etc. as basically a feed address
"""

import multiprocessing as mp
import threading
from collections.abc import Callable, Generator
from datetime import UTC, datetime
from enum import StrEnum
from itertools import count
from typing import Annotated, Any, Literal, TypedDict, cast

import zmq
from pydantic import BaseModel, Discriminator, Field, TypeAdapter
from zmq.eventloop.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream

from noob.event import Event
from noob.input import InputCollection
from noob.node import Node, NodeSpecification, Signal, Slot
from noob.scheduler import Scheduler
from noob.store import EventStore


class Callbacks(TypedDict, total=False):
    on_inbox: Callable[[Event], Any]
    on_command: Callable[[Event], Any]


class MessageType(StrEnum):
    announce = "announce"
    identify = "identify"
    start = "start"
    stop = "stop"
    event = "event"


class NodeIdentifyMessage(TypedDict):
    node_id: str
    outbox: str
    signals: list[Signal] | None
    slots: list[Slot] | None


class AnnounceMessage(TypedDict):
    """Command node 'announces' identities of other peers and the events they emit"""

    inbox: str
    nodes: dict[str, NodeIdentifyMessage]


class Message(BaseModel):
    type_: MessageType = Field(..., alias="type")
    node_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    value: dict | str | None = None

    @classmethod
    def from_bytes(cls, msg: list[bytes]) -> "Message":
        raise NotImplementedError()

    def to_bytes(self) -> list[bytes]:
        raise NotImplementedError()


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
    value: list[Event]


MessageUnion = Annotated[
    AnnounceMessage | IdentifyMsg | StartMsg | StopMsg | EventMsg | Message, Discriminator("type")
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
            self._context = zmq.Context()
        return self._context

    @property
    def loop(self) -> IOLoop:
        if self._loop is None:
            self._loop = IOLoop().current()
        return self._loop

    def start_loop(self) -> None:
        if self._thread is not None:
            raise RuntimeWarning("Command node already started")

        self._quitting.clear()

        def _run() -> None:
            while not self._quitting.is_set():
                try:
                    self.loop.start()
                except RuntimeError:
                    # loop already started
                    break
            self._thread = None

        self._thread = threading.Thread(target=_run)
        self._thread.start()

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
        self._pub = None
        self._router = None
        self._context = None
        self._loop = None
        self._thread: threading.Thread | None = None
        self._quitting = threading.Event()
        self._nodes: dict[str, NodeIdentifyMessage] = {}

    @property
    def pub_address(self) -> str:
        """Address the publisher bound to"""
        if self.protocol == "ipc":
            return f"{self.protocol}://tmp/noob/{self.runner_id}/command/outbox"
        else:
            raise NotImplementedError()

    @property
    def router_address(self) -> str:
        """Address the return router is bound to"""
        if self.protocol == "ipc":
            return f"{self.protocol}://tmp/noob/{self.runner_id}/command/inbox"
        else:
            raise NotImplementedError()

    def start(self) -> None:
        self.init_sockets()
        self.start_loop()

    def stop(self) -> None:
        self.stop_loop()

    def init_sockets(self) -> None:
        self._pub = self.init_pub()
        self._router = self.init_router()

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
        return router

    def announce(self) -> None:
        msg = AnnounceMsg(
            node_id="command", value=AnnounceMessage(inbox=self.router_address, nodes=self._nodes)
        )
        self._pub.send_multipart(msg.to_bytes())

    def on_inbox(self, msg: list[bytes]) -> None:

        print("INBOX")
        print(msg)
        msg = Message.from_bytes(msg)
        if msg.type_ == MessageType.identify:
            self.on_identify(msg)
        else:
            raise NotImplementedError()

    def on_identify(self, msg: IdentifyMsg) -> None:
        self._nodes[msg.node_id] = msg.value
        self.announce()


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
        self._event_block: mp.Event = mp.Event()

    @property
    def pub_address(self) -> str:
        if self.protocol == "ipc":
            return f"{self.protocol}://tmp/noob/{self.runner_id}/nodes/{self.spec.id}/outbox"
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
            value = runner._node.process(*args, **kwargs)
            events = runner.store.add_value(
                runner._node.signals, value, runner._node.node_id, epoch
            )
            runner.update_graph(events)
            runner.publish_events(events)

        runner.deinit()

    def await_inputs(self) -> Generator[tuple[list[Any], dict[Any], int]]:

        while not self._process_quitting.is_set():
            ready = self.scheduler.await_node(self.spec.id)
            edges = self._node.edges
            inputs = self.store.collect(edges, ready["epoch"])

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
        raise NotImplementedError()

    def init(self) -> None:
        self.start_sockets()
        self.init_node()
        self.identify()

    def deinit(self) -> None:
        self._node.deinit()
        self.stop_loop()
        self.context.destroy()

    def identify(self) -> None:
        """
        Send the command node an announce to say we're alive
        """
        ann = IdentifyMsg(
            node_id=self.spec.id,
            value=NodeIdentifyMessage(
                node_id=self.spec.id,
                outbox=self.pub_address,
                signals=self._node.signals,
                slots=list(self._node.slots.values()),
            ),
        )
        self._dealer.send_multipart(ann.to_bytes())

    def start_sockets(self) -> None:
        self.init_sockets()
        self.start_loop()

    def init_node(self) -> None:
        self._node = Node.from_specification(self.spec, self.input_collection)
        self._node.init()
        self.scheduler = Scheduler(nodes={self.spec.id: self.spec}, edges=self._node.edges)
        self.scheduler.add_epoch()

    def init_sockets(self) -> None:
        self._dealer = self.init_dealer()
        self._pub = self.init_pub()
        self._sub = self.init_sub()

    def init_dealer(self) -> ZMQStream:
        dealer = self.context.socket(zmq.DEALER)
        dealer.setsockopt_string(zmq.IDENTITY, self.spec.id)
        dealer.connect(self.command_inbox)
        dealer = ZMQStream(dealer, self.loop)
        dealer.on_recv(self.on_outbox)
        return dealer

    def init_pub(self) -> ZMQStream:
        pub = self.context.socket(zmq.PUB)
        pub = pub.setsockopt_string(zmq.IDENTITY, self.spec.id)
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
        sub.connect(self.command_outbox)
        sub = ZMQStream(sub, self.loop)
        sub.on_recv(self.on_inbox)
        return sub

    def on_outbox(self, msg: list[bytes]) -> None:
        print(f"DEALER {self.spec.id}")
        print(msg)

    def on_inbox(self, msg: list[bytes]) -> None:
        msg = Message.from_bytes(msg)

        print(f"INBOX {self.spec.id}")
        print(msg)
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
                self._sub.connect(msg.value["nodes"][node_id]["outbox"])
        self._nodes = msg.value["nodes"]

    def on_event(self, msg: EventMsg) -> None:
        events = msg.value
        to_add = [e for e in events if (e["node_id"], e["signal"]) in self.depends]
        for event in to_add:
            self.store.add_value(
                self._nodes[event["node_id"]]["signals"], event["value"], event["node_id"]
            )

        if to_add:
            # notify that new events that may be ready to process have been received.
            self._event_block.set()

        for event in events:
            if (event["node_id"], event["signal"]) in self.depends:
                self.store.add_value(
                    self._nodes[event["node_id"]]["signals"], event["value"], event["node_id"]
                )
                added_any = True

        if added_any:

            self._event_block.set()
