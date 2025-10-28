"""
- Central command pub/sub
- each sub-runner has its own set of sockets for publishing and consuming events
- use the `node_id.signal` etc. as basically a feed address
"""

import multiprocessing as mp
import threading
from collections.abc import Callable
from enum import StrEnum
from itertools import count
from typing import Any, TypedDict

import zmq
from pydantic import BaseModel, Field
from zmq.eventloop.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream

from noob.event import Event
from noob.input import InputCollection
from noob.node import Node, NodeSpecification, Signal, Slot
from noob.store import EventStore


class NodeAnnounce(TypedDict):
    node_id: str
    outbox: str
    signals: list[Signal] | None
    slots: list[Slot] | None


class AnnounceMessage(TypedDict):
    """Command node 'announces' identities of other peers and the events they emit"""

    inbox: str
    nodes: dict[str, NodeAnnounce]


class Callbacks(TypedDict, total=False):
    on_inbox: Callable[[Event], Any]
    on_command: Callable[[Event], Any]


class MessageType(StrEnum):
    announce = "announce"
    start = "start"
    stop = "stop"
    event = "event"


class Message(BaseModel):
    type_: MessageType = Field(..., alias="type")
    node_id: str
    value: Event | AnnounceMessage | str | None = None

    @classmethod
    def from_bytes(cls, msg: list[bytes]) -> "Message":
        raise NotImplementedError()

    def to_bytes(self) -> list[bytes]:
        raise NotImplementedError()


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
        self._nodes: dict[str, NodeAnnounce] = {}

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

    def on_inbox(self, msg: list[bytes]) -> None:
        print("INBOX")
        print(msg)


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
        if callbacks is None:
            self.callbacks = Callbacks()
        else:
            self.callbacks = callbacks

        self._dealer = None
        self._pub = None
        self._sub = None
        self._node: Node | None = None
        self._counter = count()
        self._process_quitting = mp.Event()

    @property
    def pub_address(self) -> str:
        if self.protocol == "ipc":
            return f"{self.protocol}://tmp/noob/{self.runner_id}/nodes/{self.spec.id}/outbox"
        else:
            raise NotImplementedError()

    @classmethod
    def run(cls, spec: NodeSpecification, **kwargs: Any) -> None:
        """
        Target for multiprocessing.run,
        init the class and start it!
        """
        runner = NodeRunner(spec=spec, **kwargs)
        runner.start_sockets()
        runner.init_node()

    def start_sockets(self) -> None:
        self.init_sockets()
        self.start_loop()

    def init_node(self) -> None:
        self._node = Node.from_specification(self.spec, self.input_collection)
        self._node.init()

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

    def init_pub(self) -> zmq.Socket:
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

    def init_sub(self) -> ZMQStream:
        """
        Init the subscriber, but don't attempt to subscribe to anything but the command yet!
        we do that when we get node Announces
        """
        sub = self.context.socket(zmq.SUB)
        sub.setsockopt_string(zmq.IDENTITY, self.spec.id)
        sub = ZMQStream(sub, self.loop)
        sub.on_recv(self.on_inbox)
        return sub

    def on_outbox(self, msg: list[bytes]) -> None:
        print(f"DEALER {self.spec.id}")
        print(msg)

    def on_inbox(self, msg: list[bytes]) -> None:
        print(f"INBOX {self.spec.id}")
        print(msg)
