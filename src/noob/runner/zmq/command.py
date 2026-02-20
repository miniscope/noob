import asyncio
import threading
from time import time
from typing import cast

import zmq

from noob import init_logger
from noob.config import config
from noob.network.loop import EventloopMixin
from noob.network.message import (
    AnnounceMsg,
    AnnounceValue,
    DeinitMsg,
    IdentifyMsg,
    IdentifyValue,
    Message,
    MessageType,
    PingMsg,
    ProcessMsg,
    StartMsg,
    StatusMsg,
    StopMsg,
)
from noob.types import Epoch, NodeID


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

    def process(self, epoch: Epoch, input: dict | None = None) -> None:
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
