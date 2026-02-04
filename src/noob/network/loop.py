import asyncio
import sys
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any

try:
    import zmq
    from zmq.asyncio import Context, Poller, Socket
except ImportError as e:
    raise ImportError(
        "Attempted to import zmq runner, but zmq deps are not installed. install with `noob[zmq]`",
    ) from e

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

from noob.logging import init_logger
from noob.network.message import Message
from noob.utils import iscoroutinefunction_partial


class _CallbackDict(TypedDict):
    sync: list[Callable[[Message], Any]]
    asyncio: list[Callable[[Message], Coroutine]]


class EventloopMixin:
    """
    Mixin to provide common asyncio zmq scaffolding to networked classes.

    Inheriting classes should, in order

    * call the ``_init_loop`` method to create the eventloop, context, and poller
    * populate the private ``_sockets``  and ``_receivers`` dicts
    * await the ``_poll_sockets`` method, which polls indefinitely.

    Inheriting classes **must** ensure that ``_init_loop``
    is called in the thread it is intended to run in,
    and that thread must already have a running eventloop.
    asyncio eventloops (and most of asyncio) are **not** thread safe.

    To help avoid cross-threading issues, the :meth:`.context` , :meth:`.poller` , :meth:`.loop`
    properties do *not* automatically create the objects,
    raising a :class:`.RuntimeError` if they are accessed before ``_init_loop`` is called.
    """

    def __init__(self):
        self._context = None
        self._loop = None
        self._poller = None
        self._quitting = None
        self._sockets: dict[str, Socket] = {}
        """
        All sockets, mapped from some common name to the socket.
        The same key used here should be shared between _receivers and _callbacks
        """
        self._receivers: dict[str, Socket] = {}
        """Sockets that should be polled for incoming messages"""
        self._callbacks: dict[str, _CallbackDict] = defaultdict(
            lambda: _CallbackDict(sync=[], asyncio=[])
        )
        """Callbacks for each receiver socket"""
        if not hasattr(self, "logger") or self.logger is None:
            self.logger = init_logger("eventloop")

    @property
    def context(self) -> Context:
        if self._context is None:
            raise RuntimeError("Loop has not been initialized with _init_loop!")
        return self._context

    @property
    def poller(self) -> Poller:
        if self._poller is None:
            raise RuntimeError("Loop has not been initialized with _init_loop!")
        return self._poller

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("Loop has not been initialized with _init_loop!")
        return self._loop

    @property
    def sockets(self) -> dict[str, Socket]:
        return self._sockets

    def register_socket(self, name: str, socket: Socket, receiver: bool = False) -> None:
        """Register a socket, optionally declaring it as a receiver socket to poll"""
        if name in self._sockets:
            raise KeyError(f"Socket {name} already declared!")
        self._sockets[name] = socket
        if receiver:
            self._receivers[name] = socket
            self.poller.register(socket, zmq.POLLIN)

    def add_callback(
        self, socket: str, callback: Callable[[Message], Any] | Callable[[Message], Coroutine]
    ) -> None:
        """
        Add a callback to be called when the socket receives a message.
        Callbacks are called in the order in which they are added.
        """
        if socket not in self._receivers:
            raise KeyError(f"Socket {socket} does not exist or is not a receiving socket")
        if iscoroutinefunction_partial(callback):
            self._callbacks[socket]["asyncio"].append(callback)
        else:
            self._callbacks[socket]["sync"].append(callback)

    def clear_callbacks(self) -> None:
        self._callbacks = defaultdict(lambda: _CallbackDict(sync=[], asyncio=[]))

    def _init_loop(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._poller = Poller()
        self._context = Context.instance()
        self._quitting = asyncio.Event()

    def _stop_loop(self) -> None:
        if self._quitting is None:
            return
        self._quitting.set()

    async def _poll_receivers(self) -> None:
        while not self._quitting.is_set():
            # timeout to avoid hanging here when quitting
            events = await self.poller.poll(1)
            events = dict(events)
            for name, socket in self._receivers.items():
                if socket in events:
                    msg_bytes = await socket.recv_multipart()
                    try:
                        msg = Message.from_bytes(msg_bytes)
                    except Exception as e:
                        self.logger.exception(
                            "Exception decoding message for socket %s: %s,  %s", name, msg_bytes, e
                        )
                        continue
                    for acb in self._callbacks[name]["asyncio"]:
                        await acb(msg)
                    for cb in self._callbacks[name]["sync"]:
                        self.loop.run_in_executor(None, cb, msg)
