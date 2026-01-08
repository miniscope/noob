import asyncio
import threading

try:
    import zmq
    from tornado.ioloop import IOLoop
except ImportError as e:
    raise ImportError(
        "Attempted to import zmq runner, but zmq deps are not installed. install with `noob[zmq]`",
    ) from e


class EventloopMixin:
    """
    Provide an eventloop in a separate thread to an inheriting class.
    Any eventloop that is running in the current context is not used
    because the inheriting classes are presumed to operate mostly synchronously for now,
    pending a refactor to all async networking classes.
    """

    def __init__(self):
        self._context = None
        self._loop = None
        self._quitting = asyncio.Event()
        self._thread: threading.Thread | None = None

    @property
    def context(self) -> zmq.Context:
        if self._context is None:
            self._context = zmq.Context.instance()
        return self._context

    @property
    def loop(self) -> IOLoop:
        # To ensure that the loop is always created in the spawned thread,
        # we don't create it here (since this property could be accessed elsewhere)
        # and throw to protect that.
        if self._loop is None:
            raise RuntimeError("Loop is not running")
        return self._loop

    def start_loop(self) -> None:
        if self._thread is not None:
            raise RuntimeWarning("Node already started")

        self._quitting.clear()

        _ready = threading.Event()

        def _signal_ready() -> None:
            _ready.set()

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = IOLoop.current()
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
                self.logger.debug("Eventloop stopped")
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
        self.loop.add_callback(self.loop.stop)
