import threading

import zmq
from tornado.ioloop import IOLoop


class EventloopMixin:

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
            raise RuntimeWarning("Node already started")

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
