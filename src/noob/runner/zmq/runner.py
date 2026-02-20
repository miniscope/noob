import concurrent.futures
import math
import multiprocessing as mp
import threading
from collections.abc import Generator, MutableSequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from multiprocessing.synchronize import Event as EventType
from time import time
from typing import Any, cast, overload
from uuid import uuid4

from noob.event import Event, MetaEvent, MetaEventType, MetaSignal
from noob.exceptions import InputMissingError
from noob.input import InputScope
from noob.network.message import ErrorMsg, ErrorValue, EventMsg, Message, MessageType
from noob.node import Return
from noob.runner import TubeRunner
from noob.runner.zmq.command import CommandNode
from noob.runner.zmq.node import NodeRunner
from noob.store import EventStore
from noob.types import Epoch, NodeID, ReturnNodeType


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
    _current_epoch: Epoch = Epoch(0)
    _epoch_futures: dict[Epoch, concurrent.futures.Future] = field(default_factory=dict)

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

        epoch = self._current_epoch[0].epoch
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
                    self.await_epoch(Epoch(epoch))
                    ret = self.collect_return(Epoch(epoch))
                    epoch += 1
                    self._current_epoch = Epoch(epoch)
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
            self._current_epoch = self.await_epoch(Epoch(self._current_epoch[0].epoch + n))
            return None

    def stop(self) -> None:
        """
        Stop running the tube.
        """
        self.command = cast(CommandNode, self.command)
        self._ignore_events = False
        self.command.stop()
        self._running.clear()

    async def on_event(self, msg: Message) -> None:
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
                ready_epochs = self.tube.scheduler.get_ready(epoch, self._return_node.id)
                for ready in ready_epochs:
                    self._logger.debug("Marking return node ready in epoch %s", ready["epoch"])
                    ep_ended = self.tube.scheduler.expire(ready["epoch"], self._return_node.id)
                    if ep_ended is not None:
                        events.append(ep_ended)
            roots = set(e.root for e in epochs)
            for root in roots:
                if self.tube.scheduler.epoch_completed(root):
                    events.append(
                        MetaEvent(
                            id=uuid4().int,
                            timestamp=datetime.now(UTC),
                            node_id="meta",
                            signal=MetaEventType.EpochEnded,
                            value=root,
                            epoch=root,
                        )
                    )
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

    def collect_return(self, epoch: Epoch | None = None) -> Any:
        if epoch is None:
            raise ValueError("Must specify epoch in concurrent runners")
        if self._return_node is None:
            return None
        else:
            if self.tube.scheduler.subepochs[epoch]:
                epochs = sorted(self.tube.scheduler.subepochs[epoch]) + [epoch]
            else:
                epochs = [epoch]
            for ep in epochs:
                events = self.store.collect(
                    self._return_node.edges, ep, eventmap=self._return_node.injections.get("events")
                )
                if not events:
                    return MetaSignal.NoEvent
                args, kwargs = self.store.split_args_kwargs(events)
                self._return_node.process(*args, **kwargs)  # type: ignore[call-arg]
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

    def await_epoch(self, epoch: Epoch) -> Epoch:
        if self.tube.scheduler.epoch_completed(epoch):
            return epoch

        if epoch not in self._epoch_futures:
            self._epoch_futures[epoch] = concurrent.futures.Future()
        return self._epoch_futures[epoch].result()
