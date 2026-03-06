import asyncio
import concurrent.futures
import contextlib
import multiprocessing as mp
import os
import signal
import traceback
import uuid
from collections import deque
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from functools import cached_property, partial
from itertools import count
from types import FrameType
from typing import Any, cast

import zmq
from pydantic import ValidationError

from noob import init_logger
from noob.asset import AssetScope, AssetSpecification
from noob.config import config
from noob.event import Event, MetaEvent
from noob.exceptions import AlreadyDoneError, EpochCompletedError
from noob.input import InputCollection
from noob.network.loop import EventloopMixin
from noob.network.message import (
    AnnounceMsg,
    DeinitMsg,
    EpochEndedMsg,
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
from noob.node import Edge, Node, NodeSpecification, Signal
from noob.scheduler import Scheduler
from noob.state import State
from noob.store import EventStore
from noob.types import Epoch
from noob.utils import iscoroutinefunction_partial


class NodeRunner(EventloopMixin):
    """
    Runner for a single node

    - DEALER to communicate with command inbox
    - PUB (outbox) to publish events
    - SUB (inbox) to subscribe to events from other nodes.

    .. admonition:: Assets & Mutation.
        :class: important

        Assets behave slightly differently in the ZMQRunner and NodeRunner
        than they do in the other, local runners.
        :ref:`assets` are objects that are shraed within some :class:`.AssetScope` -
        in turn implying that mutations of that object will also be shared.
        However since each Node runner runs in a separate process,
        shared asset state must work differently.

        * :attr:`~.AssetScope.node` -scoped assets work the same as other runners.
        * :attr:`~.AssetScope.process` -scoped assets are instantiated
          **for every root** :class:`.Epoch` by the nodes
          in the first topological generation that depends on the asset,
          and then the asset is forwarded on as an :class:`.Event` to subsequent generations.
          The asset **will** be initialized multiple times if there are multiple
          nodes in the same topological epoch that depend on the asset.
          The only guarantee **which** version of the asset downstream nodes
          will receive is the topology of the graph:
          e.g. if nodes ``a, b, c, d`` all depend on an asset,
          and ``a, b`` run in the first generation and ``c, d`` run in the second,
          ``c, d`` could receive the version of the asset emitted by **either**
          ``a, b``.
        * :attr:`~.AssetScope.runner` -scoped assets are instantiated
          **once per runner initialization** by the nodes in the first topological epoch
          that depends on the asset. The same caveats as ``process``-scoped assets apply.
          Additionally, unless an asset uses :attr:`~.Asset.depends` ,
          mutations in later topo generations are not propagated to subsequent epochs.
          e.g. if nodes ``a, b`` depend on an asset and run in the first and second
          generation, and they both increment the asset by 1, the values will be

          * Epoch 0: ``{a: 1, b: 2}``;
          * Epoch 1: ``{a: 2, b: 3}``;

          Rather than ``a`` receiving ``2`` after the mutation from ``b``.

        As is the case with all runners,
        if you want to mutate an asset with multiple nodes in a tube,
        the safest and most predictable way to do that is to put the mutation order
        in the topology of the graph:
        only depend on an asset directly with the first node that mutates it,
        and pass the asset through the graph by emitting it as an event.
        To persist data between epochs,
        use :attr:`~.Asset.depends` to store the value after the last desired mutation.
        See :ref:`persisting-data`

    """

    def __init__(
        self,
        spec: NodeSpecification,
        runner_id: str,
        command_outbox: str,
        command_router: str,
        input_collection: InputCollection,
        asset_specs: dict[str, AssetSpecification] | None = None,
        asset_generations: dict[str, list[tuple[str, ...]]] | None = None,
        edges: list[Edge] | None = None,
        protocol: str = "ipc",
    ):
        self.spec = spec
        self.runner_id = runner_id
        self.input_collection = input_collection
        self.command_outbox = command_outbox
        self.command_router = command_router
        self.protocol = protocol
        self.store = EventStore()
        self.asset_specs = asset_specs or {}
        self.asset_generations = asset_generations or {}
        self.edges = edges or []
        self.state: State = None  # type: ignore[assignment]
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

    @cached_property
    def subscribes_to(self) -> set[str]:
        """
        The set of node IDs that we should subscribe to.
        All the nodes we depend on,
        and all those that we listen for mutated assets from
        """
        if self.depends is None:
            return set()
        subscribes = set(d[0] for d in self.depends) | self.scheduler.upstream_nodes(self.spec.id)
        for senders in self.receives_assets_from.values():
            subscribes.update(senders)
        subscribes.difference_update({"assets", "input"})
        return subscribes

    @cached_property
    def inits_assets(self) -> set[str]:
        """
        The set of assets that we should initialize since
        * They are node scoped and we depend on them, or
        * We are in the first topo generation that uses them
        """
        self._node = cast(Node, self._node)
        inits: set[str] = set()
        if not self.depends:
            return inits
        for asset, generations in self.asset_generations.items():
            if (
                self.asset_specs[asset].scope == AssetScope.node
                and ("assets", asset) in self.depends
            ) or (generations and self._node.id in generations[0]):
                inits.add(asset)
        return inits

    @cached_property
    def publishes_assets(self) -> set[str]:
        """
        The set of assets that we publish for downstream nodes to consume
        """
        self._node = cast(Node, self._node)
        publishes = set()
        for asset, generations in self.asset_generations.items():
            spec = self.asset_specs[asset]
            if spec.scope == AssetScope.node:
                continue
            if len(generations) > 1 and any(self._node.id in gen for gen in generations[:-1]):
                publishes.add(asset)
        return publishes

    @cached_property
    def receives_assets_from(self) -> dict[str, set[str]]:
        """
        The map of assets that we must receive from other nodes
        (since we are not in the first topo generation that uses them)
        to the set of nodes that we may receive them from.
        """
        self._node = cast(Node, self._node)
        asset_sources = {}
        for asset, generations in self.asset_generations.items():
            spec = self.asset_specs[asset]
            if spec.scope == AssetScope.node:
                continue
            in_generation = -1
            for i, generation in enumerate(generations):
                if self._node.id in generation:
                    in_generation = i
                    break

            if in_generation > 0:
                # subscribe to all the nodes that mutate the asset before us
                asset_sources[asset] = set(generations[in_generation - 1])
            elif in_generation == 0 and spec.scope == AssetScope.runner and spec.depends:
                asset_sources[asset] = {spec.depends.split(".")[0]}
        return asset_sources

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
                        await self.publish_events(events, epoch)
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

                self.state.init(AssetScope.node, self._node.edges)
                self.state.init(AssetScope.process, self._node.edges)
                assets = self.state.collect(self._node.edges)
                inputs |= assets if assets else {}

                if self._node.injections.get("epoch"):
                    inputs[self._node.injections["epoch"]] = ready["epoch"]

                args, kwargs = self.store.split_args_kwargs(inputs)
                yield args, kwargs, ready["epoch"]
                self.state.deinit(AssetScope.node, self._node.edges)

            if self.scheduler.node_is_done(
                self.spec.id, epoch
            ) and not self.scheduler.epoch_completed(epoch):
                self.scheduler.end_epoch(epoch)

            if self.scheduler.epoch_completed(epoch):
                self.logger.debug("Epoch completed: %s", epoch)
                self.state.deinit(AssetScope.process, self._node.edges)
                self.store.clear(epoch)
                epoch = None

    async def publish_events(self, events: list[Event], epoch: Epoch) -> None:
        # re-emit any events that we initialized or received:
        for asset in self.publishes_assets:
            if asset in self.state.specs:
                asset_val = self.state.assets[asset].obj
                asset_evt = Event(
                    node_id="assets",
                    signal=asset,
                    epoch=epoch,
                    id=uuid.uuid4().int,
                    timestamp=datetime.now(UTC),
                    value=asset_val,
                )

            else:
                asset_evt = self.store.get("assets", asset, epoch)
            events.append(asset_evt)

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

        if self.state is not None:
            self.state.deinit(AssetScope.runner)
            self.state.deinit(AssetScope.process)

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
        self.state = State.from_specification(
            specs={asset: self.asset_specs[asset] for asset in self.inits_assets}
        )

        # we can not receive edges from the full tube,
        # so we at least listen to the edges we know about from the node spec
        # and supplement if we can
        edges = [
            e
            for e in self._node.edges
            if e.source_node != "assets" or e.source_signal in self.receives_assets_from
        ]
        edges += [e for e in self.edges if e.source_node != "assets"]
        self.scheduler = Scheduler(
            nodes={self.spec.id: self.spec},
            edges=edges,
            _logger=init_logger(f"noob.scheduler.{self.spec.id}"),
        )
        self.state.init(AssetScope.runner, self._node.edges)
        async with self._ready_condition:
            ep = self.scheduler.add_epoch()
            if self.state.dependencies and len(self.state.dependencies) == len(
                self.receives_assets_from
            ):
                self.scheduler.done(ep, "assets")
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
        elif message.type_ == MessageType.epoch_ended:
            message = cast(EpochEndedMsg, message)
            await self.on_epoch_ended(message)
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

        if self.subscribes_to:
            self.logger.debug("Should subscribe to %s", self.subscribes_to)
        for node_id in msg.value["nodes"]:
            if node_id in self.subscribes_to and node_id not in self._nodes:
                # TODO: a way to check if we're already connected, without storing it locally?
                outbox = msg.value["nodes"][node_id]["outbox"]
                self.logger.debug("Subscribing to %s at %s", node_id, outbox)
                self.sockets["inbox"].connect(outbox)
                self.logger.debug("Subscribed to %s at %s", node_id, outbox)
        self._nodes = msg.value["nodes"]
        if set(self._nodes) >= self.subscribes_to and self.status == NodeStatus.waiting:
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

        to_update = [e for e in events if e["node_id"] != "assets"]
        for event in events:
            if event["node_id"] == "meta":
                continue
            event = cast(Event, event)
            self.store.add(event)

        async with self._ready_condition:
            # we might have already been told the epoch was completed,
            # so this information is redundant.
            with contextlib.suppress(EpochCompletedError):
                self.scheduler.update(to_update)
            self._handle_assets(msg)
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

    async def on_epoch_ended(self, msg: EpochEndedMsg) -> None:
        """
        Command node has told us that an epoch has ended.
        Under most conditions, we don't need to be told this explicitly,
        but when we are a node that stores an asset from a later node,
        we need to mark the asset as done manually.
        """
        if not self.scheduler.epoch_completed(msg.value):
            async with self._ready_condition:
                self.scheduler.end_epoch(msg.value)
                self._ready_condition.notify_all()

        if self.state.dependencies and not self._assets_done(msg.value + 1):
            with contextlib.suppress(AlreadyDoneError):
                async with self._ready_condition:
                    self.scheduler.done(epoch=msg.value + 1, node_id="assets")
                    self._ready_condition.notify_all()

    async def await_node(self, epoch: Epoch) -> list[MetaEvent]:
        """
        Block until a node is ready

        Args:
            epoch (Epoch, None): if `int` , wait until the node is ready in the given epoch,
                otherwise wait until the node is ready in any epoch

        """
        async with self._ready_condition:
            await self._ready_condition.wait_for(
                lambda: self.scheduler.node_is_ready(self.spec.id, epoch)
                or self.scheduler.node_is_done(self.spec.id, epoch)
            )
            if self.scheduler.node_is_done(self.spec.id, epoch):
                return []

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

    def _handle_assets(self, msg: EventMsg) -> None:
        """
        Mark assets as done in the scheduler (according to _assets_done)
        and update the assets stored in the `state` collection from asset dependencies.
        """

        if not self.receives_assets_from:
            return
        epochs = set(e["epoch"] for e in msg.value)
        if len(epochs) > 1:
            self.logger.warning(
                "Received multiple epochs in a single event message, asset logic may be inaccurate"
            )
        epoch = epochs.pop()
        if self._assets_done(epoch):
            with contextlib.suppress(AlreadyDoneError):
                self.scheduler.done(epoch, "assets")
        if msg.node_id in self.state.dependencies:
            self.state.update(msg.value)
            if self._assets_done(epoch + 1):
                self.scheduler.done(epoch + 1, "assets")

    def _assets_done(self, epoch: Epoch) -> bool:
        """Whether we've received all the events we expect to have received for the given epoch"""
        # FIXME: Dependencies in the topo graph should really be node.signal pairs
        # See issue #152
        if not self.receives_assets_from:
            # we don't need to wait on any assets, so they're not in our topo sorter at all.
            return False

        for asset in self.receives_assets_from:
            try:
                self.store.get("assets", asset, epoch)
            except KeyError:
                if self.asset_specs[asset].depends is not None:
                    # no idea why mypy can't tell `depends` is a string here
                    node_id, signal = self.asset_specs[asset].depends.split(".")  # type: ignore[union-attr]
                    try:
                        self.store.get(node_id, signal, epoch - 1)
                    except KeyError:
                        return False
                else:
                    return False
        return True
