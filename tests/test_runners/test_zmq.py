from asyncio import sleep
from datetime import UTC, datetime
from time import time
from typing import cast

import pytest

from noob import Tube
from noob.event import Event
from noob.input import InputCollection
from noob.network.message import EventMsg, IdentifyMsg, IdentifyValue, Message, NodeStatus
from noob.node.spec import NodeSpecification
from noob.runner.zmq import NodeRunner, ZMQRunner

pytestmark = pytest.mark.zmq_runner


@pytest.mark.xfail
def test_message_union():
    """
    Message union correctly casts messages depending on their `type`,
    including casting unrecognized messages to the parent class
    """
    raise NotImplementedError()


@pytest.mark.xfail
def test_nonjsonable_event():
    """
    We can serialize/deserialize values that require pickling
    """
    raise NotImplementedError()


def test_message_roundtrip():
    tube = Tube.from_specification("testing-basic")
    node = tube.nodes["b"]

    msg = IdentifyMsg(
        node_id=node.spec.id,
        value=IdentifyValue(
            node_id=node.spec.id,
            status=NodeStatus.ready,
            outbox="ipc:///abc/123",
            signals=[s.name for s in node.signals],
            slots=[s for s in node.slots],
        ),
    )
    as_bytes = msg.to_bytes()
    recreated = IdentifyMsg.from_bytes([as_bytes])
    assert msg == recreated


def test_error_reporting():
    """When a zmq runner node has an error, it sends it back to the command node"""
    tube = Tube.from_specification("testing-error")
    runner = ZMQRunner(tube)
    with pytest.raises(ValueError) as exc_info:
        runner.process()

    assert exc_info.type is ValueError
    # original error message
    assert str(exc_info.value) == "This node just emits errors"
    # additional information in the notes
    note = exc_info.value.__notes__[0]
    # notice that this error was raised in another process
    assert "Error re-raised from node runner process" in note
    # the location of the original exception
    assert "testing/nodes.py" in note


@pytest.mark.asyncio
async def test_statefulness():
    """
    Nodes that are marked as stateful should process events in order,
    even when they are received out of order.
    """
    tube = Tube.from_specification("testing-stateful")

    events: list[Event] = []

    def _event_cb(event: Message) -> None:
        nonlocal events
        event = cast(EventMsg, event)
        events.extend(event.value)

    runner = ZMQRunner(tube=tube, autoclear_store=False)
    with runner:
        runner.command.add_callback("inbox", _event_cb)
        # skip first epoch
        runner.command.process(1, input={"multiply": 3})

        start = time()
        while len(events) < 1 and time() - start < 1:
            await sleep(0.1)
        # # we should have only received an event from the stateless count source
        assert len(events) == 1
        assert events[0]["node_id"] == "c"
        # since the generator takes no input,
        # even though we forced the process calls to be out of order,
        # it should emit an event with epoch 0 because the first thing out of the generator
        # is, by definition, epoch 0.
        assert events[0]["epoch"] == 0

        # just for good measure, skip another
        runner.command.process(2, input={"multiply": 7})
        start = time()
        while len(events) < 2 and time() - start < 1:
            await sleep(0.1)
        assert len(events) == 2

        # then when we send epoch 0, we should get all of them
        runner.command.process(0, input={"multiply": 11})
        start = time()
        while len(events) < 12 and time() - start < 1:
            await sleep(0.1)

    # should have gotten 3 events from 4 nodes, so 12 total events
    assert len(events) == 12
    assert runner.store.events[0]["a"]["value"][0]["value"] == 11
    assert runner.store.events[0]["b"]["value"][0]["value"] == 11
    assert runner.store.events[0]["d"]["value"][0]["value"] == 11

    assert runner.store.events[1]["a"]["value"][0]["value"] == 3 * 2
    assert runner.store.events[1]["b"]["value"][0]["value"] == 3
    assert runner.store.events[1]["d"]["value"][0]["value"] == 3 * 2 * 2

    assert runner.store.events[2]["a"]["value"][0]["value"] == 7 * 3
    assert runner.store.events[2]["b"]["value"][0]["value"] == 7
    assert runner.store.events[2]["d"]["value"][0]["value"] == 7 * 3 * 3


@pytest.mark.asyncio
async def test_statelessness():
    """
    Nodes that are marked as stateless should process as soon as they are ready in any epoch,
    but they should still match events according to their epoch -
    i.e. epoch 0 inputs should still be labeled as being in epoch 0,
    even if they are run after epoch 1 events.
    """
    tube = Tube.from_specification("testing-stateless")

    events: list[Event] = []

    def _event_cb(event: Message) -> None:
        nonlocal events
        event = cast(EventMsg, event)
        events.extend(event.value)

    runner = ZMQRunner(tube=tube, autoclear_store=False)
    for i in range(3):
        runner.tube.scheduler.add_epoch(i)
        runner.tube.scheduler.done(i, "input")
    with runner:
        runner.command.add_callback("inbox", _event_cb)
        # skip first epoch
        runner.command.process(1, input={"multiply": 3})
        # should have received events from all except d,
        # which has not yet received the epoch 0 input to match with the epoch 0
        # event from the count source
        start = time()
        while len(events) < 3 and time() - start < 1:
            await sleep(0.1)
        # assert len(events) == 3

        # here we should get an overlap between the epoch 1 input
        # and the epoch 1 event from count source
        # so we get 4 events now
        runner.command.process(2, input={"multiply": 7})
        start = time()
        while len(events) < 7 and time() - start < 1:
            await sleep(0.1)
        # assert len(events) == 7

        # then when we send epoch 0, we should get all of them
        runner.command.process(0, input={"multiply": 11})
        start = time()
        while len(events) < 12 and time() - start < 5:
            await sleep(0.1)

    # should have gotten 3 events from 4 nodes, so 12 total events
    assert len(events) == 12
    assert runner.store.events[0]["a"]["value"][0]["value"] == 11 * 3
    assert runner.store.events[0]["b"]["value"][0]["value"] == 11
    # node d runs epoch 1 first, so here it should be
    # 1 (from count source) * 2 (from internal state) * 11 (input)
    assert runner.store.events[0]["d"]["value"][0]["value"] == 11 * 2

    assert runner.store.events[1]["a"]["value"][0]["value"] == 3
    assert runner.store.events[1]["b"]["value"][0]["value"] == 3
    # 2 (from count source) * 1 (from internal state) * 3
    assert runner.store.events[1]["d"]["value"][0]["value"] == 3 * 2 * 1

    assert runner.store.events[2]["a"]["value"][0]["value"] == 7 * 2
    assert runner.store.events[2]["b"]["value"][0]["value"] == 7
    assert runner.store.events[2]["d"]["value"][0]["value"] == 7 * 3 * 3


@pytest.mark.asyncio
async def test_run_freeruns():
    """
    When `run` is called, the zmq runner should "freerun" -
    allow nodes to execute as quickly as they can, whenever their deps are satisfied.
    """
    tube = Tube.from_specification("testing-long-add")
    runner = ZMQRunner(tube, autoclear_store=False)

    # events = []
    #
    # def _event_cb(event: Message) -> None:
    #     nonlocal events
    #     events.append(event)

    with runner:
        runner.run()
        assert runner.running
        await sleep(0.5)
        runner.stop()

    # main thing we're testing here is whether we freerun -
    # i.e. that nodes don't wait for an epoch to complete before running, if they're ready.
    # so we should have way more events from the source node than from the long_add nodes,
    # which sleep and take a long time on purpose.
    # since there are way more long_add nodes than the single count node,
    # if we ran epoch by epoch, we would expect there to be more non-count events.
    count_events = []
    non_count_events = []
    for event in runner.store.iter():
        if event["node_id"] == "count":
            count_events.append(event)
        else:
            non_count_events.append(event)

    assert len(count_events) > 0
    assert len(non_count_events) > 0
    assert len(count_events) > len(non_count_events)
    # we should theoretically get only 2 epochs from the long add nodes,
    # but allow up to 5 in the case of network latency, this is not that important
    assert len(set(e["epoch"] for e in non_count_events)) < 5
    # it should be in the hundreds, but all we care about is that it's more than 1 greater
    # 10 is a good number.
    assert len(set(e["epoch"] for e in count_events)) > 10


@pytest.mark.asyncio
async def test_start_stop():
    """
    The runner can be started and stopped without deinitializing
    """
    tube = Tube.from_specification("testing-count-init")
    events: list[Event] = []
    router_events: list[Message] = []

    def _event_cb(event: Message) -> None:
        nonlocal events
        event = cast(EventMsg, event)
        events.extend(event.value)

    def _router_cb(msg: Message) -> None:
        nonlocal router_events
        router_events.append(msg)

    runner = ZMQRunner(tube=tube)
    with runner:
        runner.command.add_callback("inbox", _event_cb)
        runner.command.add_callback("router", _router_cb)
        runner.run()
        await sleep(0.1)
        runner.stop()
        await sleep(0.1)
        runner.run()
        await sleep(0.2)

    # sometimes we get the last stop event,
    # sometimes we dont - we don't wait for it
    assert 4 >= len(router_events) >= 3
    assert router_events[1].value == "stopped"
    first_events = [e for e in events if e["timestamp"] < router_events[1].timestamp]
    stopped_events = [
        e
        for e in events
        if e["timestamp"] > router_events[1].timestamp
        and e["timestamp"] < router_events[2].timestamp
    ]
    end_events = [e for e in events if e["timestamp"] > router_events[2].timestamp]

    # with the node's sleep, there are time for ~10 runs if there was no latency,
    # so say we should at least do 5 (10 events)
    assert len(first_events) > 10
    # there can be one additional run of the node after the stopped message is sent
    # if the node is already running when the stop message is received.
    # (two events, because the node emits two signals)
    assert len(stopped_events) <= 2
    assert len(end_events) > 10

    # even though we stopped and started, the nodes should have stated initialized
    # (and not been deinit'd and reinit'd)
    inits = [e for e in events if e["signal"] == "inits"]
    deinits = [e for e in events if e["signal"] == "deinits"]
    assert len(inits) > 0
    assert len(deinits) > 0
    assert all(e["value"] == 1 for e in inits)
    assert all(e["value"] == 0 for e in deinits)


def test_iter_gather(mocker):
    """
    itering over gather should heuristically request more iterations as we go
    """
    tube = Tube.from_specification("testing-gather-n")
    runner = ZMQRunner(tube=tube)
    # the gather_n pipeline only returns every 5 epochs.
    # so we are going to request 11 return values, which should require 55 epochs
    # after we run 11 epochs, (after returning two results)
    # we should notice that we haven't returned the correct amount yet
    # and request more.
    spy = mocker.spy(runner, "_request_more")
    stop_spy = mocker.spy(runner, "stop")
    results = []
    with runner:
        for result in runner.iter(11):
            results.append(result)

        # after exhausting the iterator, but before deinit, we should have called stop
        stop_spy.assert_called_once()

    # for cases like this with deterministic n_gather, only should need to call once.
    assert spy.call_count == 1
    spy.assert_called_once_with(n=11, current_iter=2, n_epochs=11)
    # we don't get an *exact* number of iters to run,
    # e.g. here we have chosen 11 since it isn't a multiple of 5,
    # and all we see is "we've run 11 times and gotten 2 results,"
    # so if 11 epochs yields 2 results, and we want 9 more,
    # then a reasonable amount of additional times to run might be
    # ceil((11/2)*9) = 50
    assert spy.spy_return == 50


def test_noderunner_stores_clear():
    """
    Stores in the noderunners should clear after they use the events from an epoch
    """
    spec = NodeSpecification(
        id="test_node",
        type="noob.testing.multiply",
        depends=[{"left": "other.left"}, {"right": "other.right"}],
    )
    runner = NodeRunner(
        spec=spec,
        runner_id="testing",
        command_outbox="/notreal/unused",
        command_router="/notreal/unused",
        input_collection=InputCollection(),
    )
    runner.init_node()

    # fake a few events
    events = []
    for i in range(3):

        msg = EventMsg(
            node_id="other",
            value=[
                Event(
                    id=i * 2,
                    timestamp=datetime.now(UTC),
                    signal="left",
                    value=i * 2,
                    node_id="other",
                    epoch=i,
                ),
                Event(
                    id=(i * 2) + 1,
                    timestamp=datetime.now(UTC),
                    signal="right",
                    value=(i * 2) + 1,
                    node_id="other",
                    epoch=i,
                ),
            ],
        )
        runner.on_event(msg)
        events.append(msg)

    runner._freerun.set()
    assert len(runner.store.events) == 3
    args, kwargs, epoch = next(runner.await_inputs())
    assert len(runner.store.events) == 2
    assert epoch not in runner.store.events


@pytest.fixture
def _zmq_runner_basic() -> ZMQRunner:
    tube = Tube.from_specification("testing-basic")
    runner = ZMQRunner(tube=tube)
    runner.init()
    yield runner
    runner.deinit()


def test_zmqrunner_stores_clear_process(_zmq_runner_basic):
    """
    ZMQRunner stores clear after returning values from process
    """
    runner = _zmq_runner_basic
    runner.process()
    assert len(runner.store.events) == 0


def test_zmqrunner_stores_clear_iter(_zmq_runner_basic):
    """
    ZMQRunner stores clear after returning values while iterating
    """
    runner = _zmq_runner_basic
    for i, _ in enumerate(runner.iter(5)):
        # we will receive more events from epochs that run ahead of our processing
        # but we should clear the epoch as we return it from iter
        assert i not in runner.store.events


@pytest.mark.asyncio
async def test_zmqrunner_stores_clear_freerun(_zmq_runner_basic):
    """
    ZMQRunner doesn't store events while freerunning.
    """
    runner = _zmq_runner_basic
    runner.run()
    await sleep(0.1)
    assert len(runner.store.events) == 0
