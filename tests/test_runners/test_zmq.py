from asyncio import sleep
from time import time
from typing import cast

import pytest

from noob import Tube
from noob.event import Event
from noob.network.message import EventMsg, IdentifyMsg, IdentifyValue, Message, NodeStatus
from noob.runner.zmq import ZMQRunner

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

    runner = ZMQRunner(tube=tube)
    with runner:
        runner.command.add_callback("inbox", _event_cb)
        # skip first epoch
        runner.command.process(1, input={"multiply": 3})
        # start = time()
        # while len(events) < 1 and time() - start < 1:
        #     await sleep(0.1)
        # # we should have only received an event from the stateless count source
        # assert len(events) == 1
        # assert events[0]["node_id"] == "c"
        # since the generator takes no input,
        # even though we forced the process calls to be out of order,
        # it should emit an event with epoch 0 because the first thing out of the generator
        # is, by definition, epoch 0.
        # assert events[0]["epoch"] == 0
        # just for good measure, skip another
        runner.command.process(2, input={"multiply": 7})
        # start = time()
        # while len(events) < 2 and time() - start < 1:
        #     await sleep(0.1)
        # assert len(events) == 2
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

    runner = ZMQRunner(tube=tube)
    with runner:
        runner.command.add_callback("inbox", _event_cb)
        # skip first epoch
        runner.command.process(1, input={"multiply": 3})
        # should have received events from all except d,
        # which has not yet received the epoch 0 input to match with the epoch 0
        # event from the count source
        # start = time()
        # while len(events) < 3 and time() - start < 1:
        #     await sleep(0.1)
        # assert len(events) == 3

        # here we should get an overlap between the epoch 1 input
        # and the epoch 1 event from count source
        # so we get 4 events now
        runner.command.process(2, input={"multiply": 7})
        # start = time()
        # while len(events) < 7 and time() - start < 1:
        #     await sleep(0.1)
        # assert len(events) == 7
        # then when we send epoch 0, we should get all of them
        runner.command.process(0, input={"multiply": 11})
        start = time()
        while len(events) < 12 and time() - start < 1:
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
