import pytest

from noob import Tube
from noob.network.message import IdentifyMsg, IdentifyValue
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
    with pytest.raises(ValueError, match="This node just emits errors"):
        runner.process()
