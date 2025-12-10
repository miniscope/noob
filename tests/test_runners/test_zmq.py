import pytest

from noob import Tube
from noob.network.message import IdentifyMsg, IdentifyValue, NodeStatus
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
