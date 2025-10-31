import pytest

from noob import Tube
from noob.runner.zmq import IdentifyMsg, NodeIdentifyMessage

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
        value=NodeIdentifyMessage(
            node_id=node.spec.id,
            outbox="ipc:///abc/123",
            signals=[s.name for s in node.signals],
            slots=[s for s in node.slots],
        ),
    )
    as_bytes = msg.to_bytes()
    recreated = IdentifyMsg.from_bytes(as_bytes)
    assert msg == recreated
