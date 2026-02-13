from datetime import datetime

from pydantic import BaseModel

from noob.event import Event
from noob.network.message import EventMsg
from noob.types import Epoch


def test_epoch_contains():
    """
    Contains (based on __eq__) still works for epochs when used as dict keys
    """
    epochs = [Epoch(0), Epoch(1)]
    assert epochs[0] in epochs
    assert Epoch(0) == epochs[0]
    assert Epoch(0) in epochs


def test_epoch_roundtrips():
    """
    Epoch should survive a roundtrip serialization in a Message
    """
    epoch = Epoch(0)
    msg = EventMsg(
        node_id="test",
        value=[
            Event(
                id=0,
                timestamp=datetime.now(),
                node_id="test",
                signal="test",
                epoch=epoch,
                value=None,
            )
        ],
    )
    roundtrip = EventMsg.from_bytes([b"a", msg.to_bytes()])
    assert roundtrip.value[0]["epoch"] == epoch
    assert isinstance(roundtrip.value[0]["epoch"], Epoch)


def test_pydantic_coerces_epoch_tuples():
    """
    Pydantic casts tuples to epochs
    """

    class MyModel(BaseModel):
        epoch: Epoch

    model = MyModel(epoch=(("tube", 0),))
    assert isinstance(model.epoch, Epoch)
