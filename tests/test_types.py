from datetime import datetime

import pytest
from pydantic import BaseModel

from noob.event import Event
from noob.network.message import EventMsg
from noob.types import Epoch, EpochSegment


def test_epoch_from_int():
    """
    Creating an epoch from an integer creates a "root" epoch with the implicit "tube" node
    """
    epoch = Epoch(0)
    assert len(epoch) == 1
    assert epoch[0].epoch == 0
    assert epoch[0].node_id == "tube"


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


@pytest.mark.parametrize("segment", [EpochSegment(node_id="sub", epoch=0), ("sub", 0)])
def test_epoch_truediv(segment):
    """
    Subepochs can be created by truediv
    """
    epoch = Epoch(0)
    subepoch = epoch / segment
    assert subepoch == Epoch(
        (EpochSegment(node_id="tube", epoch=0), EpochSegment(node_id="sub", epoch=0))
    )
