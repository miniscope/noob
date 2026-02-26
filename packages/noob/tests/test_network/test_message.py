import json
from datetime import UTC, datetime

from noob.const import META_SIGNAL
from noob.event import Event, MetaSignal
from noob.network.message import EventMsg
from noob.types import Epoch


class PickleClass:
    def __init__(self, x: int):
        self.x = x

    def __eq__(self, other):
        return isinstance(other, PickleClass) and other.x == self.x


def test_jsonable_event():
    """Events that can be dumped to json are dumped to JSON without pickling"""
    msg = EventMsg(
        type_="event",
        node_id="example",
        value=[
            Event(
                id=0,
                timestamp=datetime.now(UTC),
                node_id="example",
                signal="x",
                epoch=Epoch(0),
                value=100,
            ),
            Event(
                id=1,
                timestamp=datetime.now(UTC),
                node_id="example",
                signal="y",
                epoch=Epoch(0),
                value="hello",
            ),
        ],
    )
    dumped = msg.model_dump_json()
    as_json = json.loads(dumped)
    for evt in as_json["value"]:
        assert isinstance(evt, dict)
        assert not isinstance(evt, str) or not evt.startswith("pck_")

    z = EventMsg.model_validate_json(dumped)
    assert z == msg


def test_pickleable_event():
    """Events that can't be dumped to JSON have their value field pickled"""
    msg = EventMsg(
        node_id="example",
        value=[
            Event(
                id=0,
                timestamp=datetime.now(UTC),
                node_id="example",
                signal="x",
                epoch=Epoch(0),
                value=PickleClass(x=100),
            ),
        ],
    )
    dumped = msg.model_dump_json()
    as_json = json.loads(dumped)
    assert as_json["value"][0]["value"].startswith("pck_")

    z = EventMsg.model_validate_json(dumped)
    assert msg == z
    assert isinstance(z.value[0]["value"], PickleClass)


def test_roundtrip_noevent():
    """NoEvents should roundtrip to and from a string"""
    msg = EventMsg(
        node_id="example",
        value=[
            Event(
                id=0,
                timestamp=datetime.now(UTC),
                node_id="example",
                signal=META_SIGNAL,
                epoch=Epoch(0),
                value=MetaSignal.NoEvent,
            ),
        ],
    )
    dumped = msg.model_dump_json()
    as_json = json.loads(dumped)
    assert as_json["value"][0]["value"] == "NoEvent"

    z = EventMsg.model_validate_json(dumped)
    assert msg == z
    assert isinstance(z.value[0]["value"], MetaSignal)
    assert z.value[0]["value"] is MetaSignal.NoEvent
