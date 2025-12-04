import base64
import json
import pickle
import sys
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated as A
from typing import Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    PlainSerializer,
    Tag,
    TypeAdapter,
)

from noob.event import Event

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class MessageType(StrEnum):
    announce = "announce"
    identify = "identify"
    process = "process"
    start = "start"
    stop = "stop"
    event = "event"


class Message(BaseModel):
    type_: MessageType = Field(..., alias="type")
    node_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    value: dict | str | None = None

    model_config = ConfigDict(use_enum_values=True, validate_by_alias=True, serialize_by_alias=True)

    @classmethod
    def from_bytes(cls, msg: list[bytes]) -> "Message":
        return MessageAdapter.validate_json(msg[-1].decode("utf-8"))

    def to_bytes(self) -> bytes:
        return self.model_dump_json().encode("utf-8")


class IdentifyValue(TypedDict):
    node_id: str
    outbox: str
    signals: list[str] | None
    slots: list[str] | None


class AnnounceValue(TypedDict):
    """Command node 'announces' identities of other peers and the events they emit"""

    inbox: str
    nodes: dict[str, IdentifyValue]


class AnnounceMsg(Message):
    type_: Literal[MessageType.announce] = Field(MessageType.announce, alias="type")
    value: AnnounceValue


class IdentifyMsg(Message):
    type_: Literal[MessageType.identify] = Field(MessageType.identify, alias="type")
    value: IdentifyValue


class ProcessMsg(Message):
    """Process a single iteration of the graph"""

    type_: Literal[MessageType.process] = Field(MessageType.process, alias="type")
    value: None = None


class StartMsg(Message):
    type_: Literal[MessageType.start] = Field(MessageType.start, alias="type")
    value: None = None


class StopMsg(Message):
    type_: Literal[MessageType.stop] = Field(MessageType.stop, alias="type")
    value: None = None


def _to_json(val: Event) -> str:
    try:
        return json.dumps(val)
    except TypeError:
        # pickle and b64encode
        return "pck__" + base64.b64encode(pickle.dumps(val)).decode("utf-8")


def _from_json(val: Any) -> Event:
    if isinstance(val, str):
        if val.startswith("pck__"):
            return pickle.loads(base64.b64decode(val[5:]))
        else:
            return Event(**json.loads(val))
    else:
        return val


SerializableEvent = A[
    Event, PlainSerializer(_to_json, when_used="json"), BeforeValidator(_from_json)
]


class EventMsg(Message):
    type_: Literal[MessageType.event] = Field(MessageType.event, alias="type")
    value: list[SerializableEvent]


def _type_discriminator(v: dict | Message) -> str:
    typ = v.get("type", "any") if isinstance(v, dict) else v.type_

    if typ in MessageType.__members__:
        return typ
    else:
        return "any"


MessageUnion = A[
    A[AnnounceMsg, Tag("announce")]
    | A[IdentifyMsg, Tag("identify")]
    | A[ProcessMsg, Tag("process")]
    | A[StartMsg, Tag("start")]
    | A[StopMsg, Tag("stop")]
    | A[EventMsg, Tag("event")]
    | A[Message, Tag("any")],
    Discriminator(_type_discriminator),
]
MessageAdapter = TypeAdapter(MessageUnion)
