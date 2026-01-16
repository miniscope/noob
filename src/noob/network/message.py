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
    Tag,
    TypeAdapter,
    WrapSerializer,
)
from pydantic_core.core_schema import SerializerFunctionWrapHandler

from noob.const import META_SIGNAL
from noob.event import Event, MetaSignal
from noob.types import Picklable

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class MessageType(StrEnum):
    announce = "announce"
    identify = "identify"
    process = "process"
    init = "init"
    deinit = "deinit"
    start = "start"
    status = "status"
    stop = "stop"
    event = "event"
    error = "error"


class NodeStatus(StrEnum):
    stopped = "stopped"
    """Node is deinitialized - does not have an instantiated node, etc., but is responsive."""
    waiting = "waiting"
    """Node is waiting for its dependency nodes to be ready"""
    ready = "ready"
    """Node is ready to process events"""
    running = "running"
    """
    Node is running in free-run mode.
    Note that we do not update status for every process call at the moment,
    as that level of granularity is not relevant to the command node when sending commands 
    """
    closed = "closed"
    """Node is permanently gone, should not be expected to respond to further messages."""


class Message(BaseModel):
    type_: MessageType = Field(..., alias="type")
    node_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    value: Any = None

    model_config = ConfigDict(use_enum_values=True, validate_by_alias=True, serialize_by_alias=True)

    @classmethod
    def from_bytes(cls, msg: list[bytes]) -> "Message":
        return MessageAdapter.validate_json(msg[-1].decode("utf-8"))

    def to_bytes(self) -> bytes:
        return self.model_dump_json().encode("utf-8")


class IdentifyValue(TypedDict):
    node_id: str
    outbox: str
    status: NodeStatus
    signals: list[str] | None
    slots: list[str] | None


class AnnounceValue(TypedDict):
    inbox: str
    nodes: dict[str, IdentifyValue]


class ErrorValue(TypedDict):
    err_type: type[Exception]
    err_args: tuple
    traceback: str


class ProcessValue(TypedDict):
    epoch: int
    input: dict | None


class AnnounceMsg(Message):
    """Command node 'announces' identities of other peers and the events they emit"""

    type_: Literal[MessageType.announce] = Field(MessageType.announce, alias="type")
    value: AnnounceValue


class IdentifyMsg(Message):
    """A node sends its configuration to the command node on initialization"""

    type_: Literal[MessageType.identify] = Field(MessageType.identify, alias="type")
    value: IdentifyValue


class ProcessMsg(Message):
    """Process a single iteration of the graph"""

    type_: Literal[MessageType.process] = Field(MessageType.process, alias="type")
    value: ProcessValue
    """Any process-scoped input passed to the `process` call"""


class InitMsg(Message):
    """Initialize nodes within node runners"""

    type_: Literal[MessageType.init] = Field(MessageType.init, alias="type")
    value: None = None


class DeinitMsg(Message):
    """Deinitializes nodes within node runners"""

    type_: Literal[MessageType.deinit] = Field(MessageType.deinit, alias="type")
    value: None = None


class StartMsg(Message):
    """Start free-running nodes"""

    type_: Literal[MessageType.start] = Field(MessageType.start, alias="type")
    value: int | None = None


class StatusMsg(Message):
    """Node updating its current status"""

    type_: Literal[MessageType.status] = Field(MessageType.status, alias="type")
    value: NodeStatus


class StopMsg(Message):
    """Stop processing"""

    type_: Literal[MessageType.stop] = Field(MessageType.stop, alias="type")
    value: None = None


class ErrorMsg(Message):
    """An error occurred in one of the processing nodes"""

    type_: Literal[MessageType.error] = Field(MessageType.error, alias="type")
    value: Picklable[ErrorValue]

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _to_json(val: Event, handler: SerializerFunctionWrapHandler) -> Any:
    if val["signal"] == META_SIGNAL and val["value"] is MetaSignal.NoEvent:
        val["value"] = MetaSignal.NoEvent.value

    try:
        return handler(val)
    except TypeError:
        # pickle and b64encode
        return "pck__" + base64.b64encode(pickle.dumps(val)).decode("utf-8")


def _from_json(val: Any) -> Event:
    if isinstance(val, str):
        if val.startswith("pck__"):
            evt = pickle.loads(base64.b64decode(val[5:]))
        else:
            evt = Event(**json.loads(val))  # type: ignore[typeddict-item]
        if evt["signal"] == META_SIGNAL and evt["value"] == MetaSignal.NoEvent.value:
            evt["value"] = MetaSignal.NoEvent
        return evt
    else:
        return val


SerializableEvent = A[
    Event, WrapSerializer(_to_json, when_used="json"), BeforeValidator(_from_json)
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
    | A[InitMsg, Tag("init")]
    | A[DeinitMsg, Tag("deinit")]
    | A[StartMsg, Tag("start")]
    | A[StatusMsg, Tag("status")]
    | A[StopMsg, Tag("stop")]
    | A[EventMsg, Tag("event")]
    | A[ErrorMsg, Tag("error")]
    | A[Message, Tag("any")],
    Discriminator(_type_discriminator),
]
MessageAdapter = TypeAdapter[MessageUnion](MessageUnion)
