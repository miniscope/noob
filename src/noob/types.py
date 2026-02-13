from __future__ import annotations

import base64
import builtins
import pickle
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, NamedTuple, TypeAlias, TypedDict, TypeVar

from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    GetCoreSchemaHandler,
    PlainSerializer,
    TypeAdapter,
    WrapSerializer,
)
from pydantic_core import CoreSchema, PydanticSerializationError, core_schema
from pydantic_core.core_schema import SerializerFunctionWrapHandler

from noob.const import RESERVED_IDS

if sys.version_info < (3, 13):
    from typing_extensions import TypeIs
else:
    from typing import TypeIs

if TYPE_CHECKING:
    from noob import Tube
    from noob.runner import TubeRunner

CONFIG_ID_PATTERN = r"[\w\-\/#]+"
"""
Any alphanumeric string (\\w), as well as
- ``-``
- ``/``
- ``#``
(to allow hierarchical IDs as well as fragment IDs).

Specficially excludes ``.`` to avoid confusion between IDs, paths, and python module names

May be made less restrictive in the future, will not be made more restrictive.
"""


def _is_identifier(val: str) -> str:
    assert val.isidentifier(), "Must be a valid python identifier"
    return val


def _is_absolute_identifier(val: str) -> str:
    from noob.node import SPECIAL_NODES

    if val in SPECIAL_NODES:
        return val

    if "." not in val:
        assert (
            val in builtins.__dict__
        ), "If not an absolute module.Object identifier, must be in builtins"
        return val

    assert not val.startswith("."), "Cannot use relative identifiers"
    for part in val.split("."):
        assert part.isidentifier(), f"{part} is not a valid python identifier within {val}"
    return val


def _is_signal_slot(val: str) -> str:
    """is a {node}.{signal/slot} identifier"""
    parts = val.split(".")
    assert len(parts) == 2, "Must be a {node}.{signal} identifier"
    assert parts[0] != "", "Must specify a node id"
    assert parts[1] != "", "Must specify a signal"
    _is_identifier(parts[0])
    _is_identifier(parts[1])
    return val


def _not_reserved(val: str) -> str:
    assert val not in RESERVED_IDS, f"Cannot used reserved ID {val}"
    return val


def _from_isoformat(val: str | datetime) -> datetime:
    if isinstance(val, str):
        val = datetime.fromisoformat(val)
    return val


def _to_isoformat(val: datetime) -> str:
    return val.isoformat()


def _to_jsonable_pickle(val: Any, handler: SerializerFunctionWrapHandler) -> Any:
    try:
        return handler(val)
    except (TypeError, PydanticSerializationError):
        return "pck__" + base64.b64encode(pickle.dumps(val)).decode("utf-8")


def _from_jsonable_pickle(val: Any) -> Any:
    if isinstance(val, str) and val.startswith("pck__"):
        return pickle.loads(base64.b64decode(val[5:]))
    return val


Range: TypeAlias = tuple[int, int] | tuple[float, float]
PythonIdentifier: TypeAlias = Annotated[
    str, AfterValidator(_is_identifier), AfterValidator(_not_reserved)
]
"""
A single valid python identifier and not one of the reserved identifiers.

See: https://docs.python.org/3.13/library/stdtypes.html#str.isidentifier
"""
AbsoluteIdentifier: TypeAlias = Annotated[str, AfterValidator(_is_absolute_identifier)]
"""
- A valid python identifier, including globally accessible namespace like module.submodule.ClassName
OR 
- a name of a builtin function/type
"""
DependencyIdentifier: TypeAlias = Annotated[str, AfterValidator(_is_signal_slot)]
"""
A {node_id}.{signal} identifier. 

The `node_id` part must be a valid {class}`.PythonIdentifier` .
"""


ConfigID: TypeAlias = Annotated[str, Field(pattern=CONFIG_ID_PATTERN)]
"""
A string that refers to a config file by the ``id`` field in that config
"""
ConfigSource: TypeAlias = Path | PathLike[str] | ConfigID
"""
Union of all types of config sources
"""

SerializableDatetime = Annotated[
    datetime, BeforeValidator(_from_isoformat), PlainSerializer(_to_isoformat, when_used="json")
]
TPickle = TypeVar("TPickle")
Picklable = Annotated[
    TPickle,
    BeforeValidator(_from_jsonable_pickle),
    WrapSerializer(_to_jsonable_pickle, when_used="json"),
]

# type aliases, mostly for documentation's sake
NodeID: TypeAlias = Annotated[str, AfterValidator(_is_identifier), AfterValidator(_not_reserved)]
SignalName: TypeAlias = Annotated[str, AfterValidator(_is_identifier)]

ReturnNodeType: TypeAlias = None | dict[str, Any] | Any


@dataclass
class Name:
    """Name of some node output

    Examples:
        def my_function() -> Annotated[int, Name("charlie")]: ...

    """

    name: str


def valid_config_id(val: Any) -> TypeIs[ConfigID]:
    """
    Checks whether a string is a valid config id.
    """
    return bool(re.fullmatch(CONFIG_ID_PATTERN, val))


# --------------------------------------------------
# Type adapters
# --------------------------------------------------

AbsoluteIdentifierAdapter = TypeAdapter(AbsoluteIdentifier)


class RunnerContext(TypedDict):
    runner: TubeRunner
    tube: Tube


class EpochSegment(NamedTuple):
    node_id: NodeID | Literal["tube"]
    epoch: int


class Epoch(tuple[EpochSegment, ...]):
    def __new__(cls, epoch: int | tuple[tuple[NodeID, int], ...]):
        if isinstance(epoch, int):
            epoch = (EpochSegment("tube", epoch),)
        return super().__new__(cls, epoch)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        tuple_schema = core_schema.tuple_schema([handler(EpochSegment)])

        def _cast(val: tuple | Epoch) -> Epoch:
            if not isinstance(val, Epoch):
                val = Epoch(val)
            return val

        return core_schema.chain_schema(
            steps=[tuple_schema, core_schema.no_info_plain_validator_function(_cast)],
        )

    def __eq__(self, other: Epoch | int) -> bool:
        if isinstance(other, int):
            if len(self) == 1:
                return self[0].epoch == other
            else:
                return False
        else:
            return tuple.__eq__(self, other)
            # msg = f"Can only compare equality to an int or another epoch, got {other}"
            # raise TypeError(msg)

    __hash__ = tuple.__hash__

    def __gt__(self, other: Epoch | int) -> bool:
        if isinstance(other, Epoch | tuple):
            return tuple(e.epoch for e in self) > tuple(e.epoch for e in other)
        elif isinstance(other, int):
            return self[0].epoch > other
        else:
            raise TypeError("Can only compare equality to an int or another epoch")

    def __ge__(self, other: Epoch | int) -> bool:
        return self == other or self > other

    def __lt__(self, other: Epoch | int) -> bool:
        return not self > other

    def __le__(self, other: Epoch | int) -> bool:
        return self == other or self < other
