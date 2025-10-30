from __future__ import annotations

import builtins
import re
import sys
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, TypeAlias, TypedDict

from annotated_types import Ge
from pydantic import AfterValidator, Field, TypeAdapter

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

# type aliases, mostly for documentation's sake
NodeID: TypeAlias = Annotated[str, AfterValidator(_is_identifier), AfterValidator(_not_reserved)]
Epoch: TypeAlias = Annotated[int, Ge(0)]
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
