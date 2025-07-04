import builtins
import re
import sys
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Annotated, Any, TypeAlias

from pydantic import AfterValidator, Field, TypeAdapter

if sys.version_info < (3, 13):
    from typing_extensions import TypeIs
else:
    from typing import TypeIs

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


Range: TypeAlias = tuple[int, int] | tuple[float, float]
PythonIdentifier: TypeAlias = Annotated[str, AfterValidator(_is_identifier)]
"""
A single valid python identifier.

See: https://docs.python.org/3.13/library/stdtypes.html#str.isidentifier
"""
AbsoluteIdentifier: TypeAlias = Annotated[str, AfterValidator(_is_absolute_identifier)]
"""
- A valid python identifier, including globally accessible namespace like module.submodule.ClassName
OR 
- a name of a builtin function/type
"""

ConfigID: TypeAlias = Annotated[str, Field(pattern=CONFIG_ID_PATTERN)]
"""
A string that refers to a config file by the ``id`` field in that config
"""
ConfigSource: TypeAlias = Path | PathLike[str] | ConfigID
"""
Union of all types of config sources
"""
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
