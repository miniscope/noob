from __future__ import annotations

import inspect
from functools import cached_property
from typing import TYPE_CHECKING, Annotated, TypeAlias, TypedDict

from annotated_types import Len
from pydantic import BaseModel, ConfigDict, Field, field_validator

from noob.types import AbsoluteIdentifier, DependencyIdentifier, PythonIdentifier
from noob.utils import resolve_python_identifier
from noob.yaml import id_optional_json_schema

if TYPE_CHECKING:
    from noob.node.base import Signal, Slot

_DependsBasic: TypeAlias = Annotated[
    dict[PythonIdentifier, DependencyIdentifier], Len(min_length=1, max_length=1)
]
"""
Standard dependency declaration, a mapping from a node's `slot` to a `node.signal` pair:

Examples:

    for a pair of nodes like this:
    
    ```python
    def node_a() -> Annotated[Generator[int], Name("index")]:
        yield from count()
    
    def node_b(my_value: int) -> None:
        print(my_value)
    ```
    
    one would express "pass node_a.index to node_b's `my_value`" like
    
    ```yaml
    nodes:
      a:
        type: node_a
      b:
        type: node_b
        depends:
        - my_value: a.index
    ```

"""

DependsType: TypeAlias = list[DependencyIdentifier | _DependsBasic] | DependencyIdentifier
"""
Either an absolute identifier (which is treated as a positional-only arg)
or a dict mapping as described in _DependsBasic.

Examples:

    ```python
    def example(positional_only: int, /, another_arg: str) -> None:
        return another_arg * positional_only
    ```    
    
    ```yaml
    nodes:
      zzz:
        type: example
        depends:
        - a.value
        - another_arg: b.value
    ```
    
    When a dependency is a scalar value passed to the first positional argument,
    it can be specified with a scalar reference to an absolute identifier.
    For example, if one wanted to return a scalar value from a `return` node,
    specify the dependency like this:
    
    ```yaml
    nodes:
      yyy:
        type: return
        depends: a.value
    ```
    
    and to return the same value wrapped in a list...
    
    ```yaml
    nodes:
      yyy:
        type: return
        depends: 
        - a.value
    ```
    
"""


class NodeSpecification(BaseModel):
    """
    Specification for a single processing node within a tube .yaml file.
    """

    type_: AbsoluteIdentifier = Field(..., alias="type")
    """
    Shortname of the type of node this configuration is for.

    Subclasses should override this with a default.
    """
    id: PythonIdentifier
    """The unique identifier of the node"""
    depends: DependsType | None = None
    """Dependency specification for the node.
    
    Can be specified as a simple mapping from this node's input slots 
    to another node's output signals passed as kwargs,
    or as a flat list of node.signal identifiers that are passed as positional args.
    """
    params: dict | None = None
    """Static kwargs to pass to this node, 
    parameterized the signature of a function node, or
    by a TypedDict for a class node.
    """
    enabled: bool = True
    """
    If this flag is False, the node will not be initialized 
    or included in the `:meth:.Tube.graph`.
    """
    stateful: bool | None = None
    """
    See :attr:`.Node.stateful` ,
    explicitly set statefulness on a node, overriding its default.
    If ``None`` , use the default set on the node class.
    """
    description: str | None = None
    """An optional description of the node"""

    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    @field_validator("depends", mode="after")
    @classmethod
    def slots_unique(cls, val: DependsType | None) -> DependsType | None:
        """
        Ensure slots are unique in dependency spec: can't map more than one signal to the same slot
        """
        if val is None or isinstance(val, str):
            return val
        seen = set()
        for dep in val:
            if isinstance(dep, str):
                continue
            signal = next(iter(dep.keys()))
            if signal in seen:
                raise ValueError(f"Duplicate signal in dependencies: {signal}")
            seen.add(signal)
        return val

    @cached_property
    def nodeinfo(self) -> NodeInfo:
        """Information about the node that this spec is for."""
        from noob.node.base import Node, WrapClassNode, WrapFuncNode

        obj = resolve_python_identifier(self.type_)
        if inspect.isclass(obj):
            node_cls = obj if issubclass(obj, Node) else WrapClassNode
        else:
            node_cls = WrapFuncNode

        return NodeInfo(
            node_id=self.id,
            type=self.type_,
            signals=node_cls.get_signals(self),
            slots=node_cls.get_slots(self),
        )

    __get_pydantic_json_schema__ = classmethod(id_optional_json_schema)  # type: ignore[var-annotated]


class NodeInfo(TypedDict):
    """
    Metadata about the completed spec given the combination of a node spec and a node class.

    The spec if purely static, the node class is *mostly* static,
    but some important properties - notably signals and slots -
    can be dynamic: i.e. the signals or slots of the node depend on the spec.

    This metadata is primarily used in visualization or inspection of tubes, rather than at runtime
    """

    node_id: str
    """Node ID whose spec this NodeInfo was computed from"""
    type: str
    """fully-qualified module.object name for the node type"""
    signals: dict[str, Signal]
    """Signals computed from the spec and node class"""
    slots: dict[str, Slot]
    """Slots computed from the spec and node class"""
