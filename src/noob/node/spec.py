from typing import Annotated, TypeAlias

from annotated_types import Len
from pydantic import BaseModel, Field

from noob.types import AbsoluteIdentifier, PythonIdentifier

_DependsBasic: TypeAlias = Annotated[
    dict[PythonIdentifier, AbsoluteIdentifier], Len(min_length=1, max_length=1)
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

DependsType: TypeAlias = AbsoluteIdentifier | _DependsBasic
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
"""


class NodeSpecification(BaseModel):
    """
    Specification for a single processing node within a tube .yaml file.
    Distinct from a :class:`.NodeConfig`, which is a generic TypedDict that each
    node defines to declare its parameterization.
    """

    type_: AbsoluteIdentifier = Field(..., alias="type")
    """
    Shortname of the type of node this configuration is for.

    Subclasses should override this with a default.
    """
    id: str
    """The unique identifier of the node"""
    depends: list[DependsType] | None = None
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
