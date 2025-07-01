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
    passed: dict[str, str] | None = None
    """
    FROM MIO - NOT STABLE, MAY BE REMOVED OR REPLACED
    
    Mapping of config values that must be passed when the tube is instantiated.

    Keys are the key in the config dictionary to be filled by passing, and values are a key that
    those values should be passed as.

    Examples:

        For a node with config field `height` , one can specify that it must be passed 
        on instantiation like this:

        .. code-block:: yaml

            nodes:
              node1:
                type: a_node 
                passed:
                  height: height_1
              node2:
                type: a_node
                passed:
                  height: height_2

        The tube should then be instantiated like:

        .. code-block:: python

            Tube.from_config(above_config, passed={'height_1': 1, 'height_2': 2})

    """
    fill: dict[str, str] | None = None
    """
    FROM MIO - NOT STABLE, MAY BE REMOVED OR REPLACED
    
    Values in the node config that should be dynamically filled from other nodes in the tube.

    Specified as {node_id}.{attribute}, these specify attributes and properties
    on the instantiated node class, not the config values for that node.

    This is useful for accessing some properties that might not be known until runtime
    like width and height of an input image.

    Examples:

        For a node class `camera` that has property `frame_width`,
        and node class `process` that has config value `width`,
        we would fill the config value like this: 

        .. code-block:: yaml

            nodes:
              cam:
                type: camera
              proc:
                type: process
                fill:
                  width: cam.frame_width

        The Tube class will then do something like this on instantiation:

        .. code-block:: python

            tube = TubeConfig(**the_above_values)

            cam = CameraNode(config=tube.nodes['cam'].config)

            proc_config = tube.nodes['proc'].config
            proc_config['width'] = cam.frame_width
            proc = ProcessingNode(config=proc_config)    

    """
