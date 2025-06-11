from typing import NotRequired, TypeAlias

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from noob.types import AbsoluteIdentifier, PythonIdentifier

_SlotSpec: TypeAlias = PythonIdentifier | int
"""
the left side of a dependency spec - what the dependent nodes plug into

e.g. the "slot" in the following

depends:
- slot: node_a.signal
"""

_DependsBasic: TypeAlias = dict[_SlotSpec, AbsoluteIdentifier]


class _DependsExpanded(TypedDict):
    source: AbsoluteIdentifier
    map: NotRequired[bool]
    """If True, map the input and call the node once per item in the iterator"""
    gather: NotRequired[PythonIdentifier | int]
    """
    Either an int of number of items to gather, or another slot name *on the same node*
    that triggers the node to be called with the gathered values
    
    depends:
    - items:
      source: node_a.output_0
      gather: a_key
    - a_key: node_b.output_0 
    """


DependsType: TypeAlias = _DependsBasic | dict[_SlotSpec, _DependsExpanded]


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
    to another node's output signals,
    or by an expanded parameterization that declares mapping/gathering relationships.
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
