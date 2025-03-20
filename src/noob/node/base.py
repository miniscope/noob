import sys
from abc import abstractmethod
from typing import (
    Any,
    ClassVar,
    Generic,
    TypeVar,
    Unpack,
    final,
)

from pydantic import BaseModel, Field

if sys.version_info < (3, 11) or sys.version_info < (3, 12):
    from typing import Self

    from typing_extensions import TypedDict
else:
    from typing import Self, TypedDict

TInput = TypeVar("TInput")
"""
Input Type typevar
"""
TOutput = TypeVar("TOutput", bound=dict[str, Any])
"""
Output Type typevar
"""


class _NodeMap(TypedDict):
    source: str
    target: str


class NodeConfig(TypedDict, total=False):
    """
    Abstract parent TypedDict that each node inherits from to define
    what fields it needs to be configured.
    """


class NodeSpecification(BaseModel):
    """
    Specification for a single processing node within a tube .yaml file.
    Distinct from a :class:`.NodeConfig`, which is a generic TypedDict that each
    node defines to declare its parameterization.
    """

    type_: str = Field(..., alias="type")
    """
    Shortname of the type of node this configuration is for.

    Subclasses should override this with a default.
    """
    id: str
    """The unique identifier of the node"""
    outputs: list[_NodeMap] | None = None
    """List of Node IDs to be used as output"""
    config: dict | None = None
    """Additional configuration for this node, parameterized by a TypedDict for the class"""
    passed: dict[str, str] | None = None
    """
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


class Node(BaseModel, Generic[TInput, TOutput]):
    """A node within a processing tube"""

    name: ClassVar[str]
    """
    Shortname for this type of node to match configs to node types
    """

    id: str
    """Unique identifier of the node"""
    config: NodeConfig | None = None

    input_type: ClassVar[type[TInput]]
    output_type: ClassVar[type[TOutput]]

    def start(self) -> None:
        """
        Start producing, processing, or receiving data.

        Default is a no-op.
        Subclasses do not need to override if they have no initialization logic.
        """
        pass

    def stop(self) -> None:
        """
        Stop producing, processing, or receiving data

        Default is a no-op.
        Subclasses do not need to override if they have no deinit logic.
        """
        pass

    @abstractmethod
    def process(self, **kwargs: Unpack[TInput]) -> TOutput | None:
        """Process some input, emitting it. See subclasses for details"""
        pass

    @classmethod
    def from_specification(cls, config: NodeSpecification) -> Self:
        """
        Create a node from its config
        """
        return cls(id=config.id, config=config.config)

    @classmethod
    @final
    def node_types(cls) -> dict[str, type["Node"]]:
        """
        Map of all imported :attr:`.Node.name` names to node classes
        """

        node_types = {}
        to_check = cls.__subclasses__()
        while to_check:
            node = to_check.pop()
            if node not in (Sink, Source, Transform) and node.name in node_types:
                raise ValueError(
                    f"Repeated node name identifier: {node.name}, found in:\n"
                    f"- {node_types[node.name]}\n- {node}"
                )

            to_check.extend(node.__subclasses__())
            if node not in (Sink, Source, Transform):
                node_types[node.name] = node
        return node_types


class Source(Node, Generic[TInput, TOutput]):
    """A source of data in a processing tube"""

    input_type: ClassVar[None] = None

    @abstractmethod
    def process(self) -> TOutput:
        """
        Process some data, returning an output.


        .. note::

            The `process` method should not directly call or pass
            data to subscribed output nodes, but instead return the output
            and allow a containing tube class to handle dispatching data.

        """


class Sink(Node, Generic[TInput, TOutput]):
    """A sink of data in a processing tube"""

    output_type: ClassVar[None] = None

    @abstractmethod
    def process(self, **kwargs: Unpack[TInput]) -> None:
        """
        Process some incoming data, returning None

        .. note::

            The `process` method should not directly be called or passed data,
            but instead should be called by a containing tube class.

        """


class Transform(Node, Generic[TInput, TOutput]):
    """
    An intermediate processing node that transforms some input to output
    """

    @abstractmethod
    def process(self, **kwargs: Unpack[TInput]) -> TOutput:
        """
        Process some incoming data, yielding a transformed output

        .. note::

            The `process` method should not directly call or be called by
            output or input nodes, but instead return the output
            and allow a containing tube class to handle dispatching data.

        """


class Edge(BaseModel):
    """
    Directed connection between an output slot a node and an input slot in another node
    """

    source_node: Node
    source_slot: str | None = None
    target_node: Node
    target_slot: str | None = None
