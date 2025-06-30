import inspect
import sys
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Generic, ParamSpec, TypeVar, Unpack

from pydantic import BaseModel, Field

from noob.event import Event
from noob.utils import resolve_python_identifier

if sys.version_info < (3, 11) or sys.version_info < (3, 12):

    from typing_extensions import TypedDict
else:
    from typing import TypedDict

# if TYPE_CHECKING:
from noob.node.spec import NodeSpecification

TInput = TypeVar("TInput")
"""
Input Type typevar
"""
TOutput = TypeVar("TOutput", bound=dict[str, Any])
"""
Output Type typevar
"""
PWrap = ParamSpec("PWrap")

"""
mapping from node input kwargs to another node's outputs

input_0: node_b.output_0
"""


class NodeConfig(TypedDict, total=False):
    """
    Abstract parent TypedDict that each node inherits from to define
    what fields it needs to be configured.
    """


class Node(BaseModel, Generic[TInput, TOutput]):
    """A node within a processing tube"""

    id: str
    """Unique identifier of the node"""
    spec: "NodeSpecification"
    params: NodeConfig = Field(default_factory=dict)

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
    def from_specification(cls, spec: "NodeSpecification") -> "Node":
        """
        Create a node from its config

        - resolve the node type
        - if a function, wrap it in a node class
        - if a class, just instantiate it
        """
        obj = resolve_python_identifier(spec.type_)
        # check if function by checking if callable -
        # Node classes do not have __call__ defined and thus should not be callable
        if inspect.isclass(obj):
            if issubclass(obj, Node):
                return obj(id=spec.id, spec=spec)
            else:
                raise NotImplementedError("Handle wrapping classes")
        else:
            return WrapNode(id=spec.id, fn=obj, spec=spec)


class WrapNode(Node):
    fn: Callable[PWrap, TOutput]

    def process(self, **kwargs: PWrap.kwargs) -> TOutput | None:
        kwargs.update(self.params)
        return self.fn(**kwargs)


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

    def key(self) -> str:
        """
        Return some unique string that represents the dependency

        NodeA.signal2
        NodeA.signal2[type=gather,n=5]  --> we aren't doing this anymore if edge operations are now separate nodes.
        """

    def update(self, events: list[Event]) -> None:
        """"""
