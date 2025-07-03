import inspect
from abc import abstractmethod
from collections.abc import Callable, Generator
from typing import Any, ParamSpec, TypeVar, get_args, get_origin, Annotated

from pydantic import BaseModel, ConfigDict, Field

from noob.node.spec import NodeSpecification
from noob.utils import resolve_python_identifier

TOutput = TypeVar("TOutput")
"""
Output Type typevar
"""
PWrap = ParamSpec("PWrap")


class Node(BaseModel):
    """A node within a processing tube"""

    id: str
    """Unique identifier of the node"""
    spec: NodeSpecification

    model_config = ConfigDict(extra="forbid")

    def init(self) -> None:
        """
        Start producing, processing, or receiving data.

        Default is a no-op.
        Subclasses do not need to override if they have no initialization logic.
        """
        pass

    def deinit(self) -> None:
        """
        Stop producing, processing, or receiving data

        Default is a no-op.
        Subclasses do not need to override if they have no deinit logic.
        """
        pass

    @abstractmethod
    def process(self, *args: Any, **kwargs: Any) -> Any | None:
        """Process some input, emitting it. See subclasses for details"""
        pass

    @classmethod
    def from_specification(cls, spec: "NodeSpecification") -> "Node":
        """
        Create a node from its spec

        - resolve the node type
        - if a function, wrap it in a node class
        - if a class, just instantiate it
        """
        obj = resolve_python_identifier(spec.type_)

        params = spec.params if spec.params is not None else {}

        # check if function by checking if callable -
        # Node classes do not have __call__ defined and thus should not be callable
        if inspect.isclass(obj):
            if issubclass(obj, Node):
                return obj(id=spec.id, spec=spec, **params)
            else:
                raise NotImplementedError("Handle wrapping classes")
        else:
            return WrapNode(id=spec.id, fn=obj, spec=spec, params=params)


class WrapNode(Node):
    fn: Callable[PWrap, TOutput]
    params: dict = Field(default_factory=dict)

    def process(self, *args: PWrap.args, **kwargs: PWrap.kwargs) -> TOutput | None:

        kwargs.update(self.params)
        value = self.fn(*args, **kwargs)
        return_annotation = inspect.signature(self.fn).return_annotation
        names = self._collect_slot_names(return_annotation)

        if inspect.isgenerator(value):
            value = next(value)

        if len(names) == 1:
            values = (value,)
        else:
            values = tuple(value)

        event = {name: value for name, value in zip(names, values)}
        return event

    @staticmethod
    def _collect_slot_names(return_annotation: Annotated[Any, Any]) -> list[str]:
        from noob import Name

        names = []

        if get_origin(return_annotation) is Annotated:
            for argument in get_args(return_annotation):
                if isinstance(argument, Name):
                    names.append(argument.name)

        elif get_origin(return_annotation) in [tuple, Generator]:
            for argument in get_args(return_annotation):
                if get_origin(argument) in [Annotated, Generator, tuple]:
                    names += WrapNode._collect_slot_names(argument)
                elif argument is not type(None):
                    names = ["value"]

        elif return_annotation and not return_annotation is inspect.Signature.empty:
            names = ["value"]

        return names


class Edge(BaseModel):
    """
    Directed connection between an output slot a node and an input slot in another node
    """

    source_node: str
    source_slot: str | None = None
    target_node: str
    target_slot: str | None = None
