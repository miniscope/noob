import inspect
from abc import abstractmethod
from collections.abc import Callable, Generator
from typing import Annotated, Any, ParamSpec, TypeVar, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

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

    _signals: list[str] = PrivateAttr(default_factory=list)

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

    @property
    def signals(self) -> list[str]:
        return self._signals


class WrapNode(Node):
    fn: Callable[PWrap, TOutput]
    params: dict = Field(default_factory=dict)
    _gen: Generator[TOutput, None, None] = PrivateAttr(default=None)

    def model_post_init(self, __context: None = None) -> None:
        self._signals = self.collect_signal_names(self.fn)

    def collect_signal_names(self, func: Callable) -> list[str]:
        if getattr(func, "__module__", None) == "builtins":
            return ["value"]

        return_annotation = inspect.signature(func).return_annotation
        return self._collect_signal_names(return_annotation)

    @staticmethod
    def _collect_signal_names(return_annotation: Annotated[Any, Any]) -> list[str]:
        """
        Recursive kernel for extracting name attribute of Name metadata from a type annotation.
        If the outermost origin is `typing.Annotated`, it extracts all Name objects from its
        arguments and append.
        If the outermost origin is `tuple` or `Generator`, grab the argument and recurses into
        this function.
        If the outermost is `Generator`, or `tuple` but the inner layer isn't one of `Generator`,
        `tuple`, or `Annotated`, assume it's an unnamed signal (no Name inside).
        If the outermost is none of `Generator`, `tuple`, or `Annotated`, assume it's an unnamed
        signal.

        Returns a list of names.
        """
        from noob import Name

        names = []

        if get_origin(return_annotation) is Annotated:
            for argument in get_args(return_annotation):
                if isinstance(argument, Name):
                    names.append(argument.name)

        elif get_origin(return_annotation) in [tuple, Generator]:
            for argument in get_args(return_annotation):
                if get_origin(argument) in [Annotated, Generator, tuple]:
                    names += WrapNode._collect_signal_names(argument)
                elif argument not in [type(None), None]:
                    names = ["value"]

        elif return_annotation and (return_annotation is not inspect.Signature.empty):
            names = ["value"]

        return names

    def process(self, *args: PWrap.args, **kwargs: PWrap.kwargs) -> TOutput | None:
        kwargs.update(self.params)
        value = self.fn(*args, **kwargs)

        if inspect.isgenerator(value):
            if self._gen is None:
                self._gen = value
            value = next(self._gen)

        return value


class Edge(BaseModel):
    """
    Directed connection between an output slot a node and an input slot in another node
    """

    source_node: str
    source_signal: str | None = None
    target_node: str
    target_slot: str | int | None = None
    """
    - For kwargs, target_slot is the name of the kwarg that the value is passed to.
    - For positional arguments, target_slot is an integer that indicates the index of the arg
    - For scalar arguments, target slot is None 
    """
