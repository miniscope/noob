import functools
import inspect
from abc import abstractmethod
from collections.abc import Callable, Generator
from types import GenericAlias, NoneType, UnionType
from typing import Annotated, Any, ParamSpec, TypeVar, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from noob.introspection import is_optional, is_union
from noob.node.spec import NodeSpecification
from noob.utils import resolve_python_identifier

TOutput = TypeVar("TOutput")
"""
Output Type typevar
"""
PWrap = ParamSpec("PWrap")


class Slot(BaseModel):
    name: str
    annotation: Any
    required: bool = True

    @classmethod
    def from_callable(cls, func: Callable) -> dict[str, "Slot"]:
        slots = {}
        sig = inspect.signature(func)

        for name, param in sig.parameters.items():
            slots[name] = Slot(
                name=name,
                annotation=param.annotation,
                required=not (is_optional(param.annotation) and param.default is None),
            )
        return slots


class Signal(BaseModel):
    name: str
    type_: type | NoneType | UnionType | GenericAlias

    # Unable to generate pydantic-core schema for <class 'types.UnionType'>
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_callable(cls, func: Callable) -> list["Signal"]:
        signals = []
        return_annotation = inspect.signature(func).return_annotation
        for name, type_ in cls._collect_signal_names(return_annotation):
            signals.append(Signal(name=name, type_=type_))
        return signals

    @classmethod
    def _collect_signal_names(
        cls, return_annotation: Annotated[Any, Any]
    ) -> list[tuple[str, type]]:
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

        default_name = "value"
        names = []

        # --- get yield type of generators before handling other types
        if get_origin(return_annotation) is Generator:
            return_annotation = get_args(return_annotation)[0]
        # ---

        # def func() -> Annotated[...]
        if get_origin(return_annotation) is Annotated:
            for argument in get_args(return_annotation):
                if isinstance(argument, Name):
                    names.append((argument.name, get_args(return_annotation)[0]))

        # def func() -> tuple[...]: OR def func() -> A | B:
        elif get_origin(return_annotation) is tuple or is_union(return_annotation):
            for argument in get_args(return_annotation):
                if get_origin(argument) in [Annotated, tuple]:
                    names += Signal._collect_signal_names(argument)
                elif argument not in [type(None), None]:
                    names = [(default_name, return_annotation)]

        # def func() -> type:
        elif return_annotation and (return_annotation is not inspect.Signature.empty):
            names = [(default_name, return_annotation)]

        return names


_PROCESS_METHOD_SENTINEL = "__is_process_method__"


def process_method(func: Callable) -> Callable:
    """
    Decorator to mark a method as the designated 'process' method for a class.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    setattr(wrapper, _PROCESS_METHOD_SENTINEL, True)
    return wrapper


class Node(BaseModel):
    """A node within a processing tube"""

    id: str
    """Unique identifier of the node"""
    spec: NodeSpecification

    _signals: list[Signal] = None
    _slots: dict[str, Slot] = None

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
                return WrapClassNode(id=spec.id, cls=obj, spec=spec, params=params)
        else:
            return WrapFuncNode(id=spec.id, fn=obj, spec=spec, params=params)

    @property
    def signals(self) -> list[Signal]:
        if self._signals is None:
            self._signals = self._collect_signals()
        if not self._signals:
            self._signals = [Signal(name="value", type_=Any)]
        return self._signals

    def _collect_signals(self) -> list[Signal]:
        return Signal.from_callable(self.process)

    @property
    def slots(self) -> dict[str, Slot]:
        if self._slots is None:
            self._slots = self._collect_slots()
        return self._slots

    def _collect_slots(self) -> dict[str, Slot]:
        return Slot.from_callable(self.process)


class WrapClassNode(Node):
    cls: type
    params: dict[str, Any] = Field(default_factory=dict)
    instance: type | None = None

    def init(self) -> None:

        self.instance = self._map_main_method(self.obj)(**self.params)

    def process(self, *args: Any, **kwargs: Any) -> Any:
        return self.instance.process(*args, **kwargs)

    def deinit(self) -> None:
        self.instance = None

    def _collect_signals(self) -> list[Signal]:
        return Signal.from_callable(self.instance.process)

    def _collect_slots(self) -> dict[str, Slot]:
        return Slot.from_callable(self.instance.process)

    def _map_main_method(self, cls: type) -> type:
        process_func = None
        for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
            # inspect.isfunction for classmethod and staticmethod appears as
            # wrapped methods. do we have to functools.unwrap them?
            if hasattr(member, _PROCESS_METHOD_SENTINEL) or name == "process":
                if process_func:
                    raise TypeError(
                        f"Class {cls.__name__} has multiple 'process' methods. Only one is allowed."
                    )
                process_func = member

        if process_func:
            cls.process = process_func

        return cls


class WrapFuncNode(Node):
    fn: Callable[PWrap, TOutput]
    params: dict = Field(default_factory=dict)
    _gen: Generator[TOutput, None, None] = PrivateAttr(default=None)

    def _collect_signals(self) -> list[Signal]:
        return Signal.from_callable(self.fn)

    def _collect_slots(self) -> dict[str, Slot]:
        return Slot.from_callable(self.fn)

    def process(self, *args: PWrap.args, **kwargs: PWrap.kwargs) -> TOutput | None:
        kwargs.update(self.params)
        value = self.fn(*args, **kwargs)

        if inspect.isgenerator(value):
            if self._gen is None:
                self._gen = value
            try:
                value = next(self._gen)
            except StopIteration as e:
                # generator is exhausted
                raise RuntimeError("Generator node stopped its iteration") from e

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
    required: bool = True
