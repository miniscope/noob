import functools
import inspect
from collections.abc import Callable, Generator
from types import GeneratorType, GenericAlias, NoneType, UnionType
from typing import TYPE_CHECKING, Annotated, Any, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from noob.introspection import is_optional, is_union
from noob.node.spec import NodeSpecification
from noob.utils import resolve_python_identifier

if TYPE_CHECKING:
    from noob.input import InputCollection


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
_GENERATOR_METHOD_SENTINEL = "__is_generator_method__"

_TProcess = TypeVar("_TProcess", bound=Callable | Generator)


def process_method(func: _TProcess) -> _TProcess:
    """
    Decorator to mark a method as the designated 'process' method for a class.
    """
    setattr(func, _PROCESS_METHOD_SENTINEL, True)
    return func


class Node(BaseModel):
    """A node within a processing tube"""

    id: str
    """Unique identifier of the node"""
    spec: NodeSpecification
    enabled: bool = True
    """Starting state for a node being enabled. When a node is disabled, 
    it will be deinitialized and removed from the processing graph, 
    but the node object will still be kept by the Tube. Nodes can be disabled 
    and enabled during a tube's operation without recreating the tube. 
    When a node is disabled, other nodes that depend on it will not be disabled, 
    but they may never be called since their dependencies will never be satisfied."""

    _signals: list[Signal] = None
    _slots: dict[str, Slot] = None
    _gen: GeneratorType | None = None

    model_config = ConfigDict(extra="forbid")

    def model_post_init(self, __context: Any) -> None:
        """See docstring of :meth:`.process` for description of post init wrapping of generators"""
        if inspect.isgeneratorfunction(self.process):
            self._wrap_generator(self.process)

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

    def process(self, *args: Any, **kwargs: Any) -> Any | None:
        """
        Process some input, emitting it. See subclasses for details.

        If the process method is a generator, when the Node class is instantiated,
        this method is replaced by one that wraps creating and calling the generator.

        something like this:

        ```python
        gen = self.process()
        self.process = lambda: next(gen)
        ```

        Note that `send` handling is not implemented for generators,
        so `process` methods that are generators cannot depend on events from any other nodes
        (i.e. behave like source nodes).
        """
        raise NotImplementedError()

    @classmethod
    def from_specification(
        cls, spec: "NodeSpecification", input_collection: Union["InputCollection", None] = None
    ) -> "Node":
        """
        Create a node from its spec

        - resolve the node type
        - if a function, wrap it in a node class
        - if a class, just instantiate it
        """
        obj = resolve_python_identifier(spec.type_)

        params = spec.params if spec.params is not None else {}
        if input_collection:
            params = input_collection.get_node_params(params)

        # check if function by checking if callable -
        # Node classes do not have __call__ defined and thus should not be callable
        if inspect.isclass(obj):
            if issubclass(obj, Node):
                return obj(id=spec.id, spec=spec, enabled=spec.enabled, **params)
            else:
                return WrapClassNode(
                    id=spec.id, cls=obj, spec=spec, params=params, enabled=spec.enabled
                )
        else:
            return WrapFuncNode(id=spec.id, fn=obj, spec=spec, params=params, enabled=spec.enabled)

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

    def _wrap_generator(self, proc: Callable[[], GeneratorType]) -> None:
        """
        Wrap a `process` method when it is a generator,
        invoked in `model_post_init`
        """
        self._gen = proc()

        def _process():  # noqa: ANN202
            return next(self._gen)

        signature = inspect.signature(self.process)

        args = get_args(signature.return_annotation)
        if args:
            _process.__annotations__ = {"return": args[0]}

        self.__dict__["process"] = _process


class WrapClassNode(Node):
    """
    Wrap a non-Node class that has annotated one of its methods as being the "process" method
    using the :func:`process_method` decorator.

    Wrapping allows us to use arbitrary classes as Nodes within noob,
    which expects a `process` method,
    but avoids the problem of potentially breaking the class if it has its own attribute or method
    named `process`.

    After instantiating the outer wrapping class,
    instantiate the inner wrapped class using the `params` given to the outer wrapping class
    during :meth:`.model_post_init` .
    Then dynamically assign the discovered process method on the inner class to the
    outer class as `process`.

    Dynamic discovery at instantiation time, rather than statically defining an outer
    `process` method that then calls the inner method annotated with `process_method`
    does two things:

    - Allows us to statically infer whether the method is a regular function that `return`s or
      a generator using :func:`inspect.isgeneratorfunction` , which relies on a flag
      set on a method at the time it is defined: e.g. a method that internally switches between
      `return self._wrapped()` and `yield from self._wrapped()` would not be correctly detected.
    - Avoids modifying the signature of the wrapped process method with generic args and kwargs
    """

    cls: type
    params: dict[str, Any] = Field(default_factory=dict)

    instance: type | None = None
    _gen: Generator | None = PrivateAttr(default=None)

    def model_post_init(self, context: Any, /) -> None:
        """
        Get the method decorated with :func:`.process_method`,
        assign it to `process`, see class docstring.
        """
        self.instance = self.cls(**self.params)
        fn_name = self._get_process_method(self.cls)
        fn = getattr(self.instance, fn_name)
        self.__dict__["process"] = fn
        super().model_post_init(context)

    def deinit(self) -> None:
        self.instance = None

    def _collect_signals(self) -> list[Signal]:
        return Signal.from_callable(self.process)

    def _collect_slots(self) -> dict[str, Slot]:
        return Slot.from_callable(self.process)

    def _get_process_method(self, cls: type) -> str:
        process_func = None
        for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
            if hasattr(member, _PROCESS_METHOD_SENTINEL):
                if process_func:
                    raise TypeError(
                        f"Class {cls.__name__} has multiple 'process' methods. Only one is allowed."
                    )
                process_func = name
        if process_func is None:
            if hasattr(cls, "process") and inspect.isfunction(cls.process):
                process_func = "process"
            else:
                raise TypeError(
                    "Class must have 'process' method or decorate a method with @process_method."
                )

        return process_func


class WrapFuncNode(Node):
    fn: Callable
    params: dict = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """
        Complete wrapping `fn` without calling `super()`
        because we need to pass `params` to the function if it is a generator,
        and create a :func:`functools.partial` of it if it is not.
        """
        if inspect.isgeneratorfunction(self.fn):
            self._gen = self.fn(**self.params)
            self.__dict__["process"] = lambda: next(self._gen)
        elif inspect.isasyncgenfunction(self.fn):
            raise NotImplementedError("async generators not supported")
        else:
            self.__dict__["process"] = functools.partial(self.fn, **self.params)

    def _collect_signals(self) -> list[Signal]:
        return Signal.from_callable(self.fn)

    def _collect_slots(self) -> dict[str, Slot]:
        return Slot.from_callable(self.fn)


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
