from __future__ import annotations

import inspect
from collections.abc import Callable, Generator
from typing import (  # type: ignore[attr-defined]
    TYPE_CHECKING,
    Annotated,
    Any,
    get_args,
    get_origin,
)

from pydantic import BaseModel, ConfigDict

from noob.introspection import is_optional, is_union
from noob.types import JsonStringable

if TYPE_CHECKING:
    from noob.node.spec import NodeSpecification


class Slot(BaseModel):
    name: str
    annotation: JsonStringable[Any]
    required: bool = True

    @classmethod
    def from_callable(
        cls, func: Callable, spec: NodeSpecification | None = None
    ) -> dict[str, Slot]:
        slots = {}
        sig = inspect.signature(func)

        for i, (name, param) in enumerate(sig.parameters.items()):
            if i == 0 and name == "self":
                continue
            if param.kind == inspect.Parameter.VAR_KEYWORD and spec is not None and spec.depends:
                # **kwargs - get slots from spec dependencies
                for dep in spec.depends:
                    if isinstance(dep, str):
                        # positional-only deps don't get attributed to kwargs
                        continue
                    name = next(iter(dep.keys()))
                    if name in slots:
                        # already defined by the signature, don't override
                        continue
                    slots[name] = Slot(name=name, annotation=param.annotation, required=False)
            else:
                slots[name] = Slot(
                    name=name,
                    annotation=param.annotation,
                    required=not (is_optional(param.annotation) and param.default is None),
                )
        return slots


class Signal(BaseModel):
    name: str
    annotation: JsonStringable[Any]

    # Unable to generate pydantic-core schema for <class 'types.UnionType'>
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_callable(
        cls, func: Callable, spec: NodeSpecification | None = None
    ) -> dict[str, Signal]:
        signals = {}
        return_annotation = inspect.signature(func).return_annotation
        for name, type_ in cls._collect_signal_names(return_annotation):
            signals[name] = Signal(name=name, annotation=type_)
        if not signals:
            signals["value"] = Signal(name="value", annotation=Any)
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


class Edge(BaseModel):
    """
    Directed connection between an output slot a node and an input slot in another node
    """

    source_node: str
    source_signal: str
    target_node: str
    target_slot: str | int | None = None
    """
    - For kwargs, target_slot is the name of the kwarg that the value is passed to.
    - For positional arguments, target_slot is an integer that indicates the index of the arg
    - For scalar arguments, target slot is None 
    """
    required: bool = True
