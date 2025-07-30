import inspect
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from noob.node.base import PWrap, TOutput
from noob.types import AbsoluteIdentifier
from noob.utils import resolve_python_identifier

ScopeType = Literal["function", "class", "module", "package", "session"]
"""Defines at which scale the resource should be locked."""


class AssetSpecification(BaseModel):
    id: str
    type_: AbsoluteIdentifier = Field(..., alias="type")
    scope: ScopeType
    params: dict | None = None  # or config?


class Asset(BaseModel):
    """
    Each asset needs...
    exposure (scope) system: which node can connect
    processes: resources may provide more than one way to approach the data
    data paths: slots (data going in), signals (data coming out)
    a lifecycle: init, deinit
    the actual entity is wrapped to unify these interfaces
    """

    id: str
    spec: AssetSpecification
    scope: ScopeType

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
    def push(self, *args: Any, **kwargs: Any) -> Any | None:
        """Push some input. See subclasses for details"""
        pass

    @abstractmethod
    def pull(self, *args: Any, **kwargs: Any) -> Any | None:
        """Pull some input. See subclasses for details"""
        pass

    @classmethod
    def from_specification(cls, spec: "AssetSpecification") -> "Asset":
        """
        Create a asset from its spec

        - resolve the asset type
        - if a function, wrap it in a node class -> not sure if this is applicable for assets
          - i guess functions can contain class instances (i.e. Generator)
        - if a subclass, just instantiate it
        - if not a subclass, wrap it in WrapAsset
        """
        obj = resolve_python_identifier(spec.type_)

        params = spec.params if spec.params is not None else {}
        scope = spec.scope

        # check if function by checking if callable -
        # Node classes do not have __call__ defined and thus should not be callable
        if inspect.isclass(obj):
            if issubclass(obj, Asset):
                return obj(id=spec.id, spec=spec, **params)
            else:
                return WrapClassAsset(id=spec.id, obj=obj, spec=spec, params=params, scope=scope)
        else:
            return WrapFuncAsset(id=spec.id, fn=obj, spec=spec, params=params, scope=scope)


class WrapClassAsset(Asset):
    obj: type
    params: dict[str, Any] = Field(default_factory=dict)
    instance: type | None = None


class WrapFuncAsset(Asset):
    fn: Callable[PWrap, TOutput]
    params: dict[str, Any] = Field(default_factory=dict)
    instance: type | None = None
