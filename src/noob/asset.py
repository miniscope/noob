import inspect
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from noob.node.base import PWrap, TOutput
from noob.types import AbsoluteIdentifier, PythonIdentifier
from noob.utils import resolve_python_identifier

ScopeType = Literal["function", "class", "module", "package", "session"]
"""Defines at which scale the resource should be locked."""


class AssetSpecification(BaseModel):
    id: PythonIdentifier
    type_: AbsoluteIdentifier = Field(..., alias="type")
    scope: ScopeType
    params: dict | None = None
    depends: list[AbsoluteIdentifier] | None = None


class Asset(BaseModel):
    id: PythonIdentifier
    spec: AssetSpecification
    scope: ScopeType
    params: dict[str, Any] = Field(default_factory=dict)

    obj: type | None = None

    model_config = ConfigDict(extra="forbid")

    @abstractmethod
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
                return WrapClassAsset(id=spec.id, cls=obj, spec=spec, params=params, scope=scope)
        else:
            return WrapFuncAsset(id=spec.id, fn=obj, spec=spec, params=params, scope=scope)

    def update(self, obj: Any) -> None:
        self.obj = obj


class WrapClassAsset(Asset):
    cls: type

    def init(self) -> None:
        self.obj = self.cls(**self.params)

    def deinit(self) -> None:
        self.obj = None


class WrapFuncAsset(Asset):
    fn: Callable[PWrap, TOutput]

    def init(self) -> None:
        self.obj = self.fn(**self.params)

    def deinit(self) -> None:
        self.obj = None
