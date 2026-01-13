import inspect
from collections.abc import Callable
from enum import StrEnum
from typing import Any, Generic, ParamSpec, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from noob.types import AbsoluteIdentifier, PythonIdentifier
from noob.utils import resolve_python_identifier

TOutput = TypeVar("TOutput")
PWrap = ParamSpec("PWrap")


class AssetScope(StrEnum):
    tube = "tube"
    """
    Asset persists through the entire lifespan of the tube. 
    Can be modified and passed through different epochs.
    """
    process = "process"
    """
    Asset persists through the a single process call.
    Can be modified and passed through different nodes but are
    re-instantiated at the beginning of each epoch.
    """
    node = "node"
    """
    Asset is re-instantiated every node call.
    """


"""Defines at which scale the resource should be locked."""


class AssetSpecification(BaseModel):
    """
    Specification for a single asset within a tube .yaml file.
    """

    id: PythonIdentifier
    """The unique identifier of the asset"""
    type_: AbsoluteIdentifier = Field(..., alias="type")
    """The python path to the location of the asset
    e.g. package_name.module_name.ClassName
    """
    scope: AssetScope
    """The scope of the asset. See :class:`.AssetScope`"""
    params: dict | None = None
    """Initialization parameters"""
    depends: AbsoluteIdentifier | None = None
    """
    Roundtrip dependency. Should point to the last node in a given 
    epoch that manipulates the asset.
    
    May only be used with scope == "runner"
    
    Typically this is used with assets that are mutated by multiple nodes in a tube.
    In that case, nodes should use dependencies to structure the order of mutation:
    The first node that should have it should depend directly on the asset,
    and then it and each node should emit the asset
    so that the successive node can depend on that signal.
    The node signal that this asset specification depends on will be the version of the asset
    stored and used in the next processing epoch.
    """


class Asset(BaseModel):
    """An asset within a processing tube."""

    id: PythonIdentifier
    """The unique identifier of the asset"""
    spec: AssetSpecification
    """The specs of the asset. See :class:`.AssetSpecification`"""
    scope: AssetScope
    """The scope of the asset. See :class:`.AssetScope`"""
    params: dict[str, Any] = Field(default_factory=dict)
    """Initialization parameters"""

    obj: Any | None = None
    """Instantiated asset instance"""

    model_config = ConfigDict(extra="forbid")

    def init(self) -> None:
        """
        Initialize the asset instance.

        Default is a no-op.
        Subclasses do not need to override if they have no initialization logic.
        """
        pass

    def deinit(self) -> None:
        """
        Deinitialize the asset instance.

        Default is a no-op.
        Subclasses do not need to override if they have no deinit logic.
        """
        pass

    @classmethod
    def from_specification(cls, spec: "AssetSpecification") -> "Asset":
        """
        Create an asset from its spec

        - resolve the asset type
            - if a subclass of :class:`.Asset`, just instantiate it
            - if not a subclass :class:`.Asset`, wrap it in :class:`.WrapClassAsset`
            - if a function, wrap it in :class:`.WrapFuncAsset`
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


T = TypeVar("T")


class WrapClassAsset(Asset, Generic[T]):
    """
    Wrap a non-Asset class.

    Wrapping allows us to use arbitrary classes as Assets within noob. Initializes
    the inner class to hold the class instance as an asset object.

    After instantiating the outer wrapping class, instantiate the inner wrapped class
    using the `params` given to the outer wrapping class during :meth:`.init` .
    """

    cls: type
    obj: T | None = None

    def init(self) -> None:
        self.obj = self.cls(**self.params)

    def deinit(self) -> None:
        self.obj = None


class WrapFuncAsset(Asset):
    """
    Wrap a function to build an Asset.

    The function effectively takes the role of :meth:`.__init__`, with the outer wrapping class
    `params` being injected as function parameters. The output of the function becomes the
    asset object.
    """

    fn: Callable

    def init(self) -> None:
        self.obj = self.fn(**self.params)

    def deinit(self) -> None:
        self.obj = None
