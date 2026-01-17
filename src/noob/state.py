from collections import defaultdict
from typing import Self, TypeAlias, TypedDict

from pydantic import BaseModel, Field

from noob.asset import Asset, AssetScope, AssetSpecification
from noob.event import Event
from noob.node.base import Edge
from noob.types import NodeID, PythonIdentifier


class _AssetDependency(TypedDict):
    asset_id: PythonIdentifier
    signal: PythonIdentifier


_DependencyMap: TypeAlias = dict[NodeID, _AssetDependency]


class State(BaseModel):
    """
    A collection of assets storing objects that persist through iterations of the tube.
    The target demographics generally include database connections, large arrays and statistics
    that traverse multiple processes of the tube.

    The :class:`.State` model is a container for a set of assets that are fully instantiated.
    It does not handle processing the assets -- that is handled by a TubeRunner.
    """

    assets: dict[PythonIdentifier, Asset] = Field(default_factory=dict)
    dependencies: _DependencyMap = Field(default_factory=dict)
    """
    Map from node signals that assets depend on to the asset and signal ids. 
    See :attr:`.AssetSpecification.depends` . 
    
    Only those dependencies that require copying are included here
    (assets which are not used after the node that is depended on emits them
    don't need to be copied to protect against mutation within the same epoch
    after they are stored).
    """
    scope_to_assets: dict[AssetScope, list[Asset]] = Field(
        default_factory=lambda: defaultdict(list)  # type: ignore[arg-type]
    )
    """
    Map from :class:`.AssetScope` to :class:`.Asset` to circumvent
    querying scope for each asset in :meth:`.State.init_assets` and :meth:`.State.deinit_assets`
    """

    @classmethod
    def from_specification(
        cls, specs: dict[str, AssetSpecification], edges: list[Edge] | None = None
    ) -> Self:
        """
        Instantiate a :class:`.State` model from its configuration

        Args:
            spec (dict[str, AssetSpecification]): the :class:`.State` config to instantiate
            edges (list[Edge] | None): If present, edges for the whole graph,
                used to reduce copying for assets using dependencies to store values between epochs.
                If there are no other nodes that depend on the value that the asset depends on,
                then we don't have to copy.
        """

        assets = {spec.id: Asset.from_specification(spec) for spec in specs.values()}
        dependencies = cls._get_dependencies(specs, edges)
        scope_to_assets = defaultdict(list)
        for asset in assets.values():
            scope_to_assets[asset.scope].append(asset)
        return cls(
            assets=assets,
            dependencies=dependencies,
            scope_to_assets=scope_to_assets,
        )

    def init_assets(self, scope: AssetScope) -> None:
        """
        run :meth:`.Asset.init` for assets that correspond to the given scope.
        Usually means that :attr:`.Asset.obj` attribute gets populated.
        """
        for asset in self.scope_to_assets.get(scope, []):
            asset.init()

    def deinit_assets(self, scope: AssetScope) -> None:
        """
        run :meth:`.Asset.deinit` for assets that correspond to the given scope.
        Usually means that :attr:`.Asset.obj` attribute is cleared to `None`.
        """
        for asset in self.scope_to_assets.get(scope, []):
            asset.deinit()

    def collect(self, edges: list[Edge], epoch: int) -> dict | None:
        """
        Gather events into a form that can be consumed by a :meth:`.Node.process` method,
        given the collection of inbound edges (usually from :meth:`.Tube.in_edges` ).

        If none of the requested events have been emitted, return ``None``.

        If all of the requested events have been emitted, return a kwarg-like dict

        If some of the requested events are missing but others are present,
        return ``None`` for any missing events.

        .. todo::

            Add an example

        """
        args = {}
        for edge in edges:
            if edge.source_node == "assets":
                assert edge.source_signal is not None, (
                    "Must set signal name when depending on an asset "
                    "(assets have no generic 'value' signal)"
                )
                asset = self.assets[edge.source_signal]
                if (
                    not asset.depends
                    or asset.depends.split(".")[0] not in self.dependencies
                    or epoch == asset.stored_at + 1
                ):
                    args[edge.target_slot] = asset.obj
                else:
                    raise ValueError(
                        f"Asset not ready to emit for epoch {epoch}: "
                        f"asset was last stored at epoch {asset.stored_at}."
                    )

        return None if not args or all(val is None for val in args.values()) else args

    def update(self, events: list[Event]) -> None:
        """Update asset if asset depends on a node signal"""
        for event in events:
            if (dep := self.dependencies.get(event["node_id"])) and dep["signal"] == event[
                "signal"
            ]:
                self.assets[dep["asset_id"]].update(value=event["value"], epoch=event["epoch"])

    def clear(self) -> None:
        """
        Clear assets.
        """
        self.assets.clear()

    @classmethod
    def _get_dependencies(
        cls, specs: dict[str, AssetSpecification], edges: list[Edge] | None = None
    ) -> _DependencyMap:
        deps = {}
        for asset in specs.values():
            if not asset.depends:
                continue
            node_id, signal = asset.depends.split(".")
            if edges and not any(
                edge.source_node == node_id and edge.source_signal == signal for edge in edges
            ):
                continue
            deps[node_id] = _AssetDependency(asset_id=asset.id, signal=signal)
        return deps
