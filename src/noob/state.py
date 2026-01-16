from typing import Self

from mypy.checkexpr import defaultdict
from pydantic import BaseModel, Field

from noob.asset import Asset, AssetScope, AssetSpecification
from noob.event import Event
from noob.node.base import Edge
from noob.types import PythonIdentifier, DependencyIdentifier


class State(BaseModel):
    """
    A collection of assets storing objects that persist through iterations of the tube.
    The target demographics generally include database connections, large arrays and statistics
    that traverse multiple processes of the tube.

    The :class:`.State` model is a container for a set of assets that are fully instantiated.
    It does not handle processing the assets -- that is handled by a TubeRunner.
    """

    assets: dict[PythonIdentifier, Asset] = Field(default_factory=dict)
    dependencies: dict[DependencyIdentifier, PythonIdentifier] = Field(default_factory=dict)
    """
    Map from node signals that assets depend on to the asset ids. 
    See :attr:`.AssetSpecification.depends`
    """
    scope_to_assets: dict[AssetScope, list[Asset]] = Field(default_factory=defaultdict(list))
    """
    Map from :class:`.AssetScope` to :class:`.Asset` to circumvent
    querying scope for each asset in :meth:`.State.init_assets` and :meth:`.State.deinit_assets`
    """
    need_copy: dict[PythonIdentifier, dict[PythonIdentifier, bool]] = Field(default_factory=dict)
    """    
    Assets that are used after by the downstream generation nodes
    after their dependencies for a given epoch have been satisfied.
    """

    @classmethod
    def from_specification(cls, specs: dict[str, AssetSpecification], edges: list[Edge]) -> Self:
        """
        Instantiate a :class:`.State` model from its configuration

        Args:
            spec (dict[str, AssetSpecification]): the :class:`.State` config to instantiate
        """
        assets = {spec.id: Asset.from_specification(spec) for spec in specs.values()}
        asset_dependencies = {
            spec.depends: spec.id for spec in specs.values() if spec.depends is not None
        }
        node_dependencies = {".".join((e.source_node, e.source_signal)) for e in edges}
        need_copy = {}
        for signal, asset in asset_dependencies.items():
            need_copy[signal.split(".")[0]] = {signal.split(".")[1]: signal in node_dependencies}
        scope_to_assets = defaultdict(list)
        for asset in assets.values():
            scope_to_assets[asset.scope].append(asset)
        return cls(
            assets=assets,
            dependencies=asset_dependencies,
            scope_to_assets=scope_to_assets,
            need_copy=need_copy,
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
                if not asset.depends or epoch == asset.stored_at + 1:
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
            asset_id = self.dependencies.get(".".join((event["node_id"], event["signal"])))
            if asset_id is not None:
                self.assets[asset_id].update(value=event["value"], epoch=event["epoch"])

    def clear(self) -> None:
        """
        Clear assets.
        """
        self.assets.clear()
