from copy import deepcopy
from typing import Self

from pydantic import BaseModel, Field

from noob.asset import Asset, AssetScope, AssetSpecification
from noob.event import Event
from noob.node.base import Edge
from noob.types import DependencyIdentifier, PythonIdentifier


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

    @classmethod
    def from_specification(cls, specs: dict[str, AssetSpecification]) -> Self:
        """
        Instantiate a :class:`.State` model from its configuration

        Args:
            spec (dict[str, AssetSpecification]): the :class:`.State` config to instantiate
        """
        assets = {spec.id: Asset.from_specification(spec) for spec in specs.values()}
        dependencies = {
            spec.depends: spec.id for spec in specs.values() if spec.depends is not None
        }
        return cls(assets=assets, dependencies=dependencies)

    def init_assets(self, scope: AssetScope) -> None:
        """
        run :meth:`.Asset.init` for assets that correspond to the given scope.
        Usually means that :attr:`.Asset.obj` attribute gets populated.
        """
        for asset in self.assets.values():
            if asset.scope == scope:
                asset.init()

    def deinit_assets(self, scope: AssetScope) -> None:
        """
        run :meth:`.Asset.deinit` for assets that correspond to the given scope.
        Usually means that :attr:`.Asset.obj` attribute is cleared to `None`.
        """
        for asset in self.assets.values():
            if asset.scope == scope:
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
                if not asset.depends:
                    args[edge.target_slot] = asset.obj
                else:
                    if epoch == asset.depends_satisfied_epoch + 1:
                        # asset dependency is not satisfied yet
                        args[edge.target_slot] = asset.obj
                    elif epoch == asset.depends_satisfied_epoch:
                        # asset dependency has been satisfied,
                        # so we don't want to mutate asset anymore
                        # ideally i'd like to convert whatever this is
                        # to a copy-on-write object, but it doesn't seem possible.
                        # https://stackoverflow.com/questions/12359366/python-copy-on-write-behavior
                        args[edge.target_slot] = deepcopy(asset.obj)
                    else:
                        # wrong order of epoch.
                        # wait until the correct epoch call comes in.
                        return None

        return None if not args or all(val is None for val in args.values()) else args

    def update(self, events: list[Event]) -> None:
        """Update asset if asset depends on a node signal"""
        if self.dependencies:
            for event in events:
                signal = ".".join((event["node_id"], event["signal"]))
                asset_id = self.dependencies.get(signal)
                if asset_id is not None:
                    self.assets[asset_id].update(value=event["value"], epoch=event["epoch"])

    def clear(self) -> None:
        """
        Clear assets.
        """
        self.assets.clear()
