from typing import Self

from pydantic import BaseModel, Field

from noob.asset import Asset, AssetScope, AssetSpecification
from noob.node.base import Edge
from noob.types import PythonIdentifier


class State(BaseModel):
    """
    A collection of assets storing objects that persist through iterations of the tube.
    The target demographics generally include database connections, large arrays and statistics
    that traverse multiple processes of the tube.

    The :class:`.State` model is a container for a set of assets that are fully instantiated.
    It does not handle processing the assets -- that is handled by a TubeRunner.
    """

    assets: dict[PythonIdentifier, Asset] = Field(default_factory=dict)

    @classmethod
    def from_specification(cls, specs: dict[str, AssetSpecification]) -> Self:
        """
        Instantiate a :class:`.State` model from its configuration

        Args:
            spec (dict[str, AssetSpecification]): the :class:`.State` config to instantiate
        """
        assets = {spec.id: Asset.from_specification(spec) for spec in specs.values()}
        return cls(assets=assets)

    def init_assets(self, scope: AssetScope) -> None:
        """
        run :method:`.Asset.init` for assets that correspond to the given scope.
        Usually means the :class:`.Asset`'s :attr:`.Asset.obj` attribute gets populated.
        """
        for asset in self.assets.values():
            if asset.scope == scope:
                asset.init()

    def deinit_assets(self, scope: AssetScope) -> None:
        """
        run :method:`.Asset.deinit` for assets that correspond to the given scope.
        Usually means the :class:`.Asset`'s :attr:`.Asset.obj` attribute is cleared to `None`.
        """
        for asset in self.assets.values():
            if asset.scope == scope:
                asset.deinit()

    def get(self, signal: str) -> Asset | None:
        """
        Get the event with the matching node_id and signal name

        Returns the most recent matching event, as for now we assume that
        each combination of `node_id` and `signal` is emitted only once per processing cycle,
        and we assume processing cycles are independent (and thus our events are cleared)

        ``None`` in the case that the event has not been emitted
        """
        asset = [val for key, val in self.assets.items() if key == signal]
        return None if len(asset) == 0 else asset[-1]

    def collect(self, edges: list[Edge]) -> dict | None:
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
                asset = self.get(edge.source_signal)
                obj = None if asset is None else asset.obj
                args[edge.target_slot] = obj

        return None if not args or all(val is None for val in args.values()) else args

    def clear(self) -> None:
        """
        Clear assets.
        """
        self.assets.clear()
