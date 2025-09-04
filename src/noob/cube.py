from typing import Any, Self

from pydantic import BaseModel, Field, field_validator

from noob.asset import Asset, AssetSpecification
from noob.node.base import Edge, Signal
from noob.types import AssetRef, ConfigSource, PythonIdentifier
from noob.yaml import ConfigYAMLMixin

class CubeSpecification(BaseModel):
    """
    Configuration for the assets within a cube.

    Representation of the yaml-form of a cube.
    Converted to the runtime-form with :meth:`.Cube.from_specification`.

    Not much, if any validation is performed here on the whole cube except
    that the assets have the correct fields, ignoring validity of
    e.g. type mismatches.
    Those require importing and introspecting the specified assets classes,
    which should only happen when we try and instantiate the cube -
    this class is just a carrier for the yaml spec.
    """

    assets: dict[str, AssetSpecification] = Field(default_factory=dict)
    """The assets that this cube configures"""


class Cube(BaseModel):  # or Pube
    """
    A collection of assets storing objects that persist through iterations of the tube.
    The target demographics generally include database connections, large arrays and statistics
    that traverse multiple processes of the tube.

    The Cube model is a container for a set of assets that are fully instantiated.
    It does not handle processing the assets -- that is handled by a TubeRunner.
    """

    assets: dict[PythonIdentifier, Asset] = Field(default_factory=dict)

    @classmethod
    def from_specification(cls, spec: CubeSpecification) -> Self:
        """
        Instantiate a cube model from its configuration

        Args:
            spec (CubeSpecification): the cube config to instantiate
        """
        assets = cls._init_assets(spec)

        return cls(assets=assets)

    @classmethod
    def _init_assets(cls, specs: CubeSpecification) -> dict[PythonIdentifier, Asset]:
        assets = {spec.id: Asset.from_specification(spec) for spec in specs.assets.values()}
        return assets

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
                asset = self.get(edge.source_signal)
                obj = None if asset is None else asset.obj
                args[edge.target_slot] = obj

        args = None if not args or all(val is None for val in args.values()) else args

        return args

    def clear(self) -> None:
        """
        Clear assets.
        """
        self.assets.clear()
