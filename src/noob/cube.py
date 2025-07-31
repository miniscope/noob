from typing import Any, Self

from pydantic import BaseModel, Field, field_validator

from noob.asset import Asset, AssetSpecification
from noob.node.base import Edge, Signal
from noob.types import AssetRef, ConfigSource, PythonIdentifier
from noob.yaml import ConfigYAMLMixin


class CubeSpecification(ConfigYAMLMixin):
    assets: dict[str, AssetSpecification] = Field(default_factory=dict)
    """The resources that this cube configures"""

    @field_validator("assets", mode="before")
    @classmethod
    def fill_node_ids(cls, value: dict[str, dict]) -> dict[str, dict]:
        """
        Roll down the `id` from the key in the `assets` dictionary into the asset config
        """
        assert isinstance(value, dict)
        for id_, asset in value.items():
            if "id" not in asset:
                asset["id"] = id_
        return value


class Cube(BaseModel):  # or Pube
    assets: dict[PythonIdentifier, Asset] = Field(default_factory=dict)
    edges: list[Edge] = Field(default_factory=list)

    @classmethod
    def from_specification(cls, spec: CubeSpecification | ConfigSource) -> Self:
        """
        Instantiate a cube model from its configuration

        Args:
            spec (CubeSpecification): the cube config to instantiate
        """
        spec = CubeSpecification.from_any(spec)

        assets = cls._init_assets(spec)
        edges = cls._init_edges(spec.assets)

        return cls(assets=assets, edges=edges)

    @classmethod
    def _init_assets(cls, specs: CubeSpecification) -> dict[PythonIdentifier, Asset]:
        assets = {spec.id: Asset.from_specification(spec) for spec in specs.assets.values()}
        return assets

    @classmethod
    def _init_edges(cls, asset_spec: dict[str, AssetSpecification]) -> list[Edge]:
        edges = []

        dependencies = {id_: spec.depends for id_, spec in asset_spec.items() if spec.depends}
        for asset_id, slot_inputs in dependencies.items():
            for source in slot_inputs:
                source_node, source_signal = source.split(".")

                edges.append(
                    Edge(
                        source_node=source_node,
                        source_signal=source_signal,
                        target_node="assets",
                        target_slot=asset_id,
                        required=True,
                    )
                )

        return edges

    def add(self, signals: list[Signal], value: Any, node_id: str) -> Any:
        """
        Add the result of a :meth:`.Node.process` call to the asset cube.

        Split the Signal instance into separate :class:`.Asset` s.
        Returns asset values replaced with asset.id, leaving others intact.
        Does not support dynamic creation of assets unspecified in yaml specification.

        Args:
            signals (list[Signal]): Signals from which the value was emitted by
                a :meth:`.Node.process` call
            value (Any): Value emitted by a :meth:`.Node.process` call. Gets wrapped
                with a list in case the length of signals is 1. Otherwise, it's zipped
                with :signals:
        """
        if value is None:
            return

        values = [value] if len(signals) == 1 else value

        for idx, (signal, val) in enumerate(zip(signals, values)):
            edges = [
                e for e in self.edges if e.source_node == node_id and e.source_signal == signal.name
            ]
            for edge in edges:
                self.assets[edge.target_slot].update(val)
                values[idx] = AssetRef(id=edge.target_slot)

        return values if len(signals) > 1 else values[0]

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

    def gather(self, edges: list[Edge]) -> dict | None:
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
