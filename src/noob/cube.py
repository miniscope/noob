from typing import Self

from pydantic import BaseModel, Field, field_validator

from noob.asset import Asset, AssetSpecification
from noob.types import ConfigSource, PythonIdentifier
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

    @classmethod
    def from_specification(cls, spec: CubeSpecification | ConfigSource) -> Self:
        """
        Instantiate a cube model from its configuration

        Args:
            spec (CubeSpecification): the cube config to instantiate
        """
        spec = CubeSpecification.from_any(spec)

        assets = cls._init_assets(spec)

        return cls(assets=assets)

    @classmethod
    def _init_assets(cls, specs: CubeSpecification) -> dict[PythonIdentifier, Asset]:
        assets = {spec.id: Asset.from_specification(spec) for spec in specs.nodes.values()}
        return assets