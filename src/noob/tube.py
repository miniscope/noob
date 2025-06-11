import sys
import warnings
from dataclasses import dataclass
from graphlib import TopologicalSorter
from importlib import resources
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Optional,
    Protocol,
)

from pydantic import BaseModel, Field, field_validator, model_validator

from noob.node import Edge, Node, NodeSpecification, Sink, Source, Transform
from noob.types import ConfigSource, PythonIdentifier
from noob.yaml import ConfigYAMLMixin

if sys.version_info < (3, 11) or sys.version_info < (3, 12):
    from typing import NotRequired, Self

    from typing_extensions import TypedDict
else:
    from typing import NotRequired, Self, TypedDict

if TYPE_CHECKING:
    from noob.runner import TubeRunner


class _RequiredNode(TypedDict):
    type: str
    """Node type (as determined by its :attr:`.Node.name` attr"""
    config: NotRequired[dict[str, Any]]
    """Require that config values must be set to these values"""


class TubeConfig(ConfigYAMLMixin):
    """
    Configuration for the nodes within a tube
    """

    required_nodes: ClassVar[dict[str, _RequiredNode] | None] = None
    """
    id: type mapping that a subclass can use to require a set of node types 
    with specific IDs be present
    """

    nodes: dict[str, NodeSpecification] = Field(default_factory=dict)
    """The nodes that this tube configures"""

    @model_validator(mode="after")
    def validate_required_nodes(self) -> Self:
        """Ensure required nodes are present, if any"""
        if self.required_nodes is not None:
            for id_, required in self.required_nodes.items():
                assert id_ in self.nodes, f"Required node id {id_} not in {self.nodes.keys()}"
                assert (
                    self.nodes[id_].type_ == required["type"]
                ), f"Node ID {id_} is not of type {required['type']}"
                if "config" in required:
                    for key, val in required["config"].items():
                        assert (
                            self.nodes[id_].config[key] == val
                        ), f"Required node {id_} must have config value {key} set to {val}, "
                        f"got {self.nodes[id_].config[key]} instead."
        return self

    @field_validator("nodes", mode="before")
    @classmethod
    def fill_node_ids(cls, value: dict[str, dict]) -> dict[str, dict]:
        """
        Roll down the `id` from the key in the `nodes` dictionary into the node config
        """
        assert isinstance(value, dict)
        for id, node in value.items():
            if "id" not in node:
                node["id"] = id
        return value

    # TODO: Implement these validators
    # @field_validator("nodes", mode="after")
    # @classmethod
    # def valid_passed_and_fill_keys(
    #   cls, value: dict[str, NodeSpecification]
    # ) -> dict[str, NodeSpecification]:
    #     """
    #     Passed and fill keys refer to values within the node's config type
    #     """
    #
    # @field_validator("nodes", mode="after")
    # @classmethod
    # def unique_passed_values(
    #   cls, value: dict[str, NodeSpecification]
    # ) -> dict[str, NodeSpecification]:
    #     """
    #     All passed values (
    #     """
    #
    # @field_validator("nodes", mode="after")
    # @classmethod
    # def fill_sources_present(
    #     cls, value: dict[str, NodeSpecification]
    #   # ) -> dict[str, NodeSpecification]:
    #     """
    #     Fill values refer to nodes that are present in the node graph
    #     """
    #
    # @field_validator("nodes", mode="after")
    # @classmethod
    # def fill_values_dotted(
    #     cls, value: dict[str, NodeSpecification]
    # ) -> dict[str, NodeSpecification]:
    #     """
    #     Fill values refer to a property or attribute of a node (i.e. have at least one dot)
    #     """


class Tube(BaseModel):
    """
    A graph of nodes transforming some input source(s) to some output sink(s)

    The Tube model is a container for a set of nodes that are fully instantiated
    (e.g. have their "passed" and "fill" keys processed) and connected.
    It does not handle running the tube -- that is handled by a TubeRunner.
    """

    nodes: dict[str, Node] = Field(default_factory=dict)
    """
    Dictionary mapping all nodes from their ID to the instantiated node.
    """
    edges: list[Edge] = Field(default_factory=list)
    """
    Edges connecting slots within nodes.

    The nodes within :attr:`.Edge.source_node` and :attr:`.Edge.target_node` must
    be the same objects as those in :attr:`.Tube.nodes`
    (i.e. ``edges[0].source_node is nodes[node_id]`` ).
    """

    @property
    def sources(self) -> dict[str, "Source"]:
        """All :class:`.Source` nodes in the processing graph"""
        return {k: v for k, v in self.nodes.items() if isinstance(v, Source)}

    @property
    def transforms(self) -> dict[str, "Transform"]:
        """All :class:`.Transform` s in the processing graph"""
        return {k: v for k, v in self.nodes.items() if isinstance(v, Transform)}

    @property
    def sinks(self) -> dict[str, "Sink"]:
        """All :class:`.Sink` nodes in the processing graph"""
        return {k: v for k, v in self.nodes.items() if isinstance(v, Sink)}

    def graph(self) -> TopologicalSorter:
        """
        Produce a :class:`.TopologicalSorter` based on the graph induced by
        :attr:`.Tube.nodes` and :attr:`.Tube.edges` that yields node ids
        """
        sorter = TopologicalSorter()
        for node_id, node in self.nodes.items():
            in_edges = [e.target_node.id for e in self.edges if e.target_node is node]
            sorter.add(node_id, *in_edges)
        return sorter

    def in_edges(self, node: Node | str) -> list[Edge]:
        """
        Edges going towards the given node (i.e. the node is the edge's ``target`` )

        Args:
            node (:class:`.Node`, str): Either a node or its id

        Returns:
            list[:class:`.Edge`]
        """
        if isinstance(node, Node):
            node = node.id
        return [e for e in self.edges if e.target_node.id == node]

    def out_edges(self, node: Node | str) -> list[Edge]:
        """
        Edges going away from the given node (i.e. the node is the edge's ``source`` )

        Args:
            node (:class:`.Node`, str): Either a node or its id

        Returns:
            list[:class:`.Edge`]
        """
        if isinstance(node, Node):
            node = node.id
        return [e for e in self.edges if e.source_node.id == node]

    @classmethod
    def from_config(cls, config: TubeConfig | ConfigSource) -> Self:
        """
        Instantiate a tube model from its configuration

        Args:
            config (TubeConfig): the tube config to instantiate
        """
        config = TubeConfig.from_any(config)

        nodes = cls._init_nodes(config)
        edges = cls._init_edges(nodes, config.nodes)

        return cls(nodes=nodes, edges=edges)

    @classmethod
    def _init_nodes(cls, config: TubeConfig) -> dict[PythonIdentifier, Node]:
        nodes = {spec.id: Node.from_specification(spec) for spec in config.nodes.values()}
        return nodes

    @classmethod
    def _init_edges(cls, nodes: dict[str, Node], spec: dict[str, NodeSpecification]) -> list[Edge]:
        warnings.warn("Implement me!", stacklevel=2)
        return []


class _ConfigProtocol(Protocol):
    """
    Abstract protocol type to specify that classes consuming the TubeMixin
    must have some config attribute that specifies a tube
    (without prescribing what that config object must be)
    """

    tube: TubeConfig | None = None


@dataclass(kw_only=True)
class TubeMixin:
    """Mixin for use with models that have tubes!"""

    config: _ConfigProtocol

    _tube: Tube | None = None
    _runner: Optional["TubeRunner"] = None

    @property
    def sources(self) -> dict[str, "Source"]:
        """Convenience method to access :attr:`.Tube.sources`"""
        return self.tube.sources

    @property
    def transforms(self) -> dict[str, "Transform"]:
        """Convenience method to access :attr:`.Tube.transforms`"""
        return self.tube.transforms

    @property
    def sinks(self) -> dict[str, "Sink"]:
        """Convenience method to access :attr:`.Tube.sinks`"""
        return self.tube.sinks

    @property
    def tube(self) -> Tube:
        """Instantiated Tube from tube config"""
        if self._tube is None:
            self._tube = Tube.from_config(self.config.tube)
        return self._tube

    @property
    def runner(self) -> "TubeRunner":
        """
        A :class:`.TubeRunner` that ... runs the :attr:`~.TubeMixin.tube` !

        By default, creates a :class:`.SynchronousRunner`,
        and this property should be overridden by subclasses that want to specialize
        runner instantiation.
        """
        from noob.runner import SynchronousRunner

        if self._runner is None:
            self._runner = SynchronousRunner(self.tube)
        return self._runner


class TubeClassicEdition:
    def __init__(self):
        print(str(self))

    def __str__(self) -> str:
        important = resources.files("noob") / "important.txt"
        important = important.read_text()
        return important
