from collections import defaultdict
from importlib import resources
from typing import Self

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from noob.asset import AssetSpecification
from noob.exceptions import InputMissingError
from noob.input import InputCollection, InputScope, InputSpecification
from noob.node import Edge, Node, NodeSpecification, Return
from noob.scheduler import Scheduler
from noob.state import State, StateSpecification
from noob.types import ConfigSource, PythonIdentifier
from noob.yaml import ConfigYAMLMixin


class TubeSpecification(ConfigYAMLMixin):
    """
    Configuration for the nodes within a tube.

    Representation of the yaml-form of a tube.
    Converted to the runtime-form with :meth:`.Tube.from_specification`.

    Not much, if any validation is performed here on the whole tube except
    that the nodes have the correct fields, ignoring validity of
    e.g. type mismatches or dependencies on signals that don't exist.
    Those require importing and introspecting the specified node classes,
    which should only happen when we try and instantiate the tube -
    this class is just a carrier for the yaml spec.
    """

    assets: dict[PythonIdentifier, AssetSpecification] = Field(default_factory=dict)
    """The specs of the assets that comprise the state of this tube"""

    input: dict[PythonIdentifier, InputSpecification] = Field(default_factory=dict)
    """Inputs provided at runtime"""

    nodes: dict[PythonIdentifier, NodeSpecification] = Field(default_factory=dict)
    """The nodes that this tube configures"""

    @field_validator("nodes", "assets", "input", mode="before")
    @classmethod
    def fill_node_ids(cls, value: dict[str, dict]) -> dict[str, dict]:
        """
        Roll down the `id` from the key in the `nodes` dictionary into the node config
        """
        assert isinstance(value, dict)
        for id_, node in value.items():
            if "id" not in node:
                node["id"] = id_
        return value

    @field_validator("nodes", mode="after")
    @classmethod
    def only_one_return(
        cls, value: dict[PythonIdentifier, NodeSpecification]
    ) -> dict[PythonIdentifier, NodeSpecification]:
        return_nodes = set(node_id for node_id, node in value.items() if node.type_ == "return")
        if len(return_nodes) > 1:
            raise ValueError(
                f"Only one return node is allowed per tube, instead got {return_nodes}"
            )
        return value


class Tube(BaseModel):
    """
    A graph of nodes transforming some input source(s) to some output sink(s)

    The Tube model is a container for a set of nodes that are fully instantiated
    (e.g. have their "passed" and "fill" keys processed) and connected.
    It does not handle running the tube -- that is handled by a TubeRunner.
    """

    tube_id: str
    """
    Locally unique identifier for this tube
    """
    nodes: dict[str, Node]
    """
    Dictionary mapping all nodes from their ID to the instantiated node.
    """
    edges: list[Edge]
    """
    Edges connecting slots within nodes.

    The nodes within :attr:`.Edge.source_node` and :attr:`.Edge.target_node` must
    be the same objects as those in :attr:`.Tube.nodes`
    (i.e. ``edges[0].source_node is nodes[node_id]`` ).
    """
    input: dict = Field(default_factory=dict)
    """
    tube-scoped inputs provided to nodes as parameters
    """
    input_collection: InputCollection = Field(default_factory=InputCollection)
    """
    Specifications declared by the tube to be supplied 
    """

    state: State = Field(default_factory=State)

    scheduler: Scheduler = None

    _enabled_nodes: dict[str, Node] | None = None

    @field_validator("scheduler", mode="before")
    @classmethod
    def _create_scheduler(cls, value: Scheduler | None, info: ValidationInfo) -> Scheduler:
        if value is None:
            scheduler = cls._init_scheduler(info.data["nodes"], info.data["edges"])
        else:
            scheduler = value
        return scheduler

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
        return [e for e in self.edges if e.target_node == node]

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
        return [e for e in self.edges if e.source_node == node]

    @classmethod
    def from_specification(
        cls, spec: TubeSpecification | ConfigSource, input: dict | None = None
    ) -> Self:
        """
        Instantiate a tube model from its configuration

        Args:
            spec (TubeSpecification): the tube config to instantiate

        Raises:
            InputMissingError if requested tube-scoped input is not present.
        """
        if input is None:
            input = {}

        spec = TubeSpecification.from_any(spec)

        # check inputs first to avoid doing work if the input is invalid
        input_collection = cls._init_inputs(spec)
        # adding the input validates presence of required inputs
        input_collection.add_input(InputScope.tube, input)

        nodes = cls._init_nodes(spec, input_collection)
        edges = cls._init_edges(spec.nodes, nodes)
        state = cls._init_state(spec.assets)
        scheduler = cls._init_scheduler(spec.nodes, edges)

        return cls.model_validate(
            {
                "tube_id": spec.noob_id,
                "nodes": nodes,
                "edges": edges,
                "state": state,
                "scheduler": scheduler,
                "input": input,
                "input_collection": input_collection,
            },
            context={"skip_input_presence": True},
        )

    @classmethod
    def _init_state(cls, spec: dict[str, AssetSpecification]) -> State:
        state_spec = StateSpecification(assets=spec)
        return State.from_specification(state_spec)

    @classmethod
    def _init_nodes(
        cls, specs: TubeSpecification, input_collection: InputCollection
    ) -> dict[PythonIdentifier, Node]:
        nodes = {
            spec.id: Node.from_specification(spec, input_collection)
            for spec in specs.nodes.values()
        }
        return nodes

    @classmethod
    def _init_edges(
        cls, node_spec: dict[str, NodeSpecification], nodes: dict[str, Node]
    ) -> list[Edge]:
        edges = []
        for node in nodes.values():
            edges.extend(node.edges)
        return edges

    @classmethod
    def _init_scheduler(
        cls, nodes: dict[str, NodeSpecification | Node], edges: list[Edge]
    ) -> Scheduler:
        node_specs = {
            id_: node if isinstance(node, NodeSpecification) else node.spec
            for id_, node in nodes.items()
        }
        return Scheduler.from_specification(node_specs, edges)

    @classmethod
    def _init_inputs(cls, spec: TubeSpecification) -> InputCollection:
        specs = defaultdict(dict)
        for input_spec in spec.input.values():
            specs[input_spec.scope][input_spec.id] = input_spec
        return InputCollection(specs=specs)

    @property
    def enabled_nodes(self) -> dict[str, Node]:
        """
        Produce nodes that have :attr:`.Node.enabled` set to `True`.
        """
        if self._enabled_nodes is None:
            self._enabled_nodes = {k: v for k, v in self.nodes.items() if v.enabled}
        return self._enabled_nodes

    def enable_node(self, node_id: str) -> None:
        self.nodes[node_id].enabled = True
        self.scheduler.enable_node(node_id)
        self._enabled_nodes = None  # Trigger recalculation in the next enabled_nodes call

    def disable_node(self, node_id: str) -> None:
        self.nodes[node_id].enabled = False
        self.scheduler.disable_node(node_id)
        self._enabled_nodes = None  # Trigger recalculation in the next enabled_nodes call

    @model_validator(mode="after")
    def input_present(self, info: ValidationInfo) -> Self:
        """
        Validate the presence of required tube-scoped inputs.

        Though tubes are usually instantiated from specs,
        this catches the case when they are constructed directly.
        """
        if info.context and info.context.get("skip_input_presence", False):
            return self

        try:
            self.input_collection.add_input(InputScope.tube, self.input)
        except InputMissingError as e:
            raise ValidationError("Required input missing") from e

        return self

    @field_validator("nodes", mode="after")
    def only_one_return(cls, value: dict[str, Node]) -> dict[str, Node]:
        """
        Only one return node is allowed.

        Validate both here and in the spec because tubes can be created programmatically
        """
        return_nodes = set({node_id for node_id, node in value.items() if isinstance(node, Return)})
        if len(return_nodes) > 1:
            raise ValueError(
                f"Only one return node is allowed in a tube, instead got {return_nodes}"
            )
        return value


class TubeClassicEdition:
    def __init__(self):
        print(str(self))

    def __str__(self) -> str:
        important = resources.files("noob") / "important.txt"
        important = important.read_text()
        return important
