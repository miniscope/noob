from collections import defaultdict
from collections.abc import Mapping
from importlib import resources
from itertools import permutations
from typing import Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from ruamel.yaml import CommentedMap

from noob.asset import AssetSpecification
from noob.exceptions import InputMissingError
from noob.input import InputCollection, InputScope, InputSpecification
from noob.node import Edge, Node, NodeSpecification, Return
from noob.scheduler import Scheduler
from noob.state import State
from noob.types import ConfigSource, NodeSignal, PythonIdentifier
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

    extends: list[ConfigSource] | None = None
    """
    Other tubes that this tube extends.
    
    All the node and other specifications from the other tubes are collapsed into this one
    such that later tubes in the list override earlier tubes
    (and the current tube overrides all the extended tubes).
    
    Extensions are neither completely deep nor completely shallow.
    In general, items are merged as deeply as they can be, with some exceptions:
    
    - depends: for specs with matching IDs, more recent declarations fully replace earlier ones.
      This is to facilitate extending tubes being able to *remove* dependencies - 
      rather than having a specific "remove" keyword, to remove a dependency,
      copy all but the one to be removed. 
      To add a dependency, copy all the previous ones along with the one ones
    - return: only one return node can exist in a tube, so the id given to the most recent return
      is used and all other returns are removed.
    
    In tubes that extend other tubes, fields that are typically required are treated as optional
    in the yaml spec in order to avoid needing to redefine evey property.
    For example, if a tube wants to just change a node's params from an extended tube,
    the extending tube needs to only declare the node's ID and params
    without needing to declare the type again.
    """

    description: str | None = None
    """An optional description of the tube"""

    model_config = ConfigDict(extra="forbid")

    @field_validator("nodes", "assets", "input", mode="before")
    @classmethod
    def fill_node_ids(cls, value: dict[str, dict]) -> dict[str, dict]:
        """
        Roll down the `id` from the key in the `nodes` dictionary into the node config
        """
        assert isinstance(value, dict)
        for id_, node in value.items():
            if isinstance(node, dict):
                if "id" not in node:
                    node["id"] = id_
            else:
                assert isinstance(node, AssetSpecification | InputSpecification | NodeSpecification)
                assert node.id == id_, (
                    f"Mismatch between id used as key in node/input/asset dict ({id_}) "
                    f"and id on already-instantiated object ({node.id})"
                )

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

    @model_validator(mode="after")
    def dependencies_exist(self) -> Self:
        """
        Nodes referred to in dependencies should exist, they must be
        - other nodes in the spec
        - input (and the input signal must exist)
        - assets (and the asset must exist)

        Note: we *do not* check the validity of signals here,
        as that would require us to import/inspect the code object referred to by the node,
        and we want specifications to be loadable even if the referred code is not
        present and installed in the environment.
        """
        for node_id, node in self.nodes.items():
            if node.depends is None:
                continue
            deps = [node.depends] if isinstance(node.depends, str) else node.depends
            for dep in deps:
                dep_str = next(iter(dep.values())) if isinstance(dep, dict) else dep
                # this is safe, NodeSpec already validates the string is a node_id.signal str
                dep_node_id, signal = dep_str.split(".")
                if dep_node_id == "assets":
                    assert (
                        signal in self.assets
                    ), f"Node {node_id} depends on asset {signal}, which does not exist"
                elif dep_node_id == "input":
                    assert (
                        signal in self.input
                    ), f"Node {node_id} depends on input {signal}, which does not exist"
                else:
                    assert (
                        dep_node_id in self.nodes
                    ), f"Node {node_id} depends on node {dep_node_id}, which does not exist"
        return self

    @classmethod
    def _post_load_yaml(cls, config: CommentedMap) -> Mapping:
        """Handle extensions!"""
        if (extends := config.get("extends")) and isinstance(extends, list):
            # induction step - materialize the extension tree of the depth-1 extension layer
            merge_stack: list[TubeSpecification | Mapping] = [
                *(TubeSpecification.from_any(e) for e in extends),
                config,
            ]
            # merge depth-1 specs
            new_config = merge_stack.pop(0)
            for extended in merge_stack:
                new_config = merge_tube_specs(new_config, extended)
            new_config = cast(Mapping, new_config)
            return new_config
        return config


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
    spec: TubeSpecification | None = None

    state: State = Field(default_factory=State)

    scheduler: Scheduler = None  # type: ignore[assignment]

    _enabled_nodes: dict[str, Node] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("scheduler", mode="before")
    @classmethod
    def _create_scheduler(cls, value: Scheduler | None, info: ValidationInfo) -> Scheduler:
        if value is None:
            scheduler = cls._init_scheduler(info.data["nodes"], info.data["edges"])
        else:
            scheduler = value

        assert not scheduler.has_cycle()

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
        scheduler = cls._init_scheduler(spec.nodes, edges)

        state = cls._init_state(spec, edges)

        return cls.model_validate(
            {
                "tube_id": spec.noob_id,
                "nodes": nodes,
                "edges": edges,
                "state": state,
                "scheduler": scheduler,
                "input": input,
                "input_collection": input_collection,
                "spec": spec,
            },
            context={"skip_input_presence": True},
        )

    @property
    def has_return(self) -> bool:
        """Whether the tube has a :class:`.Return` node present"""
        return any(isinstance(node, Return) for node in self.nodes.values())

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
    def _init_scheduler(cls, nodes: dict[str, NodeSpecification], edges: list[Edge]) -> Scheduler:
        node_specs = {id_: node for id_, node in nodes.items()}
        return Scheduler.from_specification(node_specs, edges)

    @classmethod
    def _init_state(cls, spec: TubeSpecification, edges: list[Edge]) -> State:
        return State.from_specification(specs=spec.assets, edges=edges)

    @classmethod
    def _init_inputs(cls, spec: TubeSpecification) -> InputCollection:
        specs: dict[InputScope, dict[PythonIdentifier, InputSpecification]] = defaultdict(dict)
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

    @model_validator(mode="after")
    def assets_exhausted_after_storage(self) -> Self:
        """
        Runner-scoped assets that depend on node outputs can't be directly depended on
        in topological generations during or after when they are stored
        in order to protect against unstructured mutation.
        """
        generations = None
        for asset in self.state.assets.values():
            if not asset.depends:
                continue

            # only compute if we actually have dependencies
            if generations is None:
                generations = self.scheduler.generations()

            # find generation that the node that returns to us is in + successor generations
            depended_node = asset.depends.split(".")[0]
            successors: list[str] = []
            for generation in generations:
                if depended_node in generation or successors:

                    successors.extend(
                        [
                            node
                            for node in generation
                            if node != depended_node and not isinstance(node, NodeSignal)
                        ]
                    )

            # assert no successor nodes depend on the asset directly.
            for successor in successors:
                assert not any(
                    edge.source_node == "assets" and edge.source_signal == asset.id
                    for edge in self.nodes[successor].edges
                ), (
                    "Nodes that run at the same time or after a node that an asset depends on "
                    "to update its value cannot depend on the asset directly. "
                    "Access it by emitting it from another node and depending on that signal.\n"
                    f"Node {successor} cannot depend on assets.{asset.id}, "
                    f"because {asset.id} depends on {depended_node}, and tube has topo generations:"
                    f"\n{generations}"
                )

        return self

    @model_validator(mode="after")
    def no_cross_map_dependencies(self) -> Self:
        """
        A node can only be downstream of one linear chain of map operations.

        Fine:

        * a -> map_b -> c
        * a -> map_b -> map_c -> d

        Not fine:

        * a -> map_a -> c; b -> map_b -> c

        """
        from noob.node.map import Map

        shadows = {node_id: downstream_nodes(self.edges, node_id) for node_id in self.nodes}
        map_nodes = set(node_id for node_id, node in self.nodes.items() if isinstance(node, Map))
        for map_a, map_b in permutations(map_nodes, 2):
            if map_b in shadows[map_a] or map_a in shadows[map_b]:
                continue
            intersection = (shadows[map_b] - {map_b}).intersection(shadows[map_a] - {map_a})
            assert not intersection, (
                f"Nodes can only be in the downstream shadow of a single map operation, "
                f"or a nested chain of map operations. "
                f"{intersection} are downstream of unrelated map nodes {map_a} and {map_b}"
            )
        return self


def downstream_nodes(edges: list[Edge], node_id: str, exclude: set[str] | None = None) -> set[str]:
    """
    The set of nodes that through some chain of dependencies depend on the given node_id

    Args:
        exclude (set[str]): Nodes (and their successors) to exclude from downstream graph
    """
    if exclude is None:
        exclude = set()

    # Build adjacency list: source -> targets
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.source_node].append(edge.target_node)

    # BFS from node_id
    downstream = {node_id}
    queue = [node_id]
    while queue:
        current = queue.pop(0)
        for successor in adjacency.get(current, []):
            if successor not in downstream and successor not in exclude:
                downstream.add(successor)
                queue.append(successor)
    return downstream


def merge_tube_specs(
    left: Mapping | TubeSpecification, right: Mapping | TubeSpecification
) -> Mapping:
    """
    Merge the yaml/json dict form of a tube spec pre-casting to :class:`.TubeSpecification`
    while resolving ``extends``.

    See :attr:`.TubeSpecification.extends` for details of merge behavior.
    """
    left, right = _dump_spec(left), _dump_spec(right)
    left, right = _merge_returns(left, right)
    left, right = _merge_nodes(left, right)
    merged = _merge_dicts(left, right)
    return merged


def _merge_nodes(left: dict, right: dict) -> tuple[dict, dict]:
    """
    Merge all nodes in a pair of tube specs
    have to have this additional wrapper in case someone wants to name a node "depends"
    for some godforsaken reason.
    """
    if "nodes" not in left or "nodes" not in right:
        return left, right
    node_ids = list(dict.fromkeys(list(left["nodes"].keys()) + list(right["nodes"].keys())))
    right["nodes"] = {
        k: _merge_node(left["nodes"].get(k, {}), right["nodes"].get(k, {})) for k in node_ids
    }

    # rm the left's nodes dict by creating a new one so the merged copy in the right is used
    left = {k: v for k, v in left.items() if k != "nodes"}
    return left, right


def _merge_node(left: dict, right: dict) -> dict:
    """
    Merge a pair of same-id nodes, preserving the rightmost depends dict.
    left as a placeholder in case there is further refinement of node merging needed.
    """
    return _merge_dicts(left, right, force_right={"depends"})


def _merge_returns(left: dict, right: dict) -> tuple[dict, dict]:
    # where left and right are full tube specs that we mutate
    # handle returns - merge, but keep the key of the right return node

    # we only need to do anything special if the left spec has a return node
    if left_return := {
        key: val for key, val in left["nodes"].items() if val.get("type", "") == "return"
    }:
        right_return = {
            key: val for key, val in right["nodes"].items() if val.get("type", "") == "return"
        }
        if len(left_return) > 1 or len(right_return) > 1:
            raise ValueError(
                f"Tubes can only have a single return node! "
                f"got left: {left_return}, right: {right_return}"
            )
        left_key, right_key = list(left_return)[0], list(right_return)[0]
        merged = _merge_node(left_return[left_key], right_return[right_key])
        left["nodes"] = {k: v for k, v in left["nodes"].items() if k != left_key}
        right["nodes"][right_key] = merged
    return left, right


def _merge_dicts(left: dict, right: dict, force_right: set | None = None) -> dict:
    # NOT a general deepmerge - recursively merge only dicts by key
    if force_right is None:
        force_right = set()

    # preserve left->right key order
    keys = list(dict.fromkeys(list(left.keys()) + list(right.keys())))

    merged = {}
    for key in keys:
        if key in left and key in right and key not in force_right:
            if isinstance(left[key], dict) and isinstance(right[key], dict):
                merged[key] = _merge_dicts(left[key], right[key])
            else:
                merged[key] = right[key]
        elif key in right:
            merged[key] = right[key]
        elif key in left:
            merged[key] = left[key]
    return merged


def _dump_spec(spec: TubeSpecification | Mapping) -> dict:
    if not isinstance(spec, TubeSpecification):
        return dict(spec)
    # dump spec while merging
    data = spec.model_dump(exclude_unset=True, exclude_defaults=True, by_alias=True)
    # materialized node specs have `id` fields filled in.
    # treat the keys in the dict as decisive
    for node in data["nodes"].values():
        if "id" in node:
            del node["id"]
    return data


class TubeClassicEdition:
    def __init__(self):
        print(str(self))

    def __str__(self) -> str:
        important = resources.files("noob") / "important.txt"
        return important.read_text()
