from collections.abc import Mapping
from graphlib import TopologicalSorter
from importlib import resources
from typing import (
    Self,
)

from pydantic import BaseModel, Field, field_validator

from noob.node import Edge, Node, NodeSpecification
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

    nodes: dict[str, NodeSpecification] = Field(default_factory=dict)
    """The nodes that this tube configures"""

    @field_validator("nodes", mode="before")
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

    def graph(self) -> TopologicalSorter:
        """
        Produce a :class:`.TopologicalSorter` based on the graph induced by
        :attr:`.Tube.nodes` and :attr:`.Tube.edges` that yields node ids
        """
        sorter = TopologicalSorter()
        for node_id in self.nodes:
            in_edges = [e.source_node for e in self.edges if e.target_node == node_id]
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
    def from_specification(cls, spec: TubeSpecification | ConfigSource) -> Self:
        """
        Instantiate a tube model from its configuration

        Args:
            spec (TubeSpecification): the tube config to instantiate
        """
        spec = TubeSpecification.from_any(spec)

        nodes = cls._init_nodes(spec)
        edges = cls._init_edges(spec.nodes)

        return cls(nodes=nodes, edges=edges)

    @classmethod
    def _init_nodes(cls, specs: TubeSpecification) -> dict[PythonIdentifier, Node]:
        nodes = {spec.id: Node.from_specification(spec) for spec in specs.nodes.values()}
        return nodes

    @classmethod
    def _init_edges(cls, node_spec: dict[str, NodeSpecification]) -> list[Edge]:
        edges = []

        dependency_map = {id_: spec.depends for id_, spec in node_spec.items() if spec.depends}
        for target_node, slot_inputs in dependency_map.items():
            if isinstance(slot_inputs, str):
                # handle scalar dependency like
                # depends: node.slot
                source_node, source_slot = slot_inputs.split(".")
                edges.append(
                    Edge(
                        source_node=source_node,
                        source_slot=source_slot,
                        target_node=target_node,
                        target_slot=None,
                    )
                )
            else:
                # handle arrays of dependencies, positional and kwargs
                position_index = 0
                for arrow in slot_inputs:
                    if isinstance(arrow, Mapping):  # keyword argument
                        target_slot, source_signal = next(iter(arrow.items()))

                    elif isinstance(arrow, str):  # positional argument
                        target_slot = position_index
                        source_signal = arrow
                        position_index += 1

                    else:
                        raise NotImplementedError(
                            "Only supporting signal-slot mapping or node pointer."
                        )

                    source_node, source_slot = source_signal.split(".")

                    edges.append(
                        Edge(
                            source_node=source_node,
                            source_slot=source_slot,
                            target_node=target_node,
                            target_slot=target_slot,
                        )
                    )

        return edges


class TubeClassicEdition:
    def __init__(self):
        print(str(self))

    def __str__(self) -> str:
        important = resources.files("noob") / "important.txt"
        important = important.read_text()
        return important
