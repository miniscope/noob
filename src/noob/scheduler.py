from graphlib import TopologicalSorter
from itertools import count

from pydantic import BaseModel, PrivateAttr

from noob.node import Edge, Node


class Scheduler(BaseModel):
    _epoch: count = PrivateAttr(default_factory=count)
    _graphs: dict[int, TopologicalSorter] = PrivateAttr(default_factory=dict)

    @property
    def graphs(self) -> dict[int, TopologicalSorter]:
        return self._graphs

    def graph(self, nodes: dict[str, Node], edges: list[Edge]) -> TopologicalSorter:
        """
        Builds a topological sorter from nodes and edges, saves it
        into :attr:`.Scheduler.graphs` with the epoch, and returns the sorter.
        """
        graph = self._init_graph(nodes, edges)
        self._graphs[next(self._epoch)] = graph
        return graph

    @staticmethod
    def _init_graph(nodes: dict[str, Node], edges: list[Edge]) -> TopologicalSorter:
        """
        Produce a :class:`.TopologicalSorter` based on the graph induced by
        a set of :class:`.Node` and a set of :class:`.Edge` that yields node ids.

        .. note:: Optional params

            Dependency graph only includes edges where `required == True` -
            aka even if we declare some dependency that passes a value to an
            optional (type annotation is `type | None`), default == `None`
            param, we still call that node even if that optional param is absent.

            Additionally, nodes with `enabled == False` are excluded, even when
            other nodes declare dependency to the disabled node. This means the
            signal of the disabled node will not be emitted and thus will not reach
            the dependent nodes. Disable nodes at your own risk.

            This behavior will likely change,
            allowing explicit parameterization of how optional values are handled,
            see: https://github.com/miniscope/noob/issues/26,

        """
        sorter = TopologicalSorter()
        enabled_nodes = [node_id for node_id, node in nodes.items()]
        for node_id in enabled_nodes:
            required_edges = [
                e.source_node
                for e in edges
                if e.target_node == node_id and e.required and e.source_node in enabled_nodes
            ]
            sorter.add(node_id, *required_edges)
        return sorter
