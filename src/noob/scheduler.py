from graphlib import TopologicalSorter
from itertools import count

from pydantic import BaseModel, PrivateAttr

from noob.event import Event
from noob.node import Edge, Node


class NodeCoord(BaseModel):
    """
    Coordinates of the nodes expressed with epoch and node ID.

    graphlib.TopologicalSorter can only "ready" node_ids of the same epoch.
    So, we add a way to track which epoch of the node_id is ready.

    """

    epoch: int
    id: str


class Scheduler(BaseModel):
    """
    Extends the functionality of :class:`.TopologicalSorter`, by wrapping
    a set of epoch-assigned :class:`.TopologicalSorter`, so that each graph
    (and the nodes within) can be linked to an epoch.

    """

    _clock: count = PrivateAttr(default_factory=count)
    _graphs: dict[int, TopologicalSorter] = PrivateAttr(default_factory=dict)

    def update(self, events: list[Event]) -> None:
        """
        When a set of events are received, update the graphs within the scheduler.

        """
        if not events:
            return

        for event in events:
            self.done(epoch=event["epoch"], node_id=event["node_id"])

    def is_active(self) -> bool:
        """
        Scheduler remains active while it holds at least one graph that is active.

        """
        return any(graph.is_active() for graph in self.graphs.values())

    def get_ready(self) -> list[NodeCoord]:
        """
        Output the set of nodes that are ready across different graphs and epochs.

        """

        ready_nodes = []

        # traverse different epochs
        for epoch, graph in self._graphs.items():
            # traverse each graph
            for node_id in graph.get_ready():
                ready_nodes.append(NodeCoord(epoch=epoch, id=node_id))

        return ready_nodes

    def done(self, epoch: int, node_id: str) -> None:
        self.graphs[epoch].done(node_id)

    def evict_cache(self):
        """
        We can evict the cached event from the node once all nodes
        that depend on the given node is marked "done."

        """

    @property
    def graphs(self) -> dict[int, TopologicalSorter]:
        return self._graphs

    def add_graph(self, nodes: dict[str, Node], edges: list[Edge]) -> None:
        """
        Builds a topological sorter from nodes and edges, saves it
        into :attr:`.Scheduler.graphs` with the epoch.

        """
        graph = self._init_graph(nodes, edges)
        graph.prepare()

        self._graphs[next(self._clock)] = graph

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
