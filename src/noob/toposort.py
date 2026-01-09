from graphlib import CycleError
from operator import attrgetter
from typing import Any

from noob.node import Edge, NodeSpecification
from noob.types import NodeID


class _NodeInfo:
    __slots__ = "node", "nqueue", "successors"

    def __init__(self, node: str) -> None:
        # The node this class is augmenting.
        self.node = node

        # Number of predecessors, generally >= 0. When this value falls to 0,
        # and is returned by get_ready(), this is set to _NODE_OUT and when the
        # node is marked done by a call to done(), set to _NODE_DONE.
        self.nqueue = 0

        # List of successor nodes. The list can contain duplicated elements as
        # long as they're all reflected in the successor's npredecessors attribute.
        self.successors: list[NodeID] = []

    def __eq__(self, other: Any) -> bool:
        """https://stackoverflow.com/a/4522896/14537948"""
        if isinstance(other, self.__class__) and self.__slots__ == other.__slots__:
            attr_getters = [attrgetter(attr) for attr in self.__slots__]
            return all(getter(self) == getter(other) for getter in attr_getters)

        return False

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)


class TopoSorter:
    """
    Provides functionality to topologically sort a graph of `class: .node.base.Node`.

    Based on graphlib.TopologicalSorter, with some minor changes
    to allow querying nodes at different stages,
    and modifying graph mid-iteration.
    """

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __init__(self, nodes: dict[str, NodeSpecification], edges: list[Edge]) -> None:
        self._node2info: dict[str, _NodeInfo] = dict()
        self._ready_nodes: set[NodeID] = set()
        self._out_nodes: set[NodeID] = set()
        self._done_nodes: set[NodeID] = set()
        self._npassedout = 0
        self._nfinished = 0

        enabled_nodes = [node_id for node_id, node in nodes.items() if node.enabled]
        for node_id in enabled_nodes:
            required_edges = [
                e.source_node
                for e in edges
                if e.target_node == node_id and e.target_node in enabled_nodes
            ]
            self.add(node_id, *required_edges)

    @property
    def ready_nodes(self) -> set[NodeID]:
        return self._ready_nodes

    @property
    def out_nodes(self) -> set[NodeID]:
        return self._out_nodes

    @property
    def done_nodes(self) -> set[NodeID]:
        return self._done_nodes

    def mark_ready(self, *nodes: NodeID) -> None:
        self._ready_nodes.update(nodes)

    def mark_out(self, *nodes: NodeID) -> None:
        for node in nodes:
            # there isn't a good way to discard multiple elements in bulk
            self._ready_nodes.discard(node)  # missing ok
        self._out_nodes.update(nodes)
        self._npassedout += len(nodes)

    def mark_expired(self, *nodes: NodeID) -> None:
        for node in nodes:
            self._ready_nodes.discard(node)
            self._out_nodes.discard(node)
        self._done_nodes.update(nodes)
        self._nfinished += len(nodes)

    def add(self, node: NodeID, *predecessors: NodeID) -> None:
        """
        Add a new node and its predecessors to the graph.

        Both the *node* and all elements in *predecessors* must be hashable.

        If called multiple times with the same node argument, the set of dependencies
        will be the union of all dependencies passed in.

        It is possible to add a node with no dependencies (*predecessors* is not provided)
        as well as provide a dependency twice. If a node that has not been provided before
        is included among *predecessors* it will be automatically added to the graph with
        no predecessors of its own.
        """
        # Refuse to add nodes that are out / done
        reject = [(self.out_nodes, "already out"), (self.done_nodes, "already done")]
        reasons = [reason for group, reason in reject if node in group]
        if reasons:
            raise ValueError(f"{node} cannot be added: {', '.join(reasons)}")

        # Create the node -> predecessor edges
        nodeinfo = self._get_nodeinfo(node)
        ndone_predeccesors = len(self.done_nodes.intersection(predecessors))
        nodeinfo.nqueue += len(predecessors) - ndone_predeccesors
        if nodeinfo.nqueue == 0:
            self.mark_ready(node)
        else:
            # in case node is called multiple times
            self._ready_nodes.discard(node)

        # Create the predecessor -> node edges
        for pred in predecessors:
            pred_info = self._get_nodeinfo(pred)
            pred_info.successors.append(node)
            if pred_info.nqueue == 0 and pred not in self.out_nodes and pred not in self.done_nodes:
                self.mark_ready(pred)

    def get_ready(self) -> tuple[str, ...]:
        """
        Return a tuple of all the nodes that are ready.

        Initially it returns all nodes with no predecessors; once those are marked
        as processed by calling "done", further calls will return all new nodes that
        have all their predecessors already processed. Once no more progress can be made,
        empty tuples are returned.
        """
        # Get the nodes that are ready and mark them
        result = tuple(self.ready_nodes)
        self.mark_out(*result)

        return result

    def is_active(self) -> bool:
        """Return ``True`` if more progress can be made and ``False`` otherwise.

        Progress can be made if cycles do not block the resolution and either there
        are still nodes ready that haven't yet been returned by "get_ready" or the
        number of nodes marked "done" is less than the number that have been returned
        by "get_ready".

        Raises ValueError if called without calling "prepare" previously.
        """
        return self._nfinished < self._npassedout or bool(self.ready_nodes)

    def done(self, *nodes: str) -> None:
        """Marks a set of nodes returned by "get_ready" as processed.

        This method unblocks any successor of each node in *nodes* for being returned
        in the future by a call to "get_ready".

        Raises ValueError if any node in *nodes* has already been marked as
        processed by a previous call to this method, if a node was not added to the
        graph by using "add" or if called without calling "prepare" previously or if
        node has not yet been returned by "get_ready".
        """
        n2i = self._node2info

        for node in nodes:

            # Check if we know about this node (it was added previously using add()
            if (nodeinfo := n2i.get(node)) is None:
                raise ValueError(f"node {node!r} was not added using add()")

            # If the node has not been marked as "out" previously, inform the user.
            if node not in self.out_nodes:
                if node in self.done_nodes:
                    raise ValueError(f"node {node!r} was already marked done")
                else:
                    raise ValueError(f"node {node!r} was not passed out")

            # Mark the node as processed
            self.mark_expired(node)

            # Go to all the successors and reduce the number of predecessors,
            # collecting all the ones that are ready to be returned in the next get_ready() call.
            for successor in nodeinfo.successors:
                successor_info = n2i[successor]
                successor_info.nqueue -= 1
                if successor_info.nqueue == 0:
                    self.mark_ready(successor)

    def find_cycle(self) -> list[str] | None:
        n2i = self._node2info
        stack: list[str] = []
        itstack = []
        seen = set()
        node2stacki: dict[str, int] = {}

        for node in n2i:
            if node in seen:
                continue

            while True:
                if node in seen:
                    # If we have seen already the node and is in the
                    # current stack we have found a cycle.
                    if node in node2stacki:
                        return stack[node2stacki[node] :] + [node]
                    # else go on to get next successor
                else:
                    seen.add(node)
                    itstack.append(iter(n2i[node].successors).__next__)
                    node2stacki[node] = len(stack)
                    stack.append(node)

                # Backtrack to the topmost stack entry with
                # at least another successor.
                while stack:
                    try:
                        node = itstack[-1]()
                        break
                    except StopIteration:
                        del node2stacki[stack.pop()]
                        itstack.pop()
                else:
                    break
        return None

    def _get_nodeinfo(self, node: str) -> _NodeInfo:
        if (result := self._node2info.get(node)) is None:
            self._node2info[node] = result = _NodeInfo(node)
        return result
