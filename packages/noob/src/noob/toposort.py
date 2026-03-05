from collections import defaultdict
from copy import copy
from operator import attrgetter
from typing import Any, TypeAlias

from noob.exceptions import AlreadyDoneError, NotAddedError
from noob.node import Edge, NodeSpecification
from noob.types import NodeID, NodeSignal

GraphItem: TypeAlias = NodeID | NodeSignal


class _NodeInfo:
    __slots__ = "node", "nqueue", "successors"

    def __init__(self, node: GraphItem) -> None:
        # The node this class is augmenting.
        self.node = node

        # Number of predecessors, generally >= 0. When this value falls to 0,
        # and is returned by get_ready(), this is set to _NODE_OUT and when the
        # node is marked done by a call to done(), set to _NODE_DONE.
        self.nqueue = 0

        # List of successor nodes. The list can contain duplicated elements as
        # long as they're all reflected in the successor's npredecessors attribute.
        self.successors: set[GraphItem] = set()

    def __eq__(self, other: Any) -> bool:
        """https://stackoverflow.com/a/4522896/14537948"""
        if isinstance(other, self.__class__) and self.__slots__ == other.__slots__:
            attr_getters = [attrgetter(attr) for attr in self.__slots__]
            return all(getter(self) == getter(other) for getter in attr_getters)

        return False

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return str((self.node, self.nqueue, self.successors))


class TopoSorter:
    """
    Provides functionality to topologically sort a graph of `class: .node.base.Node`.

    Based on graphlib.TopologicalSorter, with some minor changes
    to allow querying nodes at different stages,
    and modifying graph mid-iteration.
    """

    __slots__ = (
        "signals",
        "_node2info",
        "_ready_nodes",
        "_out_nodes",
        "_done_nodes",
        "_disabled_nodes",
        "_ran_nodes",
        "_npassedout",
        "_nfinished",
    )

    def __init__(
        self, nodes: dict[str, NodeSpecification] | None = None, edges: list[Edge] | None = None
    ) -> None:
        if nodes is None:
            nodes = {}
        if edges is None:
            edges = []

        self.signals: dict[NodeID, set[NodeSignal]] = defaultdict(set)
        self._node2info: dict[GraphItem, _NodeInfo] = dict()
        self._ready_nodes: set[GraphItem] = set()
        self._out_nodes: set[GraphItem] = set()
        self._done_nodes: set[GraphItem] = set()
        self._disabled_nodes: set[GraphItem] = set()
        self._ran_nodes: set[GraphItem] = set()
        self._npassedout = 0
        self._nfinished = 0

        # Since we can be passed edges without node specifications,
        # filter on disabled nodes rather than enabled nodes -
        # i.e., we filter edges to any node that are explicitly disabled, but pass others.
        self._disabled_nodes = set(node_id for node_id, node in nodes.items() if not node.enabled)
        for e in edges:
            if e.target_node in self._disabled_nodes:
                continue
            self.add(e.target_node, NodeSignal(e.source_node, e.source_signal))
        # add enabled nodes that have no edges
        for node_id, node in nodes.items():
            if node.enabled and node_id not in self._node2info:
                self.add(node_id)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    @property
    def node_info(self) -> dict[GraphItem, _NodeInfo]:
        return self._node2info

    @property
    def ready_nodes(self) -> set[GraphItem]:
        return self._ready_nodes

    @property
    def out_nodes(self) -> set[GraphItem]:
        return self._out_nodes

    @property
    def done_nodes(self) -> set[GraphItem]:
        return self._done_nodes

    @property
    def ran_nodes(self) -> set[GraphItem]:
        """Nodes that were actually run, marked `done`, rather than expired"""
        return self._ran_nodes

    def mark_ready(self, *nodes: GraphItem) -> None:
        """
        Manually mark a node as ready.

        Normally this is done automatically when marking predecessor nodes as :meth:`.done`
        or when adding nodes with no predecessors.
        """
        self._ready_nodes.update(nodes)

    def mark_out(self, *nodes: GraphItem) -> None:
        """
        Mark a node as being out for processing
        """
        self._ready_nodes -= set(nodes)
        self._out_nodes.update(nodes)
        self._npassedout += len(nodes)

    def mark_expired(self, *nodes: GraphItem) -> None:
        """
        Mark node(s) as having been completed without making its dependent nodes ready -
        used when a node emits ``NoEvent``
        """
        expired = set(nodes) - self._done_nodes
        for node in expired:
            self._ready_nodes.discard(node)
            if node in self._out_nodes:
                self._out_nodes.discard(node)
            else:
                self._npassedout += 1
        self._done_nodes.update(expired)
        self._nfinished += len(expired)

    def add(
        self,
        node: GraphItem,
        *predecessors: GraphItem,
    ) -> None:
        """
        Add a new node and its predecessors to the graph.

        Both the *node* and all elements in *predecessors* must be hashable.

        If called multiple times with the same node argument, the set of dependencies
        will be the union of all dependencies passed in.

        It is possible to add a node with no dependencies (*predecessors* is not provided)
        as well as provide a dependency twice. If a node that has not been provided before
        is included among *predecessors* it will be automatically added to the graph with
        no predecessors of its own.

        Generally, the structure of the topo graph that should be constructed is
        ``source_node <-- (source_node, signal) <- target_node`` ,
        where a given target node depends on a specific signal emitted by the

        Args:
            node (NodeID): The ID of the depending/downstream node
            *predecessors (NodeID | tuple[NodeID, SignalName]): If a string,
                another node ID that the node depends on (any event).
                If a tuple of two strings, the (node, signal) that the node depends on.
        """
        # Refuse to add nodes that are out / done
        reject = [(self.out_nodes, "already out"), (self.done_nodes, "already done")]
        reasons = [reason for group, reason in reject if node in group]
        if reasons:
            raise ValueError(f"{node} cannot be added: {', '.join(reasons)}")

        # Create the predecessor -> node edges
        # filter predecessors to only those that are newly being created
        new_predecessors = []
        for pred in predecessors:
            pred_info = self._get_nodeinfo(pred)
            if node in pred_info.successors:
                continue
            new_predecessors.append(pred)
            pred_info.successors.add(node)
            if isinstance(pred, tuple):
                assert len(pred) == 2, "Only NodeSignal (node_id, signal) tuples allowed"
                # (node, signal) predecessors must always depend on the node
                if not isinstance(pred, NodeSignal):
                    pred = NodeSignal(*pred)
                self.signals[pred[0]].add(pred)
                self.add(pred, pred[0])

            if (
                pred_info.nqueue == 0
                and pred not in self.out_nodes
                and pred not in self.done_nodes
                and pred not in self._disabled_nodes
            ):
                self.mark_ready(pred)

        # Create the node -> predecessor edges
        nodeinfo = self._get_nodeinfo(node)
        ndone_predeccesors = len(self.done_nodes.intersection(new_predecessors))
        nodeinfo.nqueue += len(new_predecessors) - ndone_predeccesors
        if nodeinfo.nqueue == 0:
            self.mark_ready(node)
        else:
            # in case node is called multiple times
            self._ready_nodes.discard(node)

    def get_ready(self, node_id: NodeID | None = None) -> tuple[GraphItem, ...]:
        """
        Return a tuple of all the nodes that are ready.

        Initially it returns all nodes with no predecessors; once those are marked
        as processed by calling "done", further calls will return all new nodes that
        have all their predecessors already processed. Once no more progress can be made,
        empty tuples are returned.

        Args:
            node_id (str | None): If present, only return if the given node is ready
        """
        # Get the nodes that are ready and mark them
        if node_id is None:
            result = tuple(self.ready_nodes)
        else:
            result = tuple(node for node in self.ready_nodes if node == node_id)
        self.mark_out(*result)

        return result

    def is_active(self) -> bool:
        """Return ``True`` if more progress can be made and ``False`` otherwise.

        Progress can be made if cycles do not block the resolution and either there
        are still nodes ready that haven't yet been returned by "get_ready" or the
        number of nodes marked "done" is less than the number that have been returned
        by "get_ready".
        """
        return self._nfinished < self._npassedout or bool(self.ready_nodes)

    def done(self, *nodes: GraphItem) -> None:
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
                raise NotAddedError(f"node {node!r} was not added using add()")

            # If the node has not been marked as "out" previously, inform the user.
            if node not in self.out_nodes:
                if node in self.done_nodes:
                    raise AlreadyDoneError(f"node {node!r} was already marked done")
                else:
                    # we do lots of forward-looking cancellation - if we say it's done, it's done.
                    self.mark_out(node)

            # Mark the node as processed
            self.mark_expired(node)

            # Go to all the successors and reduce the number of predecessors,
            # collecting all the ones that are ready to be returned in the next get_ready() call.
            for successor in nodeinfo.successors:
                if successor in self.done_nodes or successor in self.out_nodes:
                    continue
                successor_info = n2i[successor]
                successor_info.nqueue -= 1
                if successor_info.nqueue == 0:
                    if successor in self._disabled_nodes:
                        self.mark_expired(successor)
                    else:
                        self.mark_ready(successor)

        self._ran_nodes.update(nodes)

    def resurrect(self, *nodes: GraphItem) -> None:
        """
        If a node was marked as expired (but not run),
        returns it to the processing graph -
        the inverse of :meth:`.mark_expired`
        """
        for node in nodes:
            if node in self._ran_nodes:
                raise AlreadyDoneError(
                    f"node {node!r} was marked done, not expired! can only resurrect expired nodes."
                )
            if node not in self._done_nodes or node in self._disabled_nodes:
                continue
            self._done_nodes.remove(node)
            self._nfinished -= 1
            self._npassedout -= 1
            if self._node2info[node].nqueue == 0:
                self.mark_ready(node)

    def find_cycle(self) -> list[GraphItem] | None:
        n2i = self._node2info
        stack: list[GraphItem] = []
        itstack = []
        seen = set()
        node2stacki: dict[GraphItem, int] = {}

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

    def _get_nodeinfo(self, node: GraphItem) -> _NodeInfo:
        if (result := self._node2info.get(node)) is None:
            self._node2info[node] = result = _NodeInfo(node)
        return result

    def __deepcopy__(self, memo: dict) -> "TopoSorter":
        sorter = TopoSorter()
        for slot in self.__slots__:
            if slot == "_node2info":
                new_node2info = {}
                for node, info in self._node2info.items():
                    new_info = _NodeInfo(node)
                    for nodeslot in _NodeInfo.__slots__:
                        setattr(new_info, nodeslot, copy(getattr(info, nodeslot)))
                    new_node2info[node] = new_info
                sorter._node2info = new_node2info
            else:
                setattr(sorter, slot, copy(getattr(self, slot)))
        return sorter
