from collections import defaultdict
from collections.abc import Sequence
from copy import copy
from operator import attrgetter
from typing import Any, TypeAlias

from noob.exceptions import AlreadyDoneError, NotAddedError
from noob.node import Edge, NodeSpecification
from noob.types import NodeID, NodeSignal

GraphItem: TypeAlias = NodeID | NodeSignal


class _NodeInfo:
    __slots__ = (
        "node",
        "nqueue",
        "successors",
        "predecessors",
        "optional_predecessors",
        "optional_successors",
    )

    def __init__(self, node: GraphItem) -> None:
        # The node this class is augmenting.
        self.node = node

        # Number of predecessors, generally >= 0. When this value falls to 0,
        # and is returned by get_ready(), this is set to _NODE_OUT and when the
        # node is marked done by a call to done(), set to _NODE_DONE.
        self.nqueue = 0

        # Immediate successor nodes that run after this node does
        self.successors: set[GraphItem] = set()

        # Immediate predecessor nodes that we depend on to run
        self.predecessors: set[GraphItem] = set()

        # Optional (immediate) predecessors - we only run once their fate has been decided,
        # but we can run without them having emitted an event
        # Elsewhere the sign is flipped ("required" rather than "optional"),
        # but we use optional here, assuming that optional deps are comparatively rare,
        # and it is less costly to represent the few times we do need to handle it
        # rather than every time we don't.
        self.optional_predecessors: set[GraphItem] = set()

        # Optional **downstream** successors - nodes that have some optional dependency
        # at any graph depth beneath us that we need to decrement the nqueue of when we're expired
        # Note that this should *only* be node ids, not signals, since signals always
        # require the node that emits them to run.
        self.optional_successors: set[NodeID] = set()

    def __eq__(self, other: Any) -> bool:
        """https://stackoverflow.com/a/4522896/14537948"""
        if isinstance(other, self.__class__) and self.__slots__ == other.__slots__:
            attr_getters = [attrgetter(attr) for attr in self.__slots__]
            return all(getter(self) == getter(other) for getter in attr_getters)

        return False

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        val = {
            "nqueue": self.nqueue,
            "successors": self.successors,
            "predecessors": self.predecessors,
        }
        if self.optional_predecessors:
            val["optional_predecessors"] = self.optional_predecessors
        if self.optional_successors:
            val["optional_successors"] = self.optional_successors
        return str(val)


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
            self.add(e.target_node, NodeSignal(e.source_node, e.source_signal), required=e.required)
        # add enabled nodes that have no edges
        for node_id, node in nodes.items():
            if node.enabled and node_id not in self._node2info:
                self.add(node_id)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return all(getattr(self, slot) == getattr(other, slot) for slot in self.__slots__)
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

    def mark_expired(self, *nodes: GraphItem, unlock_optionals: bool = True) -> None:
        """
        Mark node(s) as having been completed without making its dependent nodes ready -
        used when a node emits ``NoEvent``

        Args:
            unlock_optionals (bool): If True, decrement the nqueue for downstream nodes with
                optional dependencies.
                Use False when e.g. forking subepochs where no downstream nodes should run
        """
        self._expire_nodes(*nodes)
        if not unlock_optionals:
            return
        # decrement the nqueue on any downstream nodes with optional dependencies
        for node in nodes:
            info = self._get_nodeinfo(node)
            for successor in info.optional_successors:
                successor_info = self._get_nodeinfo(successor)
                successor_info.nqueue -= 1
                if successor_info.nqueue == 0 and successor not in self.done_nodes | self.out_nodes:
                    if successor in self._disabled_nodes:
                        self.mark_expired(successor, unlock_optionals=False)
                    else:
                        self.mark_ready(successor)

    def add(self, node: GraphItem, *predecessors: GraphItem, required: bool = True) -> None:
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
            required (bool): Whether these predecessors must have emitted an event
                for this node to run.
                Optional predecessors ensure the node runs *after* the predecessors,
                even if they emit nothing.
        """
        # Refuse to add nodes that are out / done
        reject = [(self.out_nodes, "already out"), (self.done_nodes, "already done")]
        reasons = [reason for group, reason in reject if node in group]
        if reasons:
            raise ValueError(f"{node} cannot be added: {', '.join(reasons)}")

        predecessors = tuple(
            [
                (
                    NodeSignal(p[0], p[1])
                    if isinstance(p, tuple) and not isinstance(p, NodeSignal)
                    else p
                )
                for p in predecessors
            ]
        )

        # Create the predecessor -> node edges
        # filter predecessors to only those that are newly being created
        new_predecessors = []
        for pred in predecessors:
            pred_info = self._get_nodeinfo(pred)
            if node in pred_info.successors:
                continue
            new_predecessors.append(pred)
            pred_info.successors.add(node)

            if isinstance(pred, NodeSignal):
                # (node, signal) predecessors must always depend on the node
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
        nodeinfo.predecessors.update(new_predecessors)
        self._update_optionals(node, predecessors, required)
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
            self._expire_nodes(node)

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

    def _expire_nodes(self, *nodes: GraphItem) -> None:
        """
        Mark nodes as having been completed, either via done or marked explicitly expired
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

    def _update_optionals(
        self, node: GraphItem, predecessors: Sequence[GraphItem], required: bool
    ) -> None:
        """
        Update optional links for this node and predecessors:
        - On the node: update the `optional` set - newer declarations override older ones,
          so even if we have added this predecessor previously, the current requiredness overrides
        - Update downstream optionals - find downstream nodes that we need to decrement nqueue
          if we emit noevent
        - Update upstream optionals - If these predecessors aren't required, add ourselves to
          the upstream optional sets.
        """
        if isinstance(node, NodeSignal):
            return

        info = self._get_nodeinfo(node)
        if required:
            info.optional_predecessors.difference_update(predecessors)
        else:
            info.optional_predecessors.update(predecessors)

        to_visit = set(info.successors)
        seen: set[GraphItem] = set()
        while to_visit:
            current = to_visit.pop()
            current_info = self._get_nodeinfo(current)
            for next_successor in current_info.successors:
                next_info = self._get_nodeinfo(next_successor)
                if (
                    not isinstance(next_successor, NodeSignal)
                    and current in next_info.optional_predecessors
                ):
                    # optional edge! this is the one we need to update,
                    # terminate traversal of this branch, since optionalness doesn't propagate
                    info.optional_successors.add(next_successor)
                else:
                    to_visit.update(next_info.successors - seen)
                    seen.update(next_info.successors)

        # update upstream - since optional/non-optional can overlap,
        # and we can add required/optional out of order with overwriting,
        # we do this in two passes - first clearing all optionals and re-adding
        # first pass - remove optionals
        to_visit = set(info.predecessors) - info.optional_predecessors
        seen = set()
        while to_visit:
            current = to_visit.pop()
            current_info = self._get_nodeinfo(current)
            current_info.optional_successors.discard(node)
            to_visit.update(
                current_info.predecessors.difference(current_info.optional_predecessors).difference(
                    seen
                )
            )
            seen.update(current_info.predecessors)

        # second pass - re-add optionals
        to_visit = set(info.optional_predecessors)
        seen = set()
        while to_visit:
            current = to_visit.pop()
            current_info = self._get_nodeinfo(current)
            if isinstance(current, NodeSignal):
                current_info.optional_successors.add(node)
            if not current_info.optional_predecessors:
                to_visit.update(current_info.predecessors - seen)
                seen.update(current_info.predecessors)

    def __deepcopy__(self, memo: dict) -> "TopoSorter":
        sorter = TopoSorter()
        for slot in self.__slots__:
            if slot == "_node2info":
                new_node2info = {}
                for node, info in self._node2info.items():
                    new_info = _NodeInfo(node)
                    new_info.nqueue = info.nqueue
                    new_info.successors = set(info.successors)
                    new_info.predecessors = set(info.predecessors)
                    new_info.optional_predecessors = set(info.optional_predecessors)
                    new_info.optional_successors = set(info.optional_successors)
                    new_node2info[node] = new_info
                sorter._node2info = new_node2info
            elif isinstance((val := getattr(self, slot)), dict | defaultdict | set):
                getattr(sorter, slot).update(val)
            elif isinstance(val, int | float):
                setattr(sorter, slot, getattr(self, slot))
            else:
                setattr(sorter, slot, copy(getattr(self, slot)))
        return sorter
