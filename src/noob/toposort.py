from enum import Enum
from graphlib import TopologicalSorter


class _SchedulerNodeStatus(Enum):
    READY = 0
    OUT = -1
    DONE = -2


class _NodeInfo:
    __slots__ = "node", "npredecessors", "successors"

    def __init__(self, node: str) -> None:
        # The node this class is augmenting.
        self.node = node

        # Number of predecessors, generally >= 0. When this value falls to 0,
        # and is returned by get_ready(), this is set to _NODE_OUT and when the
        # node is marked done by a call to done(), set to _NODE_DONE.
        self.npredecessors = 0

        # List of successor nodes. The list can contain duplicated elements as
        # long as they're all reflected in the successor's npredecessors attribute.
        self.successors: list[str] = []


class TopoSorter(TopologicalSorter):
    """
    Provides functionality to topologically sort a graph of hashable nodes.
    Based on graphlib.TopologicalSorter.
    """

    def __init__(self, graph: dict | None = None):
        self._node2info: dict[str, _NodeInfo] = {}
        self._prepared = False  # mypy complains about _ready_nodes otherwise.
        self._ready_nodes: list[str] = []
        self._done_nodes: list[str] = []
        self._npassedout = 0
        self._nfinished = 0

        if graph is not None:
            for node, predecessors in graph.items():
                self.add(node, *predecessors)

    @property
    def ready_nodes(self) -> list[str]:
        return self._ready_nodes

    @property
    def done_nodes(self) -> list[str]:
        return self._done_nodes

    def add(self, node: str, *predecessors: str) -> None:
        """Add a new node and its predecessors to the graph.

        Both the *node* and all elements in *predecessors* must be hashable.

        If called multiple times with the same node argument, the set of dependencies
        will be the union of all dependencies passed in.

        It is possible to add a node with no dependencies (*predecessors* is not provided)
        as well as provide a dependency twice. If a node that has not been provided before
        is included among *predecessors* it will be automatically added to the graph with
        no predecessors of its own.

        Raises ValueError if called after "prepare".
        """
        if self._prepared:
            raise ValueError("Nodes cannot be added after a call to prepare()")

        # Create the node -> predecessor edges
        nodeinfo = self._get_nodeinfo(node)
        nodeinfo.npredecessors += len(predecessors)

        # Create the predecessor -> node edges
        for pred in predecessors:
            pred_info = self._get_nodeinfo(pred)
            pred_info.successors.append(node)

    def prepare(self) -> None:
        """Mark the graph as finished and check for cycles in the graph.

        If any cycle is detected, "CycleError" will be raised, but "get_ready" can
        still be used to obtain as many nodes as possible until cycles block more
        progress. After a call to this function, the graph cannot be modified and
        therefore no more nodes can be added using "add".
        """
        if self._prepared:
            raise ValueError("cannot prepare() more than once")

        self._ready_nodes = [i.node for i in self._node2info.values() if i.npredecessors == 0]
        self._prepared = True

    def get_ready(self) -> tuple[str, ...]:
        """Return a tuple of all the nodes that are ready.

        Initially it returns all nodes with no predecessors; once those are marked
        as processed by calling "done", further calls will return all new nodes that
        have all their predecessors already processed. Once no more progress can be made,
        empty tuples are returned.

        Raises ValueError if called without calling "prepare" previously.
        """
        if not self._prepared:
            raise ValueError("prepare() must be called first")

        # Get the nodes that are ready and mark them
        result = tuple(self._ready_nodes)
        n2i = self._node2info
        for node in result:
            n2i[node].npredecessors = _SchedulerNodeStatus.OUT.value

        # Clean the list of nodes that are ready and update
        # the counter of nodes that we have returned.
        self._ready_nodes.clear()
        self._npassedout += len(result)

        return result

    def is_active(self) -> bool:
        """Return ``True`` if more progress can be made and ``False`` otherwise.

        Progress can be made if cycles do not block the resolution and either there
        are still nodes ready that haven't yet been returned by "get_ready" or the
        number of nodes marked "done" is less than the number that have been returned
        by "get_ready".

        Raises ValueError if called without calling "prepare" previously.
        """
        if not self._prepared:
            raise ValueError("prepare() must be called first")
        return self._nfinished < self._npassedout or bool(self._ready_nodes)

    def mark_out(self, *nodes: str) -> None:
        """
        Manually mark a node as "out"
        """
        for node in nodes:
            self._node2info[node].npredecessors = _SchedulerNodeStatus.OUT.value
            if node in self.ready_nodes:
                self._ready_nodes.remove(node)

    def done(self, *nodes: str) -> None:
        """Marks a set of nodes returned by "get_ready" as processed.

        This method unblocks any successor of each node in *nodes* for being returned
        in the future by a call to "get_ready".

        Raises ValueError if any node in *nodes* has already been marked as
        processed by a previous call to this method, if a node was not added to the
        graph by using "add" or if called without calling "prepare" previously or if
        node has not yet been returned by "get_ready".
        """

        if not self._prepared:
            raise ValueError("prepare() must be called first")

        n2i = self._node2info

        for node in nodes:

            # Check if we know about this node (it was added previously using add()
            if (nodeinfo := n2i.get(node)) is None:
                raise ValueError(f"node {node!r} was not added using add()")

            # If the node has not being returned (marked as ready) previously, inform the user.
            stat = nodeinfo.npredecessors
            if stat != _SchedulerNodeStatus.OUT.value:
                if stat >= 0:
                    raise ValueError(f"node {node!r} was not passed out (still not ready)")
                elif stat == _SchedulerNodeStatus.DONE.value:
                    raise ValueError(f"node {node!r} was already marked done")
                else:
                    raise AssertionError(f"node {node!r}: unknown status {stat}")

            # Mark the node as processed
            nodeinfo.npredecessors = _SchedulerNodeStatus.DONE.value
            self._done_nodes.append(node)

            # Go to all the successors and reduce the number of predecessors,
            # collecting all the ones that are ready to be returned in the next get_ready() call.
            for successor in nodeinfo.successors:
                successor_info = n2i[successor]
                successor_info.npredecessors -= 1
                if successor_info.npredecessors == _SchedulerNodeStatus.READY.value:
                    self._ready_nodes.append(successor)
            self._nfinished += 1

    def mark_done(self, *nodes: str) -> None:
        """Mark nodes "done" without making their dependent nodes ready."""
        for node in nodes:
            self._node2info[node].npredecessors = _SchedulerNodeStatus.DONE.value
            self._done_nodes.append(node)
            self._nfinished += 1
            if node in self.ready_nodes:
                self._ready_nodes.remove(node)

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

    def _get_nodeinfo(self, node: str) -> dict[str, _NodeInfo]:
        if (result := self._node2info.get(node)) is None:
            self._node2info[node] = result = _NodeInfo(node)
        return result
