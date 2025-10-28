from noob.node.spec import NodeSpecification  # noqa: I001 - needs to be defined before Node is
from noob.node.base import Edge, Node, process_method, Signal, Slot
from noob.node.return_ import Return
from noob.node.gather import Gather
from noob.node.map import Map
from noob.node.tube import TubeNode

SPECIAL_NODES = {"gather": Gather, "map": Map, "return": Return, "tube": TubeNode}


__all__ = [
    "Edge",
    "Gather",
    "Map",
    "Node",
    "NodeSpecification",
    "Return",
    "Signal",
    "Slot",
    "process_method",
]
