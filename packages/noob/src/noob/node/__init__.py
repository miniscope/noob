from noob.node.spec import (  # noqa: I001 - needs to be defined before Node is
    NodeInfo,
    NodeSpecification,
)
from noob.node.base import Edge, Node, Signal, Slot, process_method
from noob.node.gather import Gather
from noob.node.map import Map
from noob.node.return_ import Return
from noob.node.tube import TubeNode

SPECIAL_NODES = {"gather": Gather, "map": Map, "return": Return, "tube": TubeNode}
"""
Map from short names used in node ``type`` values to special node classes 
"""


__all__ = [
    "Edge",
    "Gather",
    "Map",
    "Node",
    "NodeInfo",
    "NodeSpecification",
    "Return",
    "Signal",
    "Slot",
    "process_method",
]
