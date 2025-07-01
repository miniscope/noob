from noob.node.spec import NodeSpecification  # noqa: I001 - needs to be defined before Node is
from noob.node.base import Edge, Node, NodeConfig, Sink, Source, Transform
from noob.node.return_ import Return
from noob.node.gather import Gather
from noob.node.map import Map

SPECIAL_NODES = {"gather": Gather, "map": Map, "return": Return}


__all__ = [
    "Edge",
    "Gather",
    "Map",
    "Node",
    "NodeConfig",
    "NodeSpecification",
    "Return",
    "Sink",
    "Source",
    "Transform",
]
