from typing import Any

from noob.node import NodeSpecification, Signal, Slot


def test_nodeinfo():
    """NodeInfo can extract slot/signal information from a node class/function"""
    spec = NodeSpecification(id="test", type="noob.testing.concat")
    info = spec.nodeinfo
    assert info["signals"] == {"value": Signal(name="value", annotation=str)}
    assert info["slots"] == {"strings": Slot(name="strings", annotation=list[str])}


def test_dynamic_nodeinfo():
    """Nodeinfo from a node that computes its nodeinfo from the spec"""
    spec = NodeSpecification(
        id="test",
        type="noob.testing.DynamicSignals",
        params={"signals_": ["apple", "banana"], "slots_": ["cherry", "diaper"]},
    )
    info = spec.nodeinfo
    assert info["signals"] == {
        "apple": Signal(name="apple", annotation=Any),
        "banana": Signal(name="banana", annotation=Any),
    }
    assert info["slots"] == {
        "cherry": Slot(name="cherry", annotation=Any),
        "diaper": Slot(name="diaper", annotation=Any),
    }
