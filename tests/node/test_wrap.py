import datetime

import pytest

from noob.node import NodeSpecification
from noob.node.base import Node, Signal


@pytest.mark.parametrize(
    "type_, params, expected",
    [
        (
            "noob.testing.CountSource",
            {"limit": 10, "start": 5},
            [Signal(name="index", type_=int)],
        ),
        ("noob.testing.Multiply", {}, [Signal(name="product", type_=int)]),
    ],
)
def test_subclass(type_, params, expected):
    node = Node.from_specification(
        spec=NodeSpecification(
            id="test_node_subclass_signal", type=type_, params=params, depends=None
        )
    )
    assert node.signals == expected


def test_class_with_process():
    node = Node.from_specification(
        spec=NodeSpecification(
            id="test_node_process",
            type="noob.testing.VolumeProcess",
            params={"height": 5},
            depends=None,
        )
    )
    node.init()
    assert node.process(width=2, depth=3) == 5 * 2 * 3
    assert set(node.slots) == {"width", "depth"}
    assert node.signals == [Signal(name="volume", type_=int)]


def test_class_without_process():
    node = Node.from_specification(
        spec=NodeSpecification(id="test_volume", type="noob.testing.Volume", params={"height": 5})
    )
    node.init()
    assert node.process(width=2, depth=3) == 5 * 2 * 3


def test_class_without_init_params():
    node = Node.from_specification(spec=NodeSpecification(id="test_now", type="noob.testing.Now"))
    node.init()
    prefix = "What time is it?: "
    assert node.process(prefix=prefix) == f"{prefix}{datetime.datetime.now().isoformat()}"
