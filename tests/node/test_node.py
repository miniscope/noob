import pytest

from noob.node import NodeSpecification
from noob.node.base import Node, Signal


@pytest.mark.parametrize(
    "type, params, expected",
    [
        (
            "noob.testing.CountSource",
            {"limit": 10, "start": 5},
            [Signal(name="index", type_=int)],
        ),
        ("noob.testing.Multiply", {}, [Signal(name="product", type_=int)]),
    ],
)
def test_node_subclass_signal(type, params, expected):
    node = Node.from_specification(
        spec=NodeSpecification(
            id="test_node_subclass_signal",
            type=type,
            params=params,
            depends=None,
        )
    )

    assert node.signals == expected


def test_node_internal():
    node = Node.from_specification(
        spec=NodeSpecification(
            id="test_node_process",
            type="noob.testing.BasicInternal",
            params={"param1": 1, "param2": 2.5},
            depends=None,
        )
    )
    node.init()
    assert node.process(left=2, right=3) == 1 * 2.5 * 2 * 3
    assert set(node.slots) == {"left", "right"}
    assert node.signals == [Signal(name="product", type_=float)]
