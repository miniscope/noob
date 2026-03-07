import pytest

from noob.node import Edge


@pytest.fixture()
def optional_graph() -> list[Edge]:
    return [
        Edge(
            source_node="a",
            source_signal="a1",
            target_node="only_optional",
            target_slot="value",
            required=False,
        ),
        Edge(
            source_node="a",
            source_signal="a1",
            target_node="mixed",
            target_slot="optional",
            required=False,
        ),
        Edge(
            source_node="a",
            source_signal="a2",
            target_node="mixed",
            target_slot="required",
            required=True,
        ),
        Edge(
            source_node="mixed",
            source_signal="value",
            target_node="two_hop",
            target_slot="optional",
            required=False,
        ),
        Edge(
            source_node="a",
            source_signal="a2",
            target_node="two_hop",
            target_slot="required",
            required=True,
        ),
        # linear chain to test upstream node/long range dependencies
        Edge(
            source_node="a",
            source_signal="a1",
            target_node="b",
            target_slot="value",
            required=False,
        ),
        Edge(
            source_node="b",
            source_signal="b1",
            target_node="c",
            target_slot="value",
            required=True,
        ),
        Edge(
            source_node="c",
            source_signal="c1",
            target_node="d",
            target_slot="value",
            required=False,
        ),
    ]
