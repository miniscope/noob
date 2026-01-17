import pytest
from pydantic import ValidationError

from noob.tube import Tube, TubeSpecification


def test_tube_init_edges():
    """
    See if we can handle list / dict / scalar type slot inputs
    """
    spec = TubeSpecification.from_any("testing-mixed-slot")
    tube = Tube.from_specification("testing-mixed-slot")

    for node_id in tube.nodes:
        depends = spec.nodes[node_id].depends
        edges = [e for e in tube.edges if e.target_node == node_id]
        argslot_idx = 0

        if isinstance(depends, list):
            for dep in depends:
                positional = isinstance(dep, str)
                if positional:
                    source = dep
                    source_node, source_signal = source.split(".")
                    target_slot = [
                        e
                        for e in edges
                        if e.source_node == source_node and e.source_signal == source_signal
                    ][0].target_slot
                    assert target_slot == argslot_idx
                    argslot_idx += 1
                else:
                    source = str(dep[list(dep.keys())[0]])
                    source_node, source_signal = source.split(".")
                    edge = [
                        e
                        for e in edges
                        if e.source_node == source_node
                        and e.source_signal == source_signal
                        and e.target_slot == list(dep.keys())[0]
                    ]
                    assert len(edge) == 1
        elif isinstance(depends, str):
            assert len(edges) == 1

            edge = edges[0]
            assert edge.source_node == depends.split(".")[0]
            assert edge.source_signal == depends.split(".")[1]
            assert edge.target_slot is None


def test_enable_nodes():
    tube = Tube.from_specification("testing-basic")

    assert list(tube.enabled_nodes.keys()) == ["a", "b", "c"]

    tube.disable_node("c")
    assert list(tube.enabled_nodes.keys()) == ["a", "b"]

    tube.enable_node("c")
    assert list(tube.enabled_nodes.keys()) == ["a", "b", "c"]


def test_cycle_check():
    """Tubes with dependency cycles must raise ValidationError at initialization"""

    with pytest.raises(ValidationError):
        Tube.from_specification("testing-cycle")


@pytest.mark.assets
@pytest.mark.parametrize("tube", ["testing-invalid-post-depends", "testing-invalid-equal-depends"])
def test_assets_exhausted_after_storage(tube: str):
    """
    Nodes cannot depend on runner-scoped assets directly in topological generations
    equal or after to the generation that stores the value of the asset
    (via asset.depends) to protect against unstructured mutation
    """
    with pytest.raises(ValidationError):
        Tube.from_specification(tube)
