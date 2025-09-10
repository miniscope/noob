from noob.tube import Tube, TubeSpecification


def test_tube_does_nothing():
    tube = Tube()
    # hope nothing happened
    assert True


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
