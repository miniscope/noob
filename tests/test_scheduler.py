from noob import Tube
from noob.scheduler import Scheduler


def test_graph_increment() -> None:
    """
    Scheduler maintains a history of all the graphs that it has generated,
    with the epoch in which the graph was created.
    """
    scheduler = Scheduler()
    noob_ids = ["testing-basic", "testing-basic", "testing-branch", "testing-disable-node"]
    for id_ in noob_ids:
        tube = Tube.from_specification(id_)
        scheduler.graph(nodes=tube.enabled_nodes, edges=tube.edges)

    assert len(scheduler.graphs) == len(noob_ids)
    assert list(scheduler.graphs[0].static_order()) == ["a", "b", "c"]
    assert list(scheduler.graphs[1].static_order()) == ["a", "b", "c"]
    assert list(scheduler.graphs[2].static_order()) == ["a", "b", "c", "d"]
    assert list(scheduler.graphs[3].static_order()) == ["a", "c"]
