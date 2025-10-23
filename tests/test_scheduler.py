from noob import Tube
from noob.scheduler import NodeCoord, Scheduler


def test_append_graphs() -> None:
    """
    Scheduler must be able to hold multiple graphs of different statuses

    """
    scheduler = Scheduler()
    noob_ids = ["testing-basic", "testing-basic", "testing-branch", "testing-disable-node"]
    for id_ in noob_ids:
        tube = Tube.from_specification(id_)
        scheduler.add_graph(nodes=tube.enabled_nodes, edges=tube.edges)

    assert len(scheduler.graphs) == len(noob_ids)


def test_get_ready() -> None:
    """
    Scheduler must be able to identify ready nodes across different epochs

    """
    scheduler = Scheduler()
    noob_ids = ["testing-merge", "testing-merge"]
    for id_ in noob_ids:
        tube = Tube.from_specification(id_)
        scheduler.add_graph(nodes=tube.enabled_nodes, edges=tube.edges)

    ready = scheduler.get_ready()
    # should return all independent nodes (merge has 2)
    assert len(ready) == 4

    scheduler.done(0, "a")
    # should only return the newly ready node
    assert scheduler.get_ready() == [NodeCoord(epoch=0, id="c")]
