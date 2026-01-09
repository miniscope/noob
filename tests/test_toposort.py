from copy import deepcopy

import pytest

from noob.toposort import TopoSorter


@pytest.fixture
def ts() -> TopoSorter:
    ts = TopoSorter(nodes={}, edges=[])
    ts.add("c", "b")
    ts.add("b", "a")
    return ts


def test_mark_out(ts: TopoSorter) -> None:
    """
    manually marking something as out should behave exactly the same
    as if we called `get_ready()`

    seems a little redundant considering the content of `get_ready()`
    is pretty much self.mark_out()...?
    """
    expected = ts
    result = deepcopy(expected)

    exp_ready = expected.get_ready()
    res_ready = result.ready_nodes
    assert set(exp_ready) == res_ready

    result.mark_out(*res_ready)
    assert result == expected


def test_mark_expire(ts: TopoSorter) -> None:
    """
    Marks a node as having been completed without making nodes that depend on it ready.
    Should also cause the graph to be considered inactive/completed if no more nodes remain
    """

    ts.mark_expired("a")
    assert ts.ready_nodes == set()
    assert not ts.is_active()


def test_dynamic_add(ts: TopoSorter) -> None:
    """
    Adding nodes while a sorter is halfway-through correct sorts it:
    if the dependencies are already completed, place in ready_nodes...
    """
    ready_nodes = ts.get_ready()
    ts.done(*ready_nodes)
    ts.add("d", "b")
    assert ts._node2info["d"].nqueue == 1
    assert ts._node2info["b"].successors == ["c", "d"]

    ts.add("e", "a")
    assert ts._node2info["e"].nqueue == 0
    assert ts._node2info["a"].successors == ["b", "e"]
    assert "e" in ts.ready_nodes


def test_adding_merges(ts: TopoSorter) -> None:
    """
    Adding more predecessors (by calling add again) to a node that's already in the graph
    merges the predecessors with the existing ones and updates the predecessor count
    and takes it out of `ready_nodes` if applicable
    """
    ts.add("c", "d")
    assert ts._node2info["c"].nqueue == 2
    assert "c" in ts._node2info["d"].successors

    assert "a" in ts.ready_nodes
    ts.add("a", "aa")
    assert "a" not in ts.ready_nodes
    assert "aa" in ts.ready_nodes


def test_invalid_dynamic_add(ts: TopoSorter) -> None:
    """
    If the node is out or completed, disallow adding additional predecessors to it
    """
    out_nodes = ts.get_ready()
    with pytest.raises(ValueError, match="out"):
        ts.add(out_nodes[0], "d")

    ts.done(*out_nodes)
    with pytest.raises(ValueError, match="done"):
        ts.add(out_nodes[0], "d")
