from collections.abc import Collection, Generator
from copy import deepcopy
from typing import Any

import pytest

from noob.node import Edge, NodeSpecification
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


def test_add_deduplicates():
    """
    Adding a dependency multiple times is idempotent
    """
    ts = TopoSorter()
    ts.add("b", "a")
    assert ts._node2info["b"].nqueue == 1
    ts.add("b", "a")
    assert ts._node2info["b"].nqueue == 1
    ts.mark_out("a")
    ts.done("a")
    assert "b" in ts.ready_nodes
    ts.add("b", "a")
    assert "b" in ts.ready_nodes
    assert ts._node2info["b"].nqueue == 0


# --------------------------------------------------
# Tests from original graphlib implementation adapted for pytest
# https://github.com/python/cpython/blob/main/Lib/test/test_graphlib.py
# --------------------------------------------------


def _graphlib_init_to_noob(graph: dict[Any, Collection]) -> list[Edge]:
    """
    Convert graphlib-style init to noob style edges
    """
    edges = []
    for node_id, dependents in graph.items():
        for dep in dependents:
            edges.append(
                Edge(
                    source_node=str(dep),
                    source_signal="value",
                    target_node=str(node_id),
                    target_slot="value",
                )
            )
    return edges


def _static_order_with_groups(ts: TopoSorter) -> Generator[tuple[Any], None, None]:
    while ts.is_active():
        nodes = ts.get_ready()
        for node in nodes:
            ts.done(node)
        yield tuple(sorted(nodes))


def _test_graph(graph: dict[Any, Collection], expected: list[tuple]) -> None:
    # edges = _graphlib_init_to_noob(graph)
    ts = TopoSorter()
    for node_id, dependents in graph.items():
        ts.add(node_id, *dependents)
    # ts = TopoSorter(edges=edges)
    actual = list(_static_order_with_groups(ts))
    sorted_expected = [tuple(sorted(group)) for group in expected]
    assert actual == sorted_expected


def _assert_cycle(graph: dict[Any, set], cycle: list) -> None:
    ts = TopoSorter()
    for node, dependson in graph.items():
        ts.add(node, *dependson)

    found_cycle = ts.find_cycle()
    assert found_cycle == cycle


def test_simple_cases():
    _test_graph(
        {"2": {"11"}, "9": {"11", "8"}, "10": {"11", "3"}, "11": {"7", "5"}, "8": {"7", "3"}},
        [("3", "5", "7"), ("8", "11"), ("2", "9", "10")],
    )

    _test_graph({"1": {}}, [("1",)])

    _test_graph({str(x): {str(x + 1)} for x in range(10)}, [(str(x),) for x in range(10, -1, -1)])

    _test_graph(
        {
            "2": {"3"},
            "3": {"4"},
            "4": {"5"},
            "5": {"1"},
            "11": {"12"},
            "12": {"13"},
            "13": {"14"},
            "14": {"15"},
        },
        [("1", "15"), ("5", "14"), ("4", "13"), ("3", "12"), ("2", "11")],
    )

    _test_graph(
        {
            "0": ["1", "2"],
            "1": ["3"],
            "2": ["5", "6"],
            "3": ["4"],
            "4": ["9"],
            "5": ["3"],
            "6": ["7"],
            "7": ["8"],
            "8": ["4"],
            "9": [],
        },
        [("9",), ("4",), ("3", "8"), ("1", "5", "7"), ("6",), ("2",), ("0",)],
    )

    _test_graph({"0": ["1", "2"], "1": [], "2": ["3"], "3": []}, [("1", "3"), ("2",), ("0",)])

    _test_graph(
        {"0": ["1", "2"], "1": [], "2": ["3"], "3": [], "4": ["5"], "5": ["6"], "6": []},
        [("1", "3", "6"), ("2", "5"), ("0", "4")],
    )


def test_no_dependencies():
    _test_graph({"1": {"2"}, "3": {"4"}, "5": {"6"}}, [("2", "4", "6"), ("1", "3", "5")])

    _test_graph({"1": set(), "3": set(), "5": set()}, [("1", "3", "5")])


def test_the_node_multiple_times():
    # Test same node multiple times in dependencies
    _test_graph(
        {"1": {"2"}, "3": {"4"}, "0": ["2", "4", "4", "4", "4", "4"]}, [("2", "4"), ("0", "1", "3")]
    )

    # Test adding the same dependency multiple times
    ts = TopoSorter()
    ts.add("1", "2")
    ts.add("1", "2")
    ts.add("1", "2")
    assert [*_static_order_with_groups(ts)] == [("2",), ("1",)]


def test_add_dependencies_for_same_node_incrementally():
    # Test same node multiple times
    ts = TopoSorter()
    ts.add("1", "2")
    ts.add("1", "3")
    ts.add("1", "4")
    ts.add("1", "5")

    ts2 = TopoSorter(edges=_graphlib_init_to_noob({"1": {"2", "3", "4", "5"}}))
    assert [*_static_order_with_groups(ts)] == [*_static_order_with_groups(ts2)]


def test_empty():
    _test_graph({}, [])


def test_cycle():
    # Self cycle
    _assert_cycle({"1": {"1"}}, ["1", "1"])
    # Simple cycle
    _assert_cycle({"1": {"2"}, "2": {"1"}}, ["2", "1", "2"])
    # Indirect cycle
    _assert_cycle({"1": {"2"}, "2": {"3"}, "3": {"1"}}, ["2", "1", "3", "2"])
    # not all elements involved in a cycle
    _assert_cycle(
        {"1": {"2"}, "2": {"3"}, "3": {"1"}, "5": {"4"}, "4": {"6"}}, ["2", "1", "3", "2"]
    )
    # Multiple cycles
    _assert_cycle(
        {"1": {"2"}, "2": {"1"}, "3": {"4"}, "4": {"5"}, "6": {"7"}, "7": {"6"}}, ["2", "1", "2"]
    )
    # Cycle in the middle of the graph
    _assert_cycle({"1": {"2"}, "2": {"3"}, "3": {"2", "4"}, "4": {"5"}}, ["2", "3", "2"])


def test_invalid_nodes_in_done():
    ts = TopoSorter()
    ts.add("1", "2", "3", "4")
    ts.add("2", "3", "4")
    ts.get_ready()

    with pytest.raises(ValueError, match="node '2' was not passed out"):
        ts.done("2")
    with pytest.raises(ValueError, match=r"node '24' was not added using add\(\)"):
        ts.done("24")


def test_done():
    ts = TopoSorter()
    ts.add("1", "2", "3", "4")
    ts.add("2", "3")

    assert set(ts.get_ready()) == {"3", "4"}
    # If we don't mark anything as done, get_ready() returns nothing
    assert ts.get_ready() == ()
    ts.done("3")
    # Now "2" becomes available as "3" is done
    assert ts.get_ready() == ("2",)
    assert ts.get_ready() == ()
    ts.done("4")
    ts.done("2")
    # Only "1" is missing
    assert ts.get_ready() == ("1",)
    assert ts.get_ready() == ()
    ts.done("1")
    assert ts.get_ready() == ()
    assert not ts.is_active()


def test_is_active():
    ts = TopoSorter()
    ts.add("1", "2")

    assert ts.is_active()
    assert ts.get_ready() == ("2",)
    assert ts.is_active()
    ts.done("2")
    assert ts.is_active()
    assert ts.get_ready() == ("1",)
    assert ts.is_active()
    ts.done("1")
    assert not ts.is_active()


def test_not_hashable_nodes():
    ts = TopoSorter()
    with pytest.raises(TypeError):
        ts.add(dict(), "1")
    with pytest.raises(TypeError):
        ts.add("1", dict())
    with pytest.raises(TypeError):
        ts.add(dict(), dict())


def test_order_of_insertion_does_not_matter_between_groups():
    def get_groups(ts) -> Generator[tuple, None, None]:
        while ts.is_active():
            nodes = ts.get_ready()
            ts.done(*nodes)
            yield set(nodes)

    ts = TopoSorter()
    ts.add("3", "2", "1")
    ts.add("1", "0")
    ts.add("4", "5")
    ts.add("6", "7")
    ts.add("4", "7")

    ts2 = TopoSorter()
    ts2.add("1", "0")
    ts2.add("3", "2", "1")
    ts2.add("4", "7")
    ts2.add("6", "7")
    ts2.add("4", "5")

    assert list(get_groups(ts)) == list(get_groups(ts2))
