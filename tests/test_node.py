from collections.abc import Generator

import pytest

from noob.event import is_event
from noob.node import Gather, Map, Return
from noob.node.base import NodeSpecification, WrapClassNode, WrapFuncNode
from noob.runner import SynchronousRunner
from noob.tube import Tube, TubeSpecification
from noob.types import Epoch


class SomeNode:
    def process(self) -> int:
        return 1


def func_node() -> int:
    return 1


def generator_node() -> Generator[int, None, None]:
    yield 1


@pytest.mark.xfail
def test_node_missing_param_input():
    """
    When a node specifies an input in its params, but is not provided an input collection,
    it raises an error.
    """
    raise NotImplementedError()


@pytest.mark.parametrize(
    "node,key,wrapper,default",
    (
        (SomeNode, "cls", WrapClassNode, True),
        (func_node, "fn", WrapFuncNode, False),
        (generator_node, "fn", WrapFuncNode, True),
    ),
)
def test_node_default_statefulness(node, key, wrapper, default):
    """
    Class nodes and generators default stateful, function nodes default stateless,
    and all can be overridden if passed explicitly
    """
    kwargs = {key: node, "id": "tmp"}
    wrapped = wrapper(**kwargs)
    assert wrapped.stateful == default

    assert wrapper(stateful=True, **kwargs).stateful
    assert not wrapper(stateful=False, **kwargs).stateful


def test_map_emits_subepochs():
    """The Map node returns pre-formed events in sub-epochs"""
    node = Map(id="map")
    events, count = node.process([1, 2, 3], epoch=Epoch(0))
    assert count == 3
    assert all(is_event(e) for e in events)
    for i in range(3):
        assert events[i]["epoch"] == Epoch(0) / ("map", i)


def test_gather_collapses_subepochs():
    """When gathering events in subepochs, gather emits in the parent epoch"""
    node = Gather(id="gather", n=3)
    parent = Epoch(0) / ("something", 0)
    val = None
    for i in range(3):
        val = node.process(i, epoch=parent / ("subepoch", i))

    assert is_event(val)
    assert val["epoch"] == parent


def test_return_deduplicates_by_epoch():
    """Return node only returns one of every value per epoch"""
    node = Return(
        id="return",
        spec=NodeSpecification(type="return", id="return", depends=[{"value": "a.test"}]),
    )
    node.process(test=1, _Return__events={"test": {"node_id": "a", "epoch": Epoch(0), "value": 1}})
    node.process(test=2, _Return__events={"test": {"node_id": "a", "epoch": Epoch(0), "value": 1}})
    assert node.get(keep=False)["test"] == 1


def test_di_epoch():
    """Epoch can be gotten via dependency injection"""
    spec = TubeSpecification(
        noob_id="di", nodes={"di": NodeSpecification(id="di", type="noob.testing.inject_epoch")}
    )
    tube = Tube.from_specification(spec)
    runner = SynchronousRunner(tube)
    runner.process()
    assert runner.store.events[Epoch(0)]["di"]["value"][0]["value"] == Epoch(0)


def test_di_eventmap():
    """An eventmap can be gotten via dependency injection"""
    spec = TubeSpecification(
        noob_id="di",
        nodes={
            "ep": NodeSpecification(id="ep", type="noob.testing.inject_epoch"),
            "di": NodeSpecification(
                id="di",
                type="noob.testing.inject_eventmap",
                depends=[{"special_value": "ep.value"}],
            ),
        },
    )
    tube = Tube.from_specification(spec)
    runner = SynchronousRunner(tube)
    runner.process()
    val = runner.store.events[Epoch(0)]["di"]["value"][0]["value"]
    assert "special_value" in val
    assert is_event(val["special_value"])
