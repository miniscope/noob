from datetime import datetime
from multiprocessing import Queue
from queue import Empty

import pytest

from noob import SynchronousRunner, Tube
from noob.event import Event, MetaEventType
from noob.exceptions import EpochCompletedError
from noob.toposort import TopoSorter
from noob.types import Epoch


def test_epoch_increment():
    """
    Must be able to increment epoch on a Scheduler, which generates a graph
    each time.
    """
    tube = Tube.from_specification("testing-basic")
    scheduler = tube.scheduler

    # accessing makes the epoch
    assert isinstance(scheduler[0], TopoSorter)

    for i in range(5):
        assert scheduler.add_epoch() == i + 1
        assert isinstance(scheduler[i], TopoSorter)

    # if we create an epoch out of order, the next call without epoch specified increments
    assert scheduler.add_epoch(10) == 10
    assert scheduler.add_epoch() == 11


def test_tube_increments_epoch(basic_tubes):
    """
    Multiple runs of a tube should increment the epoch
    """
    tube = Tube.from_specification(basic_tubes)
    runner = SynchronousRunner(tube)

    for i in range(5):
        _ = runner.process()
        events = runner.store.events
        # we haven't cleared events
        assert len(events) == 1
        assert list(events.keys())[0] == i


def test_event_store_filter():
    """
    Must be able to filter events in EventStore by epoch.

    """
    tube = Tube.from_specification("testing-basic")
    runner = SynchronousRunner(tube)
    gen = runner.iter(5)
    for i, out in enumerate(gen):
        assert runner.store.get(node_id="b", signal="value", epoch=Epoch(i))["value"] == out


def test_epoch_completion(basic_tubes):
    """
    When all nodes within an epoch are complete, the Scheduler
    must emit an end-of-epoch event.

    """
    tube = Tube.from_specification(basic_tubes)
    scheduler = tube.scheduler
    scheduler.add_epoch()

    eoe = None
    while scheduler.is_active(Epoch(0)):
        ready_nodes = scheduler.get_ready()
        for ready in ready_nodes:
            # until we're done, eoe should have been None in the last iteration
            assert eoe is None
            # marking the very last one as done should emit the end of epoch event
            # (we should also break out of the while loop after the last assignment)
            eoe = scheduler.done(epoch=Epoch(0), node_id=ready["value"])

    assert (
        isinstance(eoe["id"], int)
        and isinstance(eoe["timestamp"], datetime)
        and eoe["node_id"] == "meta"
        and eoe["signal"] == MetaEventType("EpochEnded")
        and eoe["epoch"] == 0
    )


def test_tube_disable_node():
    """
    When a node gets disabled in a Tube, it takes effect
    for all the SUBSEQUENT epochs.

    """
    tube = Tube.from_specification("testing-basic")
    runner = SynchronousRunner(tube)
    assert runner.process() == 0
    runner.tube.disable_node("c")
    assert runner.process() is None


def test_scheduler_disable_node():
    """
    When a node gets disabled in a Scheduler, it takes effect
    for all the SUBSEQUENT epochs.

    """
    tube = Tube.from_specification("testing-basic")
    scheduler = tube.scheduler
    scheduler.add_epoch()
    scheduler.disable_node("c")
    scheduler.add_epoch()
    assert list(scheduler.nodes.keys()) == ["a", "b", "c"]
    assert list(scheduler[0]._node2info.keys()) == ["a", "b", "c"]
    assert list(scheduler[1]._node2info.keys()) == ["a", "b"]


@pytest.mark.parametrize(
    "spec, input, expected",
    [
        ("testing-basic", None, ["a"]),
        ("testing-merge", None, ["a", "b"]),
        ("testing-input-mixed", {"start": 7, "multiply_tube": 13}, ["a"]),
    ],
)
def test_identify_source_node(spec, input, expected):
    """
    Scheduler can correctly identify the source nodes of a tube.

    """
    tube = Tube.from_specification(spec, input)
    assert set(tube.scheduler.source_nodes) == set(expected)


@pytest.mark.parametrize(
    "spec, input, sources",
    [
        ("testing-basic", None, ["a"]),
        ("testing-merge", None, ["a", "b"]),
        ("testing-input-mixed", {"start": 7, "multiply_tube": 13}, ["a"]),
    ],
)
def test_source_node_completion(spec, input, sources):
    """
    Scheduler checks whether all source nodes of a given epoch is complete.

    """
    tube = Tube.from_specification(spec, input)
    scheduler = tube.scheduler
    # Epoch 0 won't be finished
    scheduler.add_epoch()
    # We will finish the latest epoch
    epoch_1 = scheduler.add_epoch()
    assert not scheduler.sources_finished()
    for node_id in sources:
        scheduler.get_ready()
        # Default checks the latest epoch
        assert not scheduler.sources_finished()
        scheduler.done(epoch=epoch_1, node_id=node_id)
    assert scheduler.sources_finished()


def test_clear_ended_epochs():
    """
    Scheduler must clear the cache of any epochs
    whose nodes have all completed.

    """
    tube = Tube.from_specification("testing-basic")
    scheduler = tube.scheduler
    # We will finish epoch 0
    scheduler.add_epoch()
    # Epoch 1 will not be finished
    scheduler.add_epoch()
    for node in scheduler.nodes:
        scheduler.get_ready()
        scheduler.done(epoch=Epoch(0), node_id=node)
        if node != "c":
            assert len(scheduler._epochs) == 2
        else:
            assert scheduler[1]
            with pytest.raises(EpochCompletedError):
                _ = scheduler[0]
            assert len(scheduler._epochs) == 1


@pytest.mark.xfail(raises=Empty)
def test_metaevents():
    """
    Meta events are emitted in callbacks, but not given to nodes or added to the store.

    marked `xfail` until Return node starts returning NoEvent instead of None.

    """
    queue = Queue()

    def callback(event: Event) -> bool:
        nonlocal queue

        if event["node_id"] == "meta":
            queue.put_nowait(event)

    tube = Tube.from_specification("testing-basic")
    runner = SynchronousRunner(tube)
    runner.add_callback(callback)
    runner.run(5)
    event = queue.get_nowait()
    assert event["node_id"] == "meta"
    assert len(runner.store.events) > 0
    assert all(event["node_id"] != "meta" for event in runner.store.flat_events)


@pytest.mark.xfail(raises=NotImplementedError)
def test_noevent_ends_epoch():
    """If a node in the middle of a tube emits a noevent, the epoch should no longer be active"""
    raise NotImplementedError("Write this test!")


def test_disable_nodes():
    """
    For A->B->C graph, when B is disabled,
    the scheduler only yields A and then becomes inactive,
    rather than yielding C.
    """
    tube = Tube.from_specification("testing-basic")
    scheduler = tube.scheduler
    scheduler.disable_node("b")
    epoch = scheduler.add_epoch()
    ready_nodes = scheduler.get_ready()

    # only "a" is returned, even though "b" also has no dependencies,
    # since "b" is disabled.
    assert {node["value"] for node in ready_nodes} == {"a"}
    scheduler.done(epoch=epoch, node_id="a")

    # nothing ready anymore
    assert not scheduler.get_ready()


def test_map_creating():
    pass
