from datetime import datetime
from multiprocessing import Queue
from queue import Empty

import pytest

from noob import SynchronousRunner, Tube
from noob.event import Event, MetaEventType
from noob.scheduler import TopoSorter


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


def test_tube_increments_epoch(no_input_tubes):
    """
    Multiple runs of a tube should increment the epoch
    """
    tube = Tube.from_specification(no_input_tubes)
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
        assert runner.store.get(node_id="b", signal="value", epoch=i)["value"] == out


def test_epoch_completion(no_input_tubes):
    """
    When all nodes within an epoch are complete, the Scheduler
    must emit an end-of-epoch event.

    """
    tube = Tube.from_specification(no_input_tubes)
    scheduler = tube.scheduler
    scheduler.add_epoch()

    eoe = None
    while scheduler.is_active(0):
        ready_nodes = scheduler.get_ready()
        for ready in ready_nodes:
            # until we're done, eoe should have been None in the last iteration
            assert eoe is None
            # marking the very last one as done should emit the end of epoch event
            # (we should also break out of the while loop after the last assignment)
            eoe = scheduler.done(epoch=0, node_id=ready["value"])

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
    assert list(scheduler[0].nodes) == ["a", "b", "c"]
    assert list(scheduler[1].nodes) == ["a", "b"]


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
    assert tube.scheduler.source_nodes == expected


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
    scheduler.add_epoch()
    assert not scheduler.sources_finished()
    for node_id in sources:
        scheduler.get_ready()
        # Default checks the latest epoch
        assert not scheduler.sources_finished()
        scheduler.done(epoch=-1, node_id=node_id)
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
        scheduler.done(epoch=0, node_id=node)
        if node != "c":
            assert len(scheduler._epochs) == 2
        else:
            assert scheduler[1]
            with pytest.raises(KeyError):
                _ = scheduler[0]


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
