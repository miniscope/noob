from datetime import datetime
from graphlib import TopologicalSorter
from multiprocessing import Queue
from queue import Empty

import pytest

from noob import SynchronousRunner, Tube
from noob.event import Event, MetaEventType


def test_epoch_increment():
    """
    Must be able to increment epoch on a Scheduler, which generates a graph
    each time.

    """
    tube = Tube.from_specification("testing-basic")
    scheduler = tube.scheduler

    with pytest.raises(KeyError):
        assert scheduler[0]

    for i in range(5):
        scheduler.add_epoch()
        assert isinstance(scheduler[i], TopologicalSorter)


def test_event_store_filter():
    """
    Must be able to filter events in EventStore by epoch.

    """
    tube = Tube.from_specification("testing-basic")
    runner = SynchronousRunner(tube)
    gen = runner.iter(5)
    for i, out in enumerate(gen):
        assert runner.store.get(node_id="b", signal="value", epoch=i)["value"] == out


def test_epoch_completion():
    """
    When all nodes within an epoch are complete, the Scheduler
    must emit an end-of-epoch event.

    """
    tube = Tube.from_specification("testing-basic")
    scheduler = tube.scheduler
    scheduler.add_epoch()
    for node_id in scheduler.nodes:
        scheduler.get_ready()
        eoe = scheduler.done(epoch=0, node_id=node_id)
        if node_id != "c":
            assert eoe is None
        else:
            assert (
                isinstance(eoe["id"], int)
                and isinstance(eoe["timestamp"], datetime)
                and eoe["node_id"] == "meta"
                and eoe["signal"] == MetaEventType("EpochEnded")
                and eoe["epoch"] == 0
                and eoe["value"] == 0
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
    assert all(event["node_id"] != "meta" for event in runner.store.events)
