from datetime import UTC, datetime
from itertools import product
from queue import Queue

import pytest

from noob import NodeSpecification, SynchronousRunner, Tube
from noob.edge import Edge
from noob.event import Event, MetaEventType
from noob.exceptions import EpochCompletedError
from noob.scheduler import Scheduler
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
            assert not eoe
            # marking the very last one as done should emit the end of epoch event
            # (we should also break out of the while loop after the last assignment)
            eoe = scheduler.done(epoch=Epoch(0), node_id=ready["value"])

    assert len(eoe) == 1
    eoe = eoe[0]

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
    assert set(scheduler[0]._node2info.keys()) == {
        ("meta", "previous_epoch"),
        ("a", "index"),
        "a",
        "b",
        ("b", "value"),
        "c",
    }
    assert set(scheduler[1]._node2info.keys()) == {
        ("meta", "previous_epoch"),
        ("a", "index"),
        "a",
        "b",
    }


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
    assert scheduler.sources_finished(epoch_1)


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


def test_epoch_log_is_set_like():
    """
    _epoch_log must support O(1) membership testing.
    After completing epochs, completed epochs must be present in the log
    and upcoming epochs must not be.
    """
    tube = Tube.from_specification("testing-basic")
    scheduler = tube.scheduler

    for epoch in range(5):
        scheduler.add_epoch()
        for node in scheduler.nodes:
            scheduler.get_ready(epoch=Epoch(epoch))
            scheduler.done(epoch=Epoch(epoch), node_id=node)

    for epoch in range(5):
        assert epoch in scheduler._epoch_log

    assert 999 not in scheduler._epoch_log


def test_epoch_log_trim_keeps_recent_epochs():
    """
    When the log is trimmed, the most recent (highest-numbered) epochs must
    be retained, and older epochs must be removed.
    """
    tube = Tube.from_specification("testing-basic")
    scheduler = tube.scheduler
    scheduler._epoch_log_trim_interval = 10
    scheduler._epoch_log_keep = 5
    for epoch in range(10):
        for ready in scheduler.iter_epoch(Epoch(epoch)):
            for r in ready:
                scheduler.done(epoch=Epoch(epoch), node_id=r["value"])
        # scheduler.add_epoch(epoch)
        # for node in scheduler.nodes:
        #     scheduler.get_ready(epoch=Epoch(epoch))
        #     scheduler.done(epoch=Epoch(epoch), node_id=node)

    remaining = scheduler._epoch_log
    assert remaining == {5, 6, 7, 8, 9}, f"Expected {{5..9}}, got {remaining}"
    for old_epoch in range(5):
        assert old_epoch not in scheduler._epoch_log


def test_epoch_log_out_of_order_trim():
    """
    Epochs that arrive out of order must not cause higher-numbered epochs
    to be cleared when trimming older ones.
    """
    tube = Tube.from_specification("testing-basic")
    scheduler = tube.scheduler
    scheduler._epoch_log_trim_interval = 10
    scheduler._epoch_log_keep = 5

    for epoch in reversed(range(10)):
        scheduler.add_epoch(Epoch(epoch))
    for epoch in reversed(range(10)):
        scheduler.end_epoch(Epoch(epoch))
    remaining = scheduler._epoch_log
    assert remaining == {5, 6, 7, 8, 9}, "Early arriving, high epoch keys incorrectly evicted"


def test_epoch_completed_out_of_order():
    """When epochs are completed out of order, we can still correctly test for completion"""
    tube = Tube.from_specification("testing-basic")
    scheduler = tube.scheduler
    scheduler.add_epoch(1)
    scheduler.add_epoch(10)
    scheduler.end_epoch(10)
    scheduler.end_epoch(1)

    assert not scheduler.epoch_completed(Epoch(2))


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

    # b is not ready
    assert "b" not in [r["value"] for r in scheduler.get_ready()]


@pytest.mark.map
def test_map_creating_subepochs_expires_parent_epoch():
    """
    When a node induces a map by creating subepochs,
    that node should be marked exhausted in the parent epoch
    """
    tube = Tube.from_specification("testing-map-basic")
    scheduler = tube.scheduler
    ep = scheduler.add_epoch()
    scheduler.done(ep, node_id="a")
    scheduler.update([_make_event(ep / ("b", i), "b") for i in range(3)])
    assert "b" in scheduler._epochs[ep].done_nodes
    assert "b" not in scheduler._epochs[ep].ran_nodes


@pytest.mark.map
def test_get_ready_yields_all_mapped_subepochs():
    """
    get_ready should yield all mapped subepochs at once when accessed with the parent epoch
    """
    tube = Tube.from_specification("testing-map-basic")
    scheduler = tube.scheduler
    ep = scheduler.add_epoch()
    scheduler.done(ep, node_id="a")
    scheduler.update([_make_event(ep / ("b", i), "b") for i in range(3)])
    ready = scheduler.get_ready(ep)
    assert len(ready) == 3
    assert {r["epoch"] for r in ready} == {ep / ("b", i) for i in range(3)}


@pytest.mark.map
def test_map_gather_only_parent_epoch():
    """
    When a gather node collapses subepochs, nodes that are exclusively downstream of the gather node
    only run in the parent epoch
    """
    tube = Tube.from_specification("testing-map-gather")
    scheduler = tube.scheduler
    ep = scheduler.add_epoch()
    scheduler.done(ep, node_id="a")
    scheduler.update([_make_event(ep / ("b", i), "b") for i in range(3)])
    for i in range(3):
        scheduler.done(ep / ("b", i), node_id="c")
    scheduler.done(ep, node_id="d")
    ready = scheduler.get_ready(ep)
    assert len(ready) == 1
    assert ready[0]["epoch"] == ep


@pytest.mark.map
def test_map_gather_mixed_epochs():
    """
    When a gather node collapses subepochs,
    it yields nodes that are exclusively downstream of the gather node in the parent epoch,
    and nodes that are in the mapped subepoch in the subepochs.
    """
    tube = Tube.from_specification("testing-map-gather")
    scheduler = tube.scheduler
    ep = scheduler.add_epoch()
    scheduler.done(ep, node_id="a")
    scheduler.update([_make_event(ep / ("b", i), "b") for i in range(3)])
    for i in range(3):
        scheduler.done(ep / ("b", i), node_id="c", signal="value")
    for i in range(3):
        scheduler.expire(ep / ("b", i), node_id="d", signal="value", unlock_optionals=False)
    scheduler.done(ep, node_id="d", signal="value")
    scheduler.done(ep, node_id="e")
    ready = scheduler.get_ready(ep)
    assert len(ready) == 3
    assert {r["epoch"] for r in ready} == {ep / ("b", i) for i in range(3)}


@pytest.mark.map
def test_is_active_when_subepochs_active():
    """Scheduler stays active for an epoch as long as there are active subepochs"""
    tube = Tube.from_specification("testing-map-basic")
    scheduler = tube.scheduler
    ep = scheduler.add_epoch()
    scheduler.done(ep, node_id="a")
    scheduler.update([_make_event(ep / ("b", i), "b") for i in range(3)])
    assert not scheduler._epochs[ep].is_active()
    assert scheduler._epochs[ep / ("b", 0)].is_active()
    assert scheduler.is_active(ep)
    scheduler.update([_make_event(ep / ("b", i), "c") for i in range(3)])
    assert not scheduler._epochs[ep].is_active()
    assert scheduler._epochs[ep / ("b", 0)].is_active()
    assert scheduler.is_active(ep)
    for i in range(3):
        scheduler.expire(ep / ("b", i), node_id="return")
        if i != 2:
            # once all the subepochs are done, the parent epoch is done
            assert scheduler.is_active(ep)
        assert not scheduler.is_active(ep / ("b", i))
    assert not scheduler.is_active(ep)


def _make_event(epoch: Epoch, node_id: str, signal: str = "value") -> Event:
    return Event(
        id=0, epoch=epoch, node_id=node_id, timestamp=datetime.now(), signal=signal, value=0
    )


def test_asset_generations():
    """Topological generations with respect to an asset"""
    tube = Tube.from_specification("testing-asset-generations")
    scheduler = tube.scheduler
    generations = scheduler.asset_generations()
    for asset in ("count_process", "count_runner_depends", "count_runner"):
        if asset == "count_runner_depends":
            assert len(generations[asset]) == 4
            assert set(generations[asset][3]) == {"d1"}
        else:
            assert len(generations[asset]) == 3
        assert set(generations[asset][0]) == {"a1", "a2"}
        assert set(generations[asset][1]) == {"b1", "b2"}
        assert set(generations[asset][2]) == {"c1", "c2"}


def test_upstream_nodes(optional_graph):
    """
    Upstream nodes detects immediate dependencies as well as nodes who can affect our running
    by indirect optional dependencies
    (i.e., if they emitted NoEvent, we would be eligible for running)
    """
    sched = Scheduler(edges=optional_graph, nodes={})
    # d should implicitly have b upstream but not a, because a->b is optional,
    # despite only directly depending on c
    # the rest are just direct dependencies and that's the `Node.edges` method, so not tested here.
    assert not any(e.source_node == "b" and e.target_node == "d" for e in optional_graph)
    assert sched.upstream_nodes("d") == {"b", "c"}


def test_subepoch_generation_race_condition():
    """
    Nodes in a parent epoch are not incorrectly run in subepochs
    when a race condition exists between the `map` node and the parent epoch nodes.

    References:
        https://github.com/miniscope/noob/issues/193

    """
    tube = Tube.from_specification("testing-map-depends")
    scheduler = tube.scheduler
    ep = scheduler.add_epoch()
    ready = scheduler.get_ready(ep)
    assert set([r["value"] for r in ready]) == {"word", "count"}

    # word completes first and is then mapped
    scheduler.done(ep, node_id="word")
    ready = scheduler.get_ready()
    assert set([r["value"] for r in ready]) == {"map"}

    # simulate the map events creating subepochs
    subepochs = ep.make_subepochs("map", 3)
    events = [
        Event(
            id=i, timestamp=datetime.now(UTC), node_id="map", signal="value", epoch=subep, value=i
        )
        for i, subep in enumerate(subepochs)
    ] + [Event(id=3, timestamp=datetime.now(UTC), node_id="map", signal="n", epoch=ep, value=3)]
    scheduler.update(events)

    # next node in subepoch "exclaim" depends on count, which is not done yet
    # we should not have accidentally yielded count in the subepochs here
    # because it wasn't done in the parent.
    ready = scheduler.get_ready(ep)
    assert len(ready) == 0

    # then when count is done, all the exclaim nodes in the subepoch should be ready.
    scheduler.update(
        [
            Event(
                id=4,
                timestamp=datetime.now(UTC),
                node_id="count",
                signal="index",
                epoch=ep,
                value=4,
            )
        ]
    )
    ready = scheduler.get_ready(ep)
    assert len(ready) == 3
    assert all(len(r["epoch"]) > 1 for r in ready)
    assert set(r["value"] for r in ready) == {"exclaim"}


def test_non_equivalent_event(non_equivalent_event):
    """
    Regression - we can handle events that have a value that can't use ==

    References:
        https://github.com/miniscope/noob/issues/192
        https://github.com/miniscope/noob/issues/201
    """
    scheduler = Scheduler(
        edges=[
            Edge(
                source_node="default",
                source_signal="value",
                target_node="target",
                target_slot="value",
            )
        ],
        nodes={
            "default": NodeSpecification(id="default", type="noob.testing.notreal"),
            "target": NodeSpecification(id="target", type="noob.testing.notreal"),
        },
    )
    scheduler.update([non_equivalent_event])


def test_statefulness():
    """
    When iterating events, we don't yield stateful nodes until they have been run in
    a previous epoch.
    """
    tube = Tube.from_specification("testing-stateful")
    scheduler = tube.scheduler
    was_ready = []
    for i in range(3):
        scheduler.add_epoch(Epoch(i))

    scheduler.done(epoch=Epoch(2), node_id="input")
    scheduler.done(epoch=Epoch(1), node_id="input")

    # add some guards against infinite iteration
    for i, ready in enumerate(scheduler.iter_ready()):
        # the inputs should all be ready, as should 'c' in epoch 0
        ready = sorted(ready, key=lambda item: item["value"])
        assert len(ready) == 2
        assert ready[0]["value"] == "c"
        assert ready[0]["epoch"] == Epoch(0)
        assert ready[1]["value"] == "input"
        assert ready[1]["epoch"] == Epoch(0)

        if i >= 1:
            raise RuntimeError("should only have iterated once!")

    # now we should get all the events from three epochs
    scheduler.done(epoch=Epoch(0), node_id="c")
    scheduler.done(epoch=Epoch(0), node_id="input")
    expected_total = len(tube.nodes) * 3
    expected_epoch = Epoch(0)
    for ready in scheduler.iter_ready():
        was_ready.extend(ready)
        for r in ready:
            # run the epochs in order
            assert r["epoch"] == expected_epoch or (
                r["value"] == "input" and r["epoch"] == expected_epoch + 1
            )
            scheduler.done(epoch=r["epoch"], node_id=r["value"])
            if r["value"] == "e":
                expected_epoch = Epoch(expected_epoch[0].epoch + 1)
        if expected_epoch >= 3:
            break

    assert len(was_ready) == expected_total


def test_statelessness():
    """
    When iterating events, we can run stateless nodes out of order, in any epoch,
    as long as their deps are met.
    """
    tube = Tube.from_specification("testing-stateful")
    scheduler = tube.scheduler

    # when just iterating without any input, for now with a single epoch-ready event,
    # should just get the generator in the first round since it doesn't require input,
    # but then in the second round we just get the 'input' ready event
    #
    for i, ready in enumerate(scheduler.iter_ready()):
        if i == 0:
            assert len(ready) == 2
            ready = sorted(ready, key=lambda item: item["value"])
            assert ready[0]["value"] == "c"
            assert ready[0]["epoch"] == Epoch(i)
            scheduler.done(epoch=ready[0]["epoch"], node_id=ready[0]["value"])
        elif i == 1:
            assert len(ready) == 1
            assert ready[0]["value"] == "input"
        if i >= 2:
            break

    # now if we give some input out of order, we run the nodes in that epoch
    scheduler.done(epoch=Epoch(2), node_id="input")

    for i, ready in enumerate(scheduler.iter_ready()):
        if i == 0:
            # should have gotten a, b, and d from epoch 2, and c from epoch 3
            assert len(ready) == 3
            for node in ("a", "b", "d"):
                assert any(r["node_id"] == node and r["epoch"] == Epoch(2) for r in ready)
            assert any(r["node_id"] == "c" and r["epoch"] == Epoch(3) for r in ready)

        elif i == 1:
            # now we should just get the return and another count
            assert len(ready) == 2
            assert any(r["node_id"] == "e" and r["epoch"] == Epoch(2) for r in ready)
            assert any(r["node_id"] == "c" and r["epoch"] == Epoch(4) for r in ready)

        elif i == 2:
            # and now that epoch 2 is exhausted, should just have the generator
            assert scheduler.epoch_completed(Epoch(2))
            assert len(ready) == 1
            assert ready[0]["node_id"] == "c" and ready[0]["epoch"] == Epoch(5)

        else:
            break

        for r in ready:
            scheduler.done(epoch=r["epoch"], node_id=r["node_id"])


def test_statelessness_source():
    """
    When we have no inputs and a stateless source, the stateless source can just go for it,
    but a stateful dependent will run in order
    """
    tube = Tube.from_specification("testing-stateless-source")
    scheduler = tube.scheduler
    all_ready = []

    for i in range(5):
        scheduler.add_epoch(Epoch(i))

    # this shouldn't really happen, but would technically be correct -
    # a stateless node should be able to run in parallel like this
    _i = -1
    for _i, ready in enumerate(scheduler.iter_ready()):
        all_ready.extend(ready)
        assert len(ready) == 5
        assert all(r["value"] == "a" for r in ready)
        break
    assert _i != -1, "did not iterate!"

    # marking the source ready out of order does not ready the stateful node
    scheduler.done(epoch=Epoch(2), node_id="a")
    for _ in scheduler.iter_ready():
        raise RuntimeError("Shouldn't have iterated")

    # marking the source node completed in the max epoch adds another epoch and yields it as ready
    scheduler.done(epoch=Epoch(4), node_id="a")
    i = -1
    for i, ready in enumerate(scheduler.iter_ready()):
        all_ready.extend(ready)
        assert len(ready) == 1
        assert ready[0]["value"] == "a"
        assert ready[0]["epoch"] == Epoch(5)
        if i >= 1:
            raise RuntimeError("Should only have iterated once")
    assert i != -1, "did not iterate!"

    # now marking the source node complete in order readies the stateful dependent one by one
    for i in range(6):
        if i != 2:
            scheduler.done(epoch=Epoch(i), node_id="a")
        j = -1
        for j, ready in enumerate(scheduler.iter_ready()):
            all_ready.extend(ready)
            ready = sorted(ready, key=lambda item: item["epoch"])

            # in the last iteration, we get the next 'a' epoch (6)
            if i == 5:
                assert len(ready) == 2
            else:
                assert len(ready) == 1
            assert ready[0]["value"] == "b"

            # when we reach epoch 1 and 3, epoch 2 and 4 have already been done, so we iter twice
            if i in (1, 3):
                assert ready[0]["epoch"] in (Epoch(i), Epoch(i + 1))
                assert j <= 1, "should have iterated at most twice"
            else:
                assert ready[0]["epoch"] == Epoch(i)
                assert j <= 0, "should have only iterated once!"

            scheduler.done(epoch=Epoch(ready[0]["epoch"]), node_id="b")
        assert j != -1 or i in (2, 4), "did not iterate!"

    # finally, assert that we got all the readies we expect
    expected = {(node_id, Epoch(i)) for node_id, i in product(("a", "b"), range(6))}
    expected.add(("a", Epoch(6)))
    actual = {(r["value"], r["epoch"]) for r in all_ready}
    assert actual == expected
