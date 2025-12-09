from unittest.mock import patch

from noob import SynchronousRunner, Tube
from noob.node import Return


def test_process_callback() -> None:
    """
    callbacks must take place after each event emission.
    """

    tube = Tube.from_specification("testing-basic")
    runner = SynchronousRunner(tube)

    runner.init()
    cb_events = []

    def _cb(event) -> None:
        nonlocal cb_events
        cb_events.append(event)

    runner.add_callback(_cb)
    runner.process()

    # Callback will get MetaEvents like epoch end,
    # but event store won't have them,
    # so the callback should have received one more event than the others
    assert len(cb_events) - 1 == len(runner.store.flat_events) == len(tube.nodes)
    assert len(cb_events) > 0


def test_disabled_node() -> None:
    tube = Tube.from_specification("testing-disable-node")
    runner = SynchronousRunner(tube)

    assert set(runner.tube.enabled_nodes.keys()) == {"a", "c"}

    all_events = []

    def _cb(event) -> None:
        nonlocal all_events
        all_events.append(event)

    runner.add_callback(_cb)
    runner.process()
    assert "b" not in {e["node_id"] for e in all_events}


def test_disabled_return() -> None:
    tube = Tube.from_specification("testing-disable-return")
    runner = SynchronousRunner(tube)

    assert set(runner.tube.enabled_nodes.keys()) == {"a", "b"}

    # Return node is disabled, so nothing comes out
    result = runner.run(n=100)
    assert result is None


def test_dynamic_disable_node() -> None:
    tube = Tube.from_specification("testing-basic")
    runner = SynchronousRunner(tube)

    iters = 100

    first = runner.run(n=iters)
    assert first == list(range(0, iters * 2, 2))

    runner.disable_node("b")
    assert runner.run(n=iters) is None

    runner.enable_node("b")
    result = runner.run(n=iters)
    # the tube has been running, just not returning anything
    start = 400
    expected = list(range(start, start + (iters * 2), 2))
    assert result == expected


def test_dynamic_disable_return() -> None:
    tube = Tube.from_specification("testing-basic")
    runner = SynchronousRunner(tube)

    iters = 100

    first = runner.run(n=iters)

    runner.disable_node("c")
    assert runner.run(n=iters) is None

    runner.enable_node("c")
    result = runner.run(n=iters)

    # continues from where it left off after the two run
    start = max(first) * 2 + 4
    expected = list(range(start, start + (iters * 2), 2))
    assert result == expected


def test_synch_unready_end_epoch():
    """
    A SynchronousRunner should end the current epoch and
    move to the next one (if there is one) if there are no
    more nodes ready in the current epoch. (Specifically,
    for cardinality reducing operations like `class: .Gather`.)

    """

    tube = Tube.from_specification("testing-gather-n")
    runner = SynchronousRunner(tube)

    n_iters = 5
    with patch("noob.scheduler.Scheduler.end_epoch") as end_epoch:
        for _ in runner.iter(n=n_iters):
            pass

        assert end_epoch.call_count == n_iters * tube.nodes["b"].n
