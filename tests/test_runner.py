from noob import SynchronousRunner, Tube


def test_process_callback() -> None:
    """
    callbacks must take place after each event emission.
    """

    tube = Tube.from_specification("testing-basic")
    runner = SynchronousRunner(tube)

    runner.init()
    events = []

    def _cb(event) -> None:
        nonlocal events
        events.append(event)

    runner.add_callback(_cb)
    runner.process()

    assert len(events) == 4


def test_disabled_node() -> None:
    tube = Tube.from_specification("testing-disable-node")
    runner = SynchronousRunner(tube)

    assert set(runner.tube.enabled_nodes.keys()) == {"a", "c"}

    all_events = []

    def _cb(event) -> None:
        nonlocal all_events
        all_events.append(event)

    runner.add_callback(_cb)
    runner.run()
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
