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

    assert len(events) == 2


def test_disabled_nodes() -> None:
    tube = Tube.from_specification("testing-enable-flag")
    runner = SynchronousRunner(tube)

    runner.init()
    assert set(runner.tube.enabled_nodes.keys()) == {"a", "b", "sink_on"}
    result = runner.process()
    assert result is None
