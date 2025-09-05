from noob import SynchronousRunner, Tube


def test_process_callback() -> None:

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
