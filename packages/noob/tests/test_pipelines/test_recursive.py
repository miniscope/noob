from noob import SynchronousRunner, Tube
from noob.event import MetaSignal
from noob.types import Epoch


def test_recursive_pipeline():
    """A tube with another tube inside of it!"""
    child_start = 1
    parent_start = 3
    child_multiply = 5
    parent_multiply = 7

    tube = Tube.from_specification(
        "testing-recursive-parent", input={"child_start": child_start, "parent_start": parent_start}
    )
    runner = SynchronousRunner(tube)

    for i, parent, child in zip(range(5), range(5, 10), range(10, 15)):
        res = runner.process(
            child_multiply=child + child_multiply, parent_multiply=parent + parent_multiply
        )
        expected_child = (parent_start + i) * (child_start + i) * (child + child_multiply)
        expected_parent = expected_child * (parent + parent_multiply)

        assert res["index"] == parent_start + i
        assert res["child"] == expected_child
        assert res["parent"] == expected_parent


def test_recursive_signals():
    """
    A parent tube can depend on the return values child tube when
    it has a dict-like return node
    """
    tube = Tube.from_specification("testing-recursive-signals")
    runner = SynchronousRunner(tube)
    for i in range(5):
        res = runner.process()
        assert res["multiply"] == i * 2 * 2
        assert res["divide"] == i / 5 / 5


def test_recursive_sporadic():
    """
    NoEvents from the return in the child tube should propagate to the parent
    """
    tube = Tube.from_specification("testing-recursive-sporadic-parent")
    runner = SynchronousRunner(tube)
    with runner:
        for i in range(5):
            res = runner.process()
            if (i + 1) % 3 != 0:
                assert res is None
                assert (
                    runner.store.events[Epoch(i)]["child"]["word"][0]["value"] == MetaSignal.NoEvent
                )
            else:
                assert isinstance(res, dict)
                assert "word" in res
                assert "doubleword" in res
                assert "word" in runner.store.events[Epoch(i)]["child"]
                assert res["doubleword"] == res["word"] * 2
