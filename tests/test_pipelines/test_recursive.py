from noob import SynchronousRunner, Tube


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
