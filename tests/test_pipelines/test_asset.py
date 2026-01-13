import pytest

from noob import SynchronousRunner, Tube


def test_runner_scoped():
    """
    say we have an asset that just provides an integer, and the node just increments it.
    we should be able to call Runner.process multiple times, and the asset should
    increment without resetting.
    """
    tube = Tube.from_specification("testing-class-asset")
    runner = SynchronousRunner(tube=tube)

    output = runner.run(n=5)


def test_process_scoped():
    """
    same setup as above, except we should get back the same integer each time
    (it is reset between process calls, but also test that it is mutable within a
    process call - e.g. if we had two nodes that increment the value, it should return 2)

    inplace operations, only.
    """
    tube = Tube.from_specification("testing-func-asset")
    runner = SynchronousRunner(tube=tube)

    runner.init()
    for _ in range(5):
        output = runner.process()
        curr_val = output.__reduce__()[1][0]
        assert curr_val == 2


@pytest.mark.xfail(raises=NotImplementedError)
def test_node_scoped():
    """
    new object created each time, ensure object ids are all different
    """
    raise NotImplementedError()
