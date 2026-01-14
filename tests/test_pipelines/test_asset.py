import pytest

from noob import SynchronousRunner, Tube


def test_runner_scoped():
    """
    'runner' scoped assets are persistent across process calls
    """
    tube = Tube.from_specification("testing-class-asset")
    runner = SynchronousRunner(tube=tube)

    n = 5
    output = runner.run(n=n)
    assert all(o is output[0] for o in output)
    assert output[0].current == n


def test_process_scoped():
    """
    'process' scoped assets are recreated every process call.
    however, a given asset is shared by nodes within a single process call.
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
