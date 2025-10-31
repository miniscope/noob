import numpy as np
import pytest

from noob import SynchronousRunner, Tube
from noob.runner import TubeRunner


@pytest.mark.parametrize("loaded_tube", ["testing-basic"], indirect=True)
def test_basic_process(loaded_tube: Tube, runner: TubeRunner):
    """The most basic tube! We can process a fixed number of events"""
    outputs = []
    for _ in range(5):
        outputs.append(runner.process())
    assert len(outputs) == 5
    assert outputs == [0, 2, 4, 6, 8]


@pytest.mark.parametrize("loaded_tube", ["testing-basic"], indirect=True)
def test_basic(loaded_tube: Tube, runner: TubeRunner):
    """The most basic tube! We can process a fixed number of events"""
    outputs = runner.run(n=5)
    assert len(outputs) == 5
    assert outputs == [0, 2, 4, 6, 8]


@pytest.mark.parametrize("loaded_tube", ["testing-basic"], indirect=True)
def test_basic_iter(loaded_tube: Tube, runner: TubeRunner):
    """We should also be able to iterate over values"""
    expected = [0, 2, 4, 6, 8]
    for e, value in zip(expected, runner.iter(n=5)):
        assert value == e


@pytest.mark.parametrize("loaded_tube", ["testing-branch"], indirect=True)
def test_branch(loaded_tube: Tube, runner: TubeRunner):
    """A nodes output can be branched and received by multiple nodes!"""
    expected = [{"multiply": i * 2, "divide": i / 5} for i in range(5)]

    for e, value in zip(expected, runner.iter(n=5)):
        assert value == e


@pytest.mark.parametrize("loaded_tube", ["testing-merge"], indirect=True)
def test_merge(loaded_tube: Tube, runner: TubeRunner):
    """Multiple node outputs can be merged into one node!"""
    expected = [(i * 2) / j for i, j in zip(range(5), range(5, 10))]

    for e, value in zip(expected, runner.iter(n=5)):
        assert value == e


@pytest.mark.parametrize("loaded_tube", ["testing-gather-n"], indirect=True)
def test_gather_n(loaded_tube: Tube, runner: TubeRunner):
    """A node can gather n inputs into one call"""
    expected = ["abcde", "fghij", "klmno", "pqrst", "uvwxy"]

    for e, value in zip(expected, runner.iter(n=5)):
        assert value == {"word": e}


@pytest.mark.parametrize("loaded_tube", ["testing-gather-dependent"], indirect=True)
def test_gather_dependent(loaded_tube: Tube, runner: TubeRunner):
    """A node can gather inputs from one slot when another slot receives an event"""
    expected = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
        [12, 13, 14],
    ]

    for e, value in zip(expected, runner.iter(n=5)):
        assert isinstance(value, dict)
        assert len(value) == 1
        value = value["word"]
        inner = value[list(value.keys())[0]]
        assert inner == e


@pytest.mark.xfail(reason="map has not been implemented.")
def test_map():
    """
    A node with a sequence output can be mapped to a node with a scalar input

    In this case, "process" should know to iterate over the mapped values
    and return them one by one, so we should get n of the mapped values,
    not n calls of the source node -> n sets of mapped values.
    """
    tube = Tube.from_specification("testing-map")
    runner = SynchronousRunner(tube)

    for value in runner.iter(n=5):
        assert len(value) == 2
        assert isinstance(value, str)
        assert value[1] == "!"


@pytest.mark.parametrize("loaded_tube", ["testing-multi-signal"], indirect=True)
def test_multi_signal(loaded_tube: Tube, runner_cls: type[TubeRunner]):
    """
    Nodes that emit multiple signals can have each used independently
    """
    tube = Tube.from_specification("testing-multi-signal")
    runner = runner_cls(tube)

    for value in runner.iter(n=5):
        assert isinstance(value, dict)
        assert isinstance(value["word"], str)
        assert value["count_sum"] == sum(value["counts"])


def test_xarray_asset():
    """
    Test should verify that the asset has been modified in place,
    (two xarray dataarray assets have been summed and assigned to one of them)
    and the modified asset is same as the returned event output.
    """
    tube = Tube.from_specification("testing-xarray-asset")
    runner = SynchronousRunner(tube=tube)

    runner.init()
    output = runner.process()

    assert np.all(output == 2)


def test_db_asset():
    tube = Tube.from_specification("testing-db-asset")
    runner = SynchronousRunner(tube=tube)

    runner.init()
    output = runner.process()

    assert output == (1, "Hannah Montana")
