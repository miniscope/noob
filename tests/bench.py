import pytest
from pytest_codspeed.plugin import BenchmarkFixture

from noob import Tube
from noob.runner.base import TubeRunner


def test_load_tube(benchmark: BenchmarkFixture) -> None:
    benchmark(lambda: Tube.from_specification("testing-kitchen-sink"))


@pytest.mark.parametrize("loaded_tube", ["testing-kitchen-sink"], indirect=True)
def test_kitchen_sink_process(benchmark: BenchmarkFixture, runner: TubeRunner) -> None:
    benchmark(lambda: runner.process())


@pytest.mark.parametrize("loaded_tube", ["testing-kitchen-sink"], indirect=True)
def test_kitchen_sink_run(benchmark: BenchmarkFixture, runner: TubeRunner) -> None:
    benchmark(lambda: runner.run(n=10))


@pytest.mark.parametrize("loaded_tube", ["testing-long-add"], indirect=True)
def test_long_add(benchmark: BenchmarkFixture, runner: TubeRunner) -> None:
    """
    ZMQ runner should be faster for tubes where nodes take a long time
    and there's lots of concurrency possibilities
    """
    benchmark(lambda: runner.process())


@pytest.mark.parametrize("loaded_tube", ["testing-kitchen-sink"], indirect=True)
def test_topo_sorter(benchmark: BenchmarkFixture, loaded_tube: Tube) -> None:
    """
    Our TopoSorter should not get uh slower
    """
    benchmark(lambda: _run_sorter(loaded_tube))


def _run_sorter(tube: Tube) -> None:
    epoch = tube.scheduler.add_epoch()
    sorter = tube.scheduler._epochs[epoch]
    while sorter.is_active():
        ready_nodes = sorter.get_ready()
        for node in ready_nodes:
            sorter.done(node)
