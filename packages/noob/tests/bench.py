import contextlib
from copy import deepcopy

import pytest
from pytest_codspeed.plugin import BenchmarkFixture

from noob import Tube
from noob.edge import Edge
from noob.exceptions import AlreadyDoneError
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
    with contextlib.suppress(AlreadyDoneError):
        tube.scheduler.done(epoch=epoch, node_id="meta", signal="previous_epoch")
    while tube.scheduler.is_active(epoch):
        ready = tube.scheduler.get_ready(epoch)
        if not ready:
            raise RuntimeError("Should not get stuck in an infinite ready loop")
        for r in ready:
            tube.scheduler.done(epoch=epoch, node_id=r['value'], with_signals=True)