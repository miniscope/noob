import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from pytest_codspeed.plugin import BenchmarkFixture

from noob import Tube
from noob.runner.base import TubeRunner, call_async_from_sync
from noob.runner.zmq import ZMQRunner


def test_load_tube(benchmark: BenchmarkFixture) -> None:
    benchmark(lambda: Tube.from_specification("testing-kitchen-sink"))


@pytest.mark.parametrize("loaded_tube", ["testing-kitchen-sink"], indirect=True)
def test_kitchen_sink_process(benchmark: BenchmarkFixture, runner: TubeRunner) -> None:
    benchmark(lambda: runner.process())


@pytest.mark.parametrize("loaded_tube", ["testing-kitchen-sink"], indirect=True)
def test_kitchen_sink_run(benchmark: BenchmarkFixture, runner: TubeRunner) -> None:
    if isinstance(runner, ZMQRunner):
        pytest.skip("ZMQ runner freerun mode not supported yet")
    benchmark(lambda: runner.run(n=10))


@pytest.mark.parametrize("loaded_tube", ["testing-long-add"], indirect=True)
def test_long_add(benchmark: BenchmarkFixture, runner: TubeRunner) -> None:
    """
    ZMQ runner should be faster for tubes where nodes take a long time
    and there's lots of concurrency possibilities
    """
    benchmark(lambda: runner.process())


@pytest.mark.parametrize("make_executor", [True, False])
@pytest.mark.asyncio
async def test_on_demand_async_executor(benchmark: BenchmarkFixture, make_executor: bool) -> None:
    """
    Just checking how much overhead there is in creating a new executor.

    Test function is itself a coroutine because we only use executor
    when calling async from sync within async lol.
    """

    async def _wait() -> None:
        # just await anything, want to actually engage the eventloop but not take any static time
        await asyncio.sleep(0)

    exc = ThreadPoolExecutor(1) if make_executor else None

    benchmark(lambda: call_async_from_sync(_wait, executor=exc))
