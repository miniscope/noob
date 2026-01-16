import asyncio
import string

import pytest

from noob import SynchronousRunner, Tube
from noob.runner import TubeRunner
from noob.utils import iscoroutinefunction_partial


@pytest.mark.parametrize("loaded_tube", ["testing-async"], indirect=True)
def test_async_in_sync(loaded_tube: Tube):
    """
    The sync runner should be able to run async nodes when there is no outer eventloop
    """
    with pytest.raises(RuntimeError):
        # there shouldn't be any eventloop running!
        asyncio.get_running_loop()

    runner = SynchronousRunner(loaded_tube)
    with runner:
        for i in range(10):
            res = runner.process()
            assert res == (
                string.ascii_lowercase[i],
                string.ascii_lowercase[i + 1],
                string.ascii_lowercase[i + 2],
            )


@pytest.mark.parametrize("loaded_tube", ["testing-async"], indirect=True)
@pytest.mark.asyncio
async def test_async_in_async(loaded_tube: Tube, all_runners: TubeRunner) -> None:
    """
    All runners should be able to handle async nodes when run in an outer eventloop
    """
    runner = all_runners
    for i in range(10):
        if iscoroutinefunction_partial(runner.process):
            res = await runner.process()
        else:
            res = runner.process()
        assert res == (
            string.ascii_lowercase[i],
            string.ascii_lowercase[i + 1],
            string.ascii_lowercase[i + 2],
        )


@pytest.mark.parametrize("loaded_tube", ["testing-async-error"], indirect=True)
@pytest.mark.asyncio
async def test_async_errors(loaded_tube: Tube, all_runners: TubeRunner) -> None:
    """
    All runners can correctly raise async errors without hanging
    """
    runner = all_runners
    with pytest.raises(ValueError, match="This is the error"):
        if iscoroutinefunction_partial(runner.process):
            res = await runner.process()
        else:
            res = runner.process()
