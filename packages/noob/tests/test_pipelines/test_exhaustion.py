"""
All runners should handle node exhaustion!
"""

import asyncio
import time
from typing import Any

import pytest

from noob.exceptions import NodeExhaustedError
from noob.runner import AsyncRunner
from noob.runner.zmq import ZMQRunner
from noob.utils import iscoroutinefunction_partial


@pytest.mark.asyncio
@pytest.mark.parametrize("loaded_tube", ["testing-exhaustion"], indirect=True)
async def test_exhaustion_process(loaded_tube, all_runners):
    """
    Runners should handle node exhaustion by erroring on process calls after exhaustion
    """
    runner = all_runners
    # first ones should work fine
    for _ in range(3):
        if iscoroutinefunction_partial(runner.process):
            res = await runner.process()
        else:
            res = runner.process()

    # the next one should raise
    with pytest.raises(NodeExhaustedError):
        if iscoroutinefunction_partial(runner.process):
            res = await runner.process()
        else:
            res = runner.process()


@pytest.mark.asyncio
@pytest.mark.parametrize("loaded_tube", ["testing-exhaustion"], indirect=True)
async def test_exhaustion_iter(loaded_tube, all_runners):
    """
    Runners should stop iteration on node exhaustion
    """
    results = []
    runner = all_runners
    if iscoroutinefunction_partial(runner.iter):
        async for result in runner.iter():
            results.append(result)
    else:
        for result in runner.iter():
            results.append(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("loaded_tube", ["testing-exhaustion"], indirect=True)
async def test_exhaustion_run(loaded_tube, all_runners):
    """
    Runners should stop running on node exhaustion
    """
    runner = all_runners
    if isinstance(runner, ZMQRunner):
        runner.autoclear_store = False
    elif isinstance(runner, AsyncRunner) and not iscoroutinefunction_partial(runner.run):
        pytest.xfail("Need to implement run method on async runner")

    results = []

    def store_result(event: Any) -> None:
        nonlocal results
        results.append(event)

    runner.add_callback(store_result)

    if iscoroutinefunction_partial(runner.run):
        await runner.run()
    else:
        runner.run()

    if isinstance(runner, ZMQRunner):
        # zmqrunner doesn't block on run
        stop_waiting = time.time() + 5
        while time.time() < stop_waiting and runner.running:
            await asyncio.sleep(0.1)

    a_events = [r for r in results if r["node_id"] == "a"]
    assert len(a_events) == 3
