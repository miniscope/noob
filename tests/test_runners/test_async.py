import string

import pytest

from noob import Tube
from noob.runner import AsyncRunner


@pytest.mark.asyncio
async def test_async_base():
    tube = Tube.from_specification("testing-async")
    runner = AsyncRunner(tube)
    await runner.init()

    # nodes execute asynchronously, but return should still order the results
    for i in range(10):
        res = await runner.process()
        assert res == (
            string.ascii_lowercase[i],
            string.ascii_lowercase[i + 1],
            string.ascii_lowercase[i + 2],
        )

    await runner.deinit()
