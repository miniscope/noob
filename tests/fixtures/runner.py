import pytest
import pytest_asyncio

from noob import Tube
from noob.runner import AsyncRunner, SynchronousRunner, TubeRunner
from noob.runner.zmq import ZMQRunner
from noob.utils import iscoroutinefunction_partial


@pytest.fixture(
    params=[
        pytest.param(SynchronousRunner, marks=[pytest.mark.sync_runner]),
        pytest.param(ZMQRunner, marks=[pytest.mark.zmq_runner]),
    ]
)
def sync_runner_cls(request: pytest.FixtureRequest) -> type[TubeRunner]:
    return request.param


@pytest.fixture(
    params=[
        pytest.param(SynchronousRunner, marks=[pytest.mark.sync_runner]),
        pytest.param(ZMQRunner, marks=[pytest.mark.zmq_runner]),
        pytest.param(AsyncRunner, marks=[pytest.mark.async_runner]),
    ]
)
def all_runner_cls(request: pytest.FixtureRequest) -> type[TubeRunner]:
    return request.param


@pytest.fixture
def runner(loaded_tube: Tube, sync_runner_cls: type[TubeRunner]) -> TubeRunner:
    r = sync_runner_cls(loaded_tube)
    r.init()
    yield r
    r.deinit()


@pytest_asyncio.fixture
async def all_runners(loaded_tube: Tube, all_runner_cls: type[TubeRunner]) -> TubeRunner:
    """all runners including async runners"""
    r = all_runner_cls(loaded_tube)
    if isinstance(r, ZMQRunner):
        r.autoclear_store = False
    if iscoroutinefunction_partial(r.init):
        await r.init()
    else:
        r.init()
    yield r
    if iscoroutinefunction_partial(r.deinit):
        await r.deinit()
    else:
        r.deinit()
