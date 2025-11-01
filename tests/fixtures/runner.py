import pytest

from noob import Tube
from noob.runner import SynchronousRunner, TubeRunner, ZMQRunner


@pytest.fixture(
    params=[
        pytest.param(SynchronousRunner, marks=[pytest.mark.sync_runner]),
        pytest.param(ZMQRunner, marks=[pytest.mark.zmq_runner]),
    ]
)
def runner_cls(request: pytest.FixtureRequest) -> type[TubeRunner]:
    return request.param


@pytest.fixture
def runner(loaded_tube: Tube, runner_cls: type[TubeRunner]) -> TubeRunner:
    r = runner_cls(loaded_tube)
    r.init()
    yield r
    r.deinit()
