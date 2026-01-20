import pytest

from noob.tube import Tube

from .paths import PIPELINE_DIR


@pytest.fixture(
    params=[pytest.param(p, id=p.stem) for p in (PIPELINE_DIR / "basic").rglob("*.y*ml")]
)
def basic_tubes(request: pytest.FixtureRequest) -> str:
    """
    Tubes that do not take input
    """
    if "map" in request.param.name:
        pytest.xfail("map not implemented")
    return request.param


@pytest.fixture()
def loaded_tube(request: pytest.FixtureRequest) -> Tube:
    return Tube.from_specification(request.param)
