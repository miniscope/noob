import pytest

from noob.tube import Tube
from noob.yaml import yaml

from .paths import PIPELINE_DIR

# all tubes except special cases
_special_case_names = ("disable_node", "cycle")
_all_tubes = [
    tube
    for tube in PIPELINE_DIR.rglob("*.y*ml")
    if not any(name in tube.name for name in _special_case_names)
]
_no_input_tubes = [t for t in _all_tubes if "input" not in yaml.load(t.read_text())]


@pytest.fixture(params=[pytest.param(p, id=p.stem) for p in _all_tubes])
def all_tubes(request: pytest.FixtureRequest) -> str:
    """
    All the test case tubes!
    """
    if "map" in request.param.name:
        pytest.xfail("map not implemented")
    return request.param


@pytest.fixture(params=[pytest.param(p, id=p.stem) for p in _no_input_tubes])
def no_input_tubes(request: pytest.FixtureRequest) -> str:
    """
    Tubes that do not take input
    """
    if "map" in request.param.name:
        pytest.xfail("map not implemented")
    return request.param


@pytest.fixture()
def loaded_tube(request: pytest.FixtureRequest) -> Tube:
    return Tube.from_specification(request.param)
