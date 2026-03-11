import json
from pathlib import Path

import pytest
from jsonschema import validate
from ruamel.yaml import YAML

REPO_ROOT = Path(__file__).parents[3]
TUBE_SCHEMA = REPO_ROOT / "schema" / "tube.schema.json"
EXAMPLES = list(REPO_ROOT.rglob("examples/**/*.y*ml"))
SPECIAL_TUBES = list(REPO_ROOT.rglob("packages/noob/tests/data/pipelines/special/**/*.y*ml"))
TEST_TUBES = list(REPO_ROOT.rglob("packages/noob/tests/data/pipelines/**/*.y*ml"))
NONSPECIAL_TUBES = set(TEST_TUBES) - set(SPECIAL_TUBES)
# special tubes are special because they sometimes do things like "be invalid"


@pytest.fixture(scope="module")
def schema() -> dict:
    with open(TUBE_SCHEMA) as f:
        data = json.load(f)
    return data


@pytest.mark.parametrize("spec", EXAMPLES)
def test_example_tubes_valid(spec: Path, schema) -> None:
    """
    All the examples should be valid against the current version of the tube schema!
    """
    yaml = YAML()
    with open(spec) as f:
        data = yaml.load(f)
    validate(instance=data, schema=schema)


@pytest.mark.parametrize("spec", NONSPECIAL_TUBES)
def test_test_tubes_valid(spec: Path, schema) -> None:
    """
    All the tubes used in tests should be valid against the current version of the tube schema!
    Except for "special" tubes, which include tubes with errors in them.
    """
    yaml = YAML()
    with open(spec) as f:
        data = yaml.load(f)
    validate(instance=data, schema=schema)
