import pytest
from pydantic import ValidationError

from noob.asset import AssetScope, AssetSpecification
from noob.input import InputScope, InputSpecification
from noob.node.spec import NodeSpecification
from noob.tube import TubeSpecification


@pytest.mark.parametrize("dep", ["assets.fake", "input.fake", "fake.value"])
@pytest.mark.parametrize("form", ["string", "list[str]", "list[dict]"])
def test_dependencies_exist(dep, form):
    """Nodes, assets, and inputs that are depended on should exist"""
    if form == "list[str]":
        dep = [dep]
    elif form == "list[dict]":
        dep = [{"value": dep}]

    with pytest.raises(ValidationError, match=r".*does not exist.*"):
        TubeSpecification(
            nodes={"mynode": NodeSpecification(type="list", id="mynode", depends=dep)}
        )


@pytest.mark.parametrize(
    "spec",
    (
        {"nodes": {"right": NodeSpecification(id="wrong", type="list")}},
        {"input": {"right": InputSpecification(id="wrong", type="list", scope=InputScope.tube)}},
        {"assets": {"right": AssetSpecification(id="wrong", type="list", scope=AssetScope.runner)}},
    ),
)
def test_id_mismatch(spec):
    """
    An ID in an instantiated node specification should be the same as the key used in the nodes dict
    """
    with pytest.raises(ValidationError, match=r"Mismatch between id.*"):
        TubeSpecification(**spec)


@pytest.mark.parametrize(
    "dep",
    (
        "list",
        ".relative",
        "node.",
        "some.absolute.python.Identifier",
        "module:function",
    ),
)
@pytest.mark.parametrize("form", ["string", "list[str]", "list[dict]"])
def test_signal_identifier(dep, form):
    """Dependencies should have a {node}.{signal} identifier form"""
    if form == "list[str]":
        dep = [dep]
    elif form == "list[dict]":
        dep = [{"value": dep}]

    with pytest.raises(ValidationError):
        NodeSpecification(type="list", id="mynode", depends=dep)


def test_slots_unique():
    """
    Slots in a dependency spec should be unique
    """
    with pytest.raises(ValidationError, match=r"Duplicate.*"):
        NodeSpecification(type="list", id="mynode", depends=[{"a": "b.value"}, {"a": "c.value"}])
