import pytest

from noob import Tube
from noob.input import InputCollection, InputScope, InputSpecification

pytestmark = pytest.mark.input


def test_input_defaults():
    """
    Inputs are correctly accessed from the innermost scope through to defaults.
    """
    inputs = InputCollection(
        specs={
            InputScope.tube: {
                "a": InputSpecification(id="a", scope=InputScope.tube, default="z", type="str"),
                "b": InputSpecification(id="b", scope=InputScope.tube, default="y", type="str"),
            },
            InputScope.process: {
                "c": InputSpecification(id="c", scope=InputScope.tube, default="x", type="str"),
                "d": InputSpecification(id="d", scope=InputScope.tube, default="w", type="str"),
            },
        }
    )

    inputs.add_input(InputScope.tube, {"a": "aaa"})
    assert inputs.get("a") == "aaa"
    assert inputs.get("b") == "y"
    assert inputs.get("c", {"c": "bbb"}) == "bbb"
    assert inputs.get("c") == "x"
    assert inputs.get("d") == "w"


def test_input_optional():
    """Inputs can be optional and be given a default"""
    tube = Tube.from_specification("testing-input-optional")
    assert tube.input_collection.get("start") == 10
    tube = Tube.from_specification("testing-input-optional", input={"start": 20})
    assert tube.input_collection.get("start") == 20


def test_enabled_by_input():
    """
    Nodes can have whether they are enabled or not controlled by an input.
    this should be propagated across the tube and scheduler state.
    """
    enabled = Tube.from_specification("testing-input-dynamic-enable", input={"enable_node": True})
    disabled = Tube.from_specification("testing-input-dynamic-enable", input={"enable_node": False})

    assert enabled.nodes["count"].enabled
    assert not disabled.nodes["count"].enabled

    assert "count" in enabled.enabled_nodes
    assert "count" not in disabled.enabled_nodes

    enabled_state = enabled.scheduler.get_epoch_state(enabled.scheduler.epoch)
    disabled_state = disabled.scheduler.get_epoch_state(disabled.scheduler.epoch)

    assert "count" in enabled_state["ready"]
    assert "count" not in disabled_state["ready"]
