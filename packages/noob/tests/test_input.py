import pytest

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
