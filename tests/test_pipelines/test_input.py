import pytest

from noob import SynchronousRunner, Tube
from noob.exceptions import ExtraInputWarning, InputMissingError
from noob.runner.base import TubeRunner

pytestmark = pytest.mark.input


def test_tube_input_params(sync_runner_cls: type[TubeRunner]):
    """tube-scoped input can be used as params"""
    tube = Tube.from_specification("testing-input-tube-params", input={"start": 7})
    runner = sync_runner_cls(tube)
    with runner:
        outputs = [runner.process() for _ in range(5)]
        assert len(outputs) == 5
        assert outputs == [7, 8, 9, 10, 11]


def test_tube_input_depends(sync_runner_cls: type[TubeRunner]):
    """tube-scoped input can be used as depends"""
    tube = Tube.from_specification("testing-input-tube-depends", input={"multiply_right": 7})
    runner = sync_runner_cls(tube)
    with runner:
        outputs = [runner.process() for _ in range(5)]
        assert len(outputs) == 5
        assert outputs == [i * 7 for i in range(5)]


def test_process_input_params():
    """process-scoped input can NOT be used as params"""
    with (
        pytest.raises(InputMissingError),
        pytest.warns(ExtraInputWarning, match=r"Ignoring extra.*"),
    ):
        Tube.from_specification("testing-input-process-params", input={"start": 7})


@pytest.mark.parametrize("loaded_tube", ["testing-input-process-depends"], indirect=True)
def test_process_input_depends(loaded_tube, runner):
    """process-scoped input can be used as depends"""
    for left, right in zip(range(5), range(5, 10)):
        assert runner.process(multiply_right=right) == left * right


@pytest.mark.parametrize("tube", ("testing-input-tube-params", "testing-input-tube-depends"))
def test_tube_input_missing(tube):
    """Input scoped as tube input raises an error when missing"""
    with pytest.raises(InputMissingError):
        Tube.from_specification(tube)

    with (
        pytest.raises(InputMissingError),
        pytest.warns(ExtraInputWarning, match=r"Ignoring extra.*"),
    ):
        Tube.from_specification(tube, input={"irrelevant": 7})


@pytest.mark.parametrize("loaded_tube", ("testing-input-process-depends",), indirect=True)
def test_process_input_missing(loaded_tube, runner):
    """Input scoped as process input raises an error when missing"""
    with pytest.raises(InputMissingError):
        runner.process()

    with (
        pytest.raises(InputMissingError),
        pytest.warns(ExtraInputWarning, match=r"Ignoring extra.*"),
    ):
        runner.process(irrelevant=1)

    # nodes were not run
    assert runner.process(multiply_right=10) == 0


def test_input_integration(sync_runner_cls):
    """
    All the different forms of input can be used together
    """
    start = 7
    multiply_tube = 13
    multiply_process = 17
    tube = Tube.from_specification(
        "testing-input-mixed", input={"start": start, "multiply_tube": multiply_tube}
    )

    runner = sync_runner_cls(tube)
    with runner:
        for i in range(5):
            this_process = multiply_process * 2 * (i + 1)
            expected = (start + i) * multiply_tube * this_process
            assert runner.process(multiply_process=this_process) == expected
