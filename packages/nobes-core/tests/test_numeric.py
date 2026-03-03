import pytest

from nobes.core.numeric import rescale


@pytest.mark.parametrize("inscale,outscale,value,expected", (((0, 1), (-1, 1), 0.5, 0),))
def test_rescale(
    inscale: tuple[float, float], outscale: tuple[float, float], value: float, expected: float
):
    """
    Rescale some value from one range to another.

    Just the simplest case for now, keep adding cases if there are more problems!
    """
    rescaled = rescale(value, input_range=inscale, output_range=outscale)
    assert rescaled == expected
