from collections.abc import Sequence
from typing import TypeAlias

import numpy as np

_numeric: TypeAlias = float | int | np.ndarray
_range: TypeAlias = tuple[float | int, float | int]


def _scale(value: _numeric, input_range: _range, output_range: _range) -> _numeric:
    in_norm = (value - input_range[0]) / (input_range[1] - input_range[0])
    return (in_norm * (output_range[1] - output_range[0])) + output_range[0]


def rescale(
    value: _numeric | Sequence[_numeric], input_range: _range, output_range: _range
) -> _numeric | Sequence[_numeric]:
    if isinstance(value, np.ndarray | float | int):
        return _scale(value, input_range, output_range)
    elif isinstance(value, Sequence):
        return [_scale(value, input_range, output_range) for value in value]
    else:
        raise TypeError("Need a numeric type or sequence of numeric types")
