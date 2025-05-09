import json
from collections.abc import Generator
from pathlib import Path
from typing import Annotated as A

import numpy as np

from noob.types import Name


def get_meta(file: Path) -> tuple[A[int, Name("width")], A[int, Name("height")]]:
    with open(file) as f:
        data = json.load(f)
    return data["meta"]["width"], data["meta"]["height"]


def data_iterator(
    file: Path, width: int, height: int
) -> Generator[A[list[list[int]], Name("frame")], None, None]:
    with open(file) as f:
        data = json.load(f)

    px_per_frame = width * height

    for i in range(0, len(data["data"]), px_per_frame):
        frame = []
        for row in range(height):
            frame.append(data["data"][i + (row * width) : i + (row * width) + width])
        yield frame


def frame_mean(frame: list[list[int]]) -> A[float, Name("value")]:
    return np.mean(frame)


def write(value: float, output_path: Path) -> None:
    with open(output_path, "a") as o:
        o.write(str(value))
