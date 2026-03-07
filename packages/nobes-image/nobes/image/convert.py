from pathlib import Path
from typing import Annotated as A
from typing import Any

import numpy as np
from PIL import Image

from noob import Name


def convert(
    source: Image.Image | np.ndarray | Path, destination: Path, **kwargs: Any
) -> A[Path, Name("destination")]:
    if isinstance(source, str):
        source = Path(source)

    if isinstance(source, Path):
        im = Image.open(source)
    elif isinstance(source, Image.Image):
        im = source
    elif isinstance(source, np.ndarray):
        im = Image.fromarray(source)
    else:
        raise TypeError(f"Unknown source type: {source}")

    destination = Path(destination)

    if destination.suffix in (".jpg", ".jpeg") and im.mode == "RGBA":
        im = im.convert("RGB")

    im.save(destination, **kwargs)
    return destination
