from pathlib import Path
from typing import Annotated as A

import numpy as np
from PIL import Image

from noob import Name


def read_image(path: Path) -> tuple[A[Image.Image, Name("image")], A[np.ndarray, Name("array")]]:
    """
    Read an image, returning a PIL Image object and the numpy array
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist!")

    image = Image.open(path)
    return image, np.asarray(image)
