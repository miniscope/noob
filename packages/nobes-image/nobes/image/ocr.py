from pathlib import Path
from typing import Annotated as A

import numpy as np
import pytesseract
from PIL import Image

from noob import Name


def tesseract(
    image: Image.Image | np.ndarray | Path, language: str = "eng"
) -> A[str, Name("text")]:
    """
    OCR an image using Tesseract OCR.
    Must have tesseract installed and available on $PATH!
    """
    if isinstance(image, str | Path):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    return pytesseract.image_to_string(image, lang=language)
