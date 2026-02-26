from collections.abc import Generator
from typing import Annotated as A

import cv2
import numpy as np

from noob import Name


def webcam(index: int = 0) -> Generator[A[np.ndarray, Name("frame")], None, None]:
    """Yields frames from a webcam (in RGB color, because BGR is silly)"""
    cap = cv2.VideoCapture(index)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
