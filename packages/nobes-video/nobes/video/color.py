from typing import Annotated as A

import cv2
import numpy as np
from annotated_types import Ge, Le

from noob import Name
from noob.logging import init_logger

log = init_logger("rotate_hue")


def rotate_hue(
    frame: np.ndarray, rotate: A[float, Ge(-np.pi * 2), Le(float(np.pi * 2))]
) -> A[np.ndarray, Name("frame")]:
    """Rotate the hue of an RGB image (in radians)"""
    h, s, v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_RGB2HSV))

    rotate = np.clip(rotate, -np.pi * 2, np.pi * 2)
    if rotate < 0:
        rotate = (np.pi * 2) + rotate

    # opencv hsv range is 0 - 179 to fit in 8-bit range (360/2)
    rotate = np.round(np.rad2deg(rotate) / 2)
    h = ((h.astype(np.uint16) + rotate) % 180).astype(np.uint8)
    hsv = cv2.merge((h, s, v))

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb
