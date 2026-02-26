import cv2
import numpy as np

from noob.node import Node


class display(Node):
    """Display (RGB) frames"""

    window: str = "default"

    def process(self, frame: np.ndarray) -> None:
        cv2.imshow(self.window, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def deinit(self) -> None:
        cv2.destroyWindow(self.window)
