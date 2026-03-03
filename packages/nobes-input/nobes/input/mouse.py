from typing import Annotated as A
from typing import Literal as L

from pynput.mouse import Controller
from screeninfo import get_monitors

from noob import Name
from noob.logging import init_logger


def mouse_position(
    scale: L["pixels", "relative"] = "pixels",
) -> tuple[A[int | float, Name("x")], A[int | float, Name("y")]]:
    logger = init_logger("mouse_position")
    mouse = Controller()
    max_x = 0
    max_y = 0

    if scale == "relative":
        monitors = get_monitors()
        for monitor in monitors:
            max_x = max(monitor.x + monitor.width, max_x)
            max_y = max(monitor.y + monitor.height, max_y)

    while True:
        position = mouse.position
        if scale == "relative":
            position = (position[0] / max_x, position[1] / max_y)

        logger.debug("position: %s", position)
        yield position[0], position[1]
