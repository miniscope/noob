from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel


class Step(BaseModel):
    transformer: str  # The transformer class
    params: dict[str, Any]  # Parameters for the transformer
    n_frames: int = 1  # Number of frames to use
    requires: Sequence[str] = []  # Optional dependencies


class StreamingConfig(BaseModel):
    general: dict[str, Any]
    preprocess: dict[str, Step]
    initialization: dict[str, Step]
    iteration: dict[str, Step]
