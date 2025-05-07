from enum import Enum

from .init.common import FootprintsInitializer, TracesInitializer
from .init.odl import (
    ComponentStatsInitializer,
    OverlapsInitializer,
    PixelStatsInitializer,
    ResidualInitializer,
)
from .iter import (
    ComponentStatsUpdater,
    Detector,
    FootprintsUpdater,
    OverlapsUpdater,
    PixelStatsUpdater,
    TracesUpdater,
)
from .preprocess import BackgroundEraser, Denoiser, Downsampler, GlowRemover, RigidStabilizer


class Node(Enum):
    Downsampler = Downsampler
    Denoiser = Denoiser
    GlowRemover = GlowRemover
    RigidStabilizer = RigidStabilizer
    BackgroundEraser = BackgroundEraser

    FootprintsInitializer = FootprintsInitializer
    TracesInitializer = TracesInitializer
    ComponentStatsInitializer = ComponentStatsInitializer
    OverlapsInitializer = OverlapsInitializer
    PixelStatsInitializer = PixelStatsInitializer
    ResidualInitializer = ResidualInitializer

    ComponentStatsUpdater = ComponentStatsUpdater
    Detector = Detector
    FootprintsUpdater = FootprintsUpdater
    OverlapsUpdater = OverlapsUpdater
    PixelStatsUpdater = PixelStatsUpdater
    TracesUpdater = TracesUpdater
