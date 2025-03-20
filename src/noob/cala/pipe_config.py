from typing import TypedDict, Any, Sequence, NotRequired


class PreprocessStep(TypedDict):
    transformer: type
    params: dict[str, Any]
    requires: NotRequired[Sequence[str]]


class InitializationStep(TypedDict):
    transformer: type  # The transformer class
    params: dict[str, Any]  # Parameters for the transformer
    n_frames: int  # Number of frames to use
    requires: NotRequired[Sequence[str]]  # Optional dependencies


class IterationStep(TypedDict):
    transformer: type  # The transformer class
    params: dict[str, Any]  # Parameters for the transformer
    requires: NotRequired[Sequence[str]]  # Optional dependencies


class StreamingConfig(TypedDict):
    preprocess: dict[str, PreprocessStep]
    initialization: dict[str, InitializationStep]
    iteration: dict[str, IterationStep]


# Example config

# config = {
#             "preprocess": {
#                 "downsample": {
#                     "transformer": Downsampler,
#                     "params": {
#                         "method": "mean",
#                         "dimensions": ["width", "height"],
#                         "strides": [2, 2],
#                     },
#                 },
#                 "denoise": {
#                     "transformer": Denoiser,
#                     "params": {
#                         "method": "gaussian",
#                         "kwargs": {"ksize": (3, 3), "sigmaX": 1.5},
#                     },
#                     "requires": ["downsample"],
#                 },
#                 "glow_removal": {
#                     "transformer": GlowRemover,
#                     "params": {},
#                     "requires": ["denoise"],
#                 },
#                 "background_removal": {
#                     "transformer": BackgroundEraser,
#                     "params": {"method": "uniform", "kernel_size": 3},
#                     "requires": ["glow_removal"],
#                 },
#                 "motion_stabilization": {
#                     "transformer": RigidTranslator,
#                     "params": {"drift_speed": 1, "anchor_frame_index": 0},
#                     "requires": ["background_removal"],
#                 },
#             },
#             "initialization": {
#                 "footprints": {
#                     "transformer": FootprintsInitializer,
#                     "params": {
#                         "threshold_factor": 0.2,
#                         "kernel_size": 3,
#                         "distance_metric": cv2.DIST_L2,
#                         "distance_mask_size": 5,
#                     },
#                     "output": [
#                     {"source": "fp", "target": "traces.fp"},
#                     {"source": "fp_stats", "target": "logger.value"}
#                    ]
#                 },
#                 "traces": {
#                     "transformer": TracesInitializer,
#                     "params": {"component_axis": "components", "frames_axis": "frame"},
#                     "n_frames": 3,
#                     "requires": ["footprints"],
#                 },
#             },
#             "extraction": { ...
#             }
#         }
