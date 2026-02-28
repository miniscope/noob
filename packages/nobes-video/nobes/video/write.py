import logging
import shutil
import subprocess as sp
from pathlib import Path
from typing import Annotated as A
from typing import Any, Self, TypeAlias, TypedDict, cast

import numpy as np
from numpydantic import NDArraySchema
from numpydantic import dtype as npdt
from pydantic import ConfigDict, Field, PrivateAttr, field_validator, model_validator

from noob import Node
from noob.logging import init_logger

DEFAULT_INPUT_PARAMS = {
    "-f": "rawvideo",
}

DEFAULT_OUTPUT_PARAMS = {"-pix_fmt": "yuv420p", "-c:v": "libx264", "-preset": "ultrafast"}

FrameDtypes = (*npdt.Integer, *npdt.Float)
GrayscaleFrame: TypeAlias = A[np.ndarray, NDArraySchema(("*", "*"), FrameDtypes)]
MultiChannelFrame: TypeAlias = A[np.ndarray, NDArraySchema(("*", "*", "1-4"), FrameDtypes)]
MultiFrames: TypeAlias = A[np.ndarray, NDArraySchema(("*", "*", "*", "1-4"), FrameDtypes)]
VideoFrame: TypeAlias = GrayscaleFrame | MultiChannelFrame | MultiFrames


class _FrameParams(TypedDict):
    input_channels: int
    bits_per_channel: int
    dtype: np.dtype
    shape: tuple[int, int, int, int]


class FFMpegWriter(Node):
    """
    Write videos to a file using ffmpeg

    Adapted from [scikit-video](https://github.com/scikit-video/scikit-video),
    which has a competent implementation but hasn't released a new version since 2018.

    FFMpeg must be installed!

    Frames given to this writer must all be the same shape and dtype.
    output video format is inferred from the first frame that's given
    if not specified in the output params.

    Frames are expected to be ordered and shaped like

    - single frames:  height x width x channel OR height x width (for a grayscale image)
    - multiple frames: frames x height x width x channel

    It is NOT supported to pass multiple grayscale images without an explicit fourth 1-length axis,
    as that would be ambiguous with a very squat video with few rows.

    Input data is assumed to be in RGB order - if you are using a non-RGB input pixel format,
    you are responsible for transposing the axes in the array before passing a frame to the writer.

    .. todo::

        Async writing! Use the asyncio.subprocess spawner.
        It's a very simple change here, but we just need to change how sync runners handle async
        to reuse a single eventloop.

    """

    fs: int
    """Frames per second of the output video"""
    output_path: Path
    """The destination for the video"""
    overwrite: bool = False
    """If an output file is already present, overwrite it. Otherwise, throw FileExistsError"""
    input_params: dict[str, str] = Field(default_factory=dict)
    """
    Input parameters to ffmpeg,
    see the ffmpeg documentation for more details.
    Keys should be passed with leading `-` as they would be called on the command line,
    e.g. setting `pix_fmt` should use the key `-pix_fmt`.
    """
    output_params: dict[str, str] = Field(default_factory=dict)
    """
    Input parameters to ffmpeg,
    see the ffmpeg documentation for more details.
    Keys should be passed with leading `-` as they would be called on the command line,
    e.g. setting `pix_fmt` should use the key `-pix_fmt`.
    
    passing `-r` here to set framerate is overridden by the value passed in `fs`
    """
    ffmpeg_path: Path
    """
    Pass an explicit path to the ffmpeg executable.
    Otherwise, inferred from :func:`shutil.which`
    """

    _logger: logging.Logger = PrivateAttr(default_factory=lambda: init_logger("nobes.FFMpegWriter"))
    _frame_params: _FrameParams | None = None
    _proc: sp.Popen | None = None

    model_config = ConfigDict(validate_default=True)

    @field_validator("input_params", mode="after")
    @classmethod
    def fill_default_inputs(cls, value: dict[str, str]) -> dict[str, str]:
        return {**DEFAULT_INPUT_PARAMS, **value}

    @field_validator("output_params", mode="after")
    @classmethod
    def fill_default_outputs(cls, value: dict[str, str]) -> dict[str, str]:
        return {**DEFAULT_OUTPUT_PARAMS, **value}

    @model_validator(mode="before")
    @classmethod
    def find_ffmpeg(cls, value: dict | Any) -> dict | Any:
        if isinstance(value, dict) and "ffmpeg_path" not in value:
            path = shutil.which("ffmpeg")
            assert path is not None, "No ffmpeg path provided, and ffmpeg not found in PATH"
            value["ffmpeg_path"] = Path(path)
        return value

    @field_validator("ffmpeg_path", mode="after")
    @classmethod
    def ffmpeg_exists(cls, value: Path) -> Path:
        assert value.exists(), f"ffmpeg path passed as {value}, but {value} does not exist"
        return value

    @model_validator(mode="after")
    def dont_overwrite_unless_requested(self) -> Self:
        if self.output_path.exists() and not self.overwrite:
            raise FileExistsError(
                f"{self.output_path} exists, pass overwrite=True to force overwrite"
            )
        return self

    def process(self, frame: VideoFrame) -> None:
        frame = cast(np.ndarray, frame)
        frame = _reshape_frame(frame)
        if self._frame_params is None:
            self._init_writer(frame)
        self._frame_shape_unchanged(frame)
        frame = _cast_to_dtype(frame, self._frame_params["dtype"])
        try:
            self._proc.stdin.write(frame.tobytes())
        except OSError as e:
            raise OSError("Error communicating with ffmpeg") from e

    def deinit(self) -> None:
        if self._proc is None:
            return
        elif self._proc.returncode is not None:
            # already completed or dead
            return

        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.wait()
        self._proc = None

    def _init_writer(self, frame: VideoFrame) -> None:
        self._frame_params = self._init_frame_params(frame)
        self._proc = self._init_process()

    def _init_frame_params(self, frame: VideoFrame) -> _FrameParams:
        if "-pix_fmt" not in self.input_params:
            self.input_params["-pix_fmt"] = _infer_input_pix_fmt(frame.dtype, frame.shape[3])

        n_channels, bits_per_channel = _bpplut[self.input_params["-pix_fmt"]]

        input_dtype = _infer_input_dtype(
            self.input_params["-pix_fmt"], bits_per_channel // n_channels
        )

        if "-s" in self.input_params:
            width, height = self.input_params["-s"].split("x")
            if int(width) != frame.shape[2] or int(height) != frame.shape[1]:
                raise ValueError(
                    f"Specified input shape with -s as {self.input_params['-s']}, "
                    f"but first frame was {frame.shape[2]}x{frame.shape[1]}"
                )
        else:
            self.input_params["-s"] = str(frame.shape[2]) + "x" + str(frame.shape[1])

        return _FrameParams(
            input_channels=n_channels,
            bits_per_channel=bits_per_channel,
            shape=frame.shape,
            dtype=input_dtype,
        )

    def _init_process(self) -> sp.Popen:
        input_args = []
        output_args = []
        for key, val in self.input_params.items():
            input_args.extend([key, val])
        for key, val in {**self.output_params, "-r": str(self.fs)}.items():
            output_args.extend([key, val])

        input_args.extend(["-i", "-"])
        cmd_parts = [str(self.ffmpeg_path), "-y", *input_args, *output_args, str(self.output_path)]
        self._logger.debug("Initializing FFMPEG with %s", cmd_parts)
        proc = sp.Popen(cmd_parts, stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.STDOUT)
        return proc

    def _frame_shape_unchanged(self, frame: VideoFrame) -> None:
        assert frame.shape[1:] == self._frame_params["shape"][1:], (
            f"Frame shape changed during run from "
            f"{self._frame_params['shape'][1:]} to {frame.shape[1:]}"
        )


def _reshape_frame(frame: np.ndarray) -> MultiFrames:
    """
    ensure frame is (n_frames x height x width x channel)
    """
    import numpy as np

    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    if len(frame.shape) == 2:
        a, b = frame.shape
        return frame.reshape(1, a, b, 1)
    elif len(frame.shape) == 3:
        a, b, c = frame.shape
        return frame.reshape(1, a, b, c)
    elif len(frame.shape) == 4:
        return frame
    else:
        raise ValueError(
            f"Frames must be 2d grayscale, 3d multichannel images, "
            f"or 4d frame * width * height * channel frame collections. "
            f"got shape: {frame.shape}"
        )


def _cast_to_dtype(frame: MultiFrames, dtype: np.dtype) -> MultiFrames:
    """clip invalid values and cast to type"""
    frame = frame.clip(0, (1 << (dtype.itemsize << 3)) - 1).astype(dtype)
    return frame


def _infer_input_pix_fmt(dtype: np.dtype, channels: int) -> str:
    # check the number channels to guess
    if dtype.kind == "u" and dtype.itemsize == 2:
        suffix = "le" if dtype.byteorder else "be"
        if channels == 1:
            return "gray16" + suffix
        elif channels == 2:
            return "ya16" + suffix
        elif channels == 3:
            return "rgb48" + suffix
        elif channels == 4:
            return "rgba64" + suffix
        else:
            raise TypeError(f"Can't infer pix_fmt from {dtype}")
    else:
        if channels == 1:
            return "gray"
        elif channels == 2:
            return "ya8"
        elif channels == 3:
            return "rgb24"
        elif channels == 4:
            return "rgba"
        else:
            raise TypeError(f"Can't infer pix_fmt from {dtype}")


def _infer_input_dtype(pix_fmt: str, bits_per_channel: int) -> np.dtype:
    if bits_per_channel == 8:
        return np.dtype("u1")  # np.uint8
    elif bits_per_channel == 16:
        suffix = pix_fmt[-2:]
        if suffix == "le":
            return np.dtype("<u2")
        elif suffix == "be":
            return np.dtype(">u2")
    else:
        raise ValueError(pix_fmt + " is not a valid pix_fmt for numpy conversion")


_bpplut = {
    "yuv420p": (3, 12),
    "yuyv422": (3, 16),
    "rgb24": (3, 24),
    "bgr24": (3, 24),
    "yuv422p": (3, 16),
    "yuv444p": (3, 24),
    "yuv410p": (3, 9),
    "yuv411p": (3, 12),
    "gray": (1, 8),
    "monow": (1, 1),
    "monob": (1, 1),
    "pal8": (1, 8),
    "yuvj420p": (3, 12),
    "yuvj422p": (3, 16),
    "yuvj444p": (3, 24),
    "xvmcmc": (0, 0),
    "xvmcidct": (0, 0),
    "uyvy422": (3, 16),
    "uyyvyy411": (3, 12),
    "bgr8": (3, 8),
    "bgr4": (3, 4),
    "bgr4_byte": (3, 4),
    "rgb8": (3, 8),
    "rgb4": (3, 4),
    "rgb4_byte": (3, 4),
    "nv12": (3, 12),
    "nv21": (3, 12),
    "argb": (4, 32),
    "rgba": (4, 32),
    "abgr": (4, 32),
    "bgra": (4, 32),
    "gray16be": (1, 16),
    "gray16le": (1, 16),
    "yuv440p": (3, 16),
    "yuvj440p": (3, 16),
    "yuva420p": (4, 20),
    "vdpau_h264": (0, 0),
    "vdpau_mpeg1": (0, 0),
    "vdpau_mpeg2": (0, 0),
    "vdpau_wmv3": (0, 0),
    "vdpau_vc1": (0, 0),
    "rgb48be": (3, 48),
    "rgb48le": (3, 48),
    "rgb565be": (3, 16),
    "rgb565le": (3, 16),
    "rgb555be": (3, 15),
    "rgb555le": (3, 15),
    "bgr565be": (3, 16),
    "bgr565le": (3, 16),
    "bgr555be": (3, 15),
    "bgr555le": (3, 15),
    "vaapi_moco": (0, 0),
    "vaapi_idct": (0, 0),
    "vaapi_vld": (0, 0),
    "yuv420p16le": (3, 24),
    "yuv420p16be": (3, 24),
    "yuv422p16le": (3, 32),
    "yuv422p16be": (3, 32),
    "yuv444p16le": (3, 48),
    "yuv444p16be": (3, 48),
    "vdpau_mpeg4": (0, 0),
    "dxva2_vld": (0, 0),
    "rgb444le": (3, 12),
    "rgb444be": (3, 12),
    "bgr444le": (3, 12),
    "bgr444be": (3, 12),
    "ya8": (2, 16),
    "bgr48be": (3, 48),
    "bgr48le": (3, 48),
    "yuv420p9be": (3, 13),
    "yuv420p9le": (3, 13),
    "yuv420p10be": (3, 15),
    "yuv420p10le": (3, 15),
    "yuv422p10be": (3, 20),
    "yuv422p10le": (3, 20),
    "yuv444p9be": (3, 27),
    "yuv444p9le": (3, 27),
    "yuv444p10be": (3, 30),
    "yuv444p10le": (3, 30),
    "yuv422p9be": (3, 18),
    "yuv422p9le": (3, 18),
    "vda_vld": (0, 0),
    "gbrp": (3, 24),
    "gbrp9be": (3, 27),
    "gbrp9le": (3, 27),
    "gbrp10be": (3, 30),
    "gbrp10le": (3, 30),
    "gbrp16be": (3, 48),
    "gbrp16le": (3, 48),
    "yuva420p9be": (4, 22),
    "yuva420p9le": (4, 22),
    "yuva422p9be": (4, 27),
    "yuva422p9le": (4, 27),
    "yuva444p9be": (4, 36),
    "yuva444p9le": (4, 36),
    "yuva420p10be": (4, 25),
    "yuva420p10le": (4, 25),
    "yuva422p10be": (4, 30),
    "yuva422p10le": (4, 30),
    "yuva444p10be": (4, 40),
    "yuva444p10le": (4, 40),
    "yuva420p16be": (4, 40),
    "yuva420p16le": (4, 40),
    "yuva422p16be": (4, 48),
    "yuva422p16le": (4, 48),
    "yuva444p16be": (4, 64),
    "yuva444p16le": (4, 64),
    "vdpau": (0, 0),
    "xyz12le": (3, 36),
    "xyz12be": (3, 36),
    "nv16": (3, 16),
    "nv20le": (3, 20),
    "nv20be": (3, 20),
    "yvyu422": (3, 16),
    "vda": (0, 0),
    "ya16be": (2, 32),
    "ya16le": (2, 32),
    "qsv": (0, 0),
    "mmal": (0, 0),
    "d3d11va_vld": (0, 0),
    "rgba64be": (4, 64),
    "rgba64le": (4, 64),
    "bgra64be": (4, 64),
    "bgra64le": (4, 64),
    "0rgb": (3, 24),
    "rgb0": (3, 24),
    "0bgr": (3, 24),
    "bgr0": (3, 24),
    "yuva444p": (4, 32),
    "yuva422p": (4, 24),
    "yuv420p12be": (3, 18),
    "yuv420p12le": (3, 18),
    "yuv420p14be": (3, 21),
    "yuv420p14le": (3, 21),
    "yuv422p12be": (3, 24),
    "yuv422p12le": (3, 24),
    "yuv422p14be": (3, 28),
    "yuv422p14le": (3, 28),
    "yuv444p12be": (3, 36),
    "yuv444p12le": (3, 36),
    "yuv444p14be": (3, 42),
    "yuv444p14le": (3, 42),
    "gbrp12be": (3, 36),
    "gbrp12le": (3, 36),
    "gbrp14be": (3, 42),
    "gbrp14le": (3, 42),
    "gbrap": (4, 32),
    "gbrap16be": (4, 64),
    "gbrap16le": (4, 64),
    "yuvj411p": (3, 12),
    "bayer_bggr8": (3, 8),
    "bayer_rggb8": (3, 8),
    "bayer_gbrg8": (3, 8),
    "bayer_grbg8": (3, 8),
    "bayer_bggr16le": (3, 16),
    "bayer_bggr16be": (3, 16),
    "bayer_rggb16le": (3, 16),
    "bayer_rggb16be": (3, 16),
    "bayer_gbrg16le": (3, 16),
    "bayer_gbrg16be": (3, 16),
    "bayer_grbg16le": (3, 16),
    "bayer_grbg16be": (3, 16),
    "yuv440p10le": (3, 20),
    "yuv440p10be": (3, 20),
    "yuv440p12le": (3, 24),
    "yuv440p12be": (3, 24),
    "ayuv64le": (4, 64),
    "ayuv64be": (4, 64),
    "videotoolbox_vld": (0, 0),
}
"""
Copied from https://github.com/scikit-video/scikit-video/blob/master/skvideo/utils/__init__.py
Map from pixel formats to the (number of components, number of bits per pixel)
"""
