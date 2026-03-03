import cv2
import numpy as np
import pytest

from nobes.video.write import FFMpegWriter


@pytest.mark.parametrize(
    "shape,dtype,input_pix_fmt",
    (
        ((640, 480), np.uint8, "gray"),
        ((640, 480, 1), np.uint8, "gray"),
        ((640, 480, 3), np.uint8, "rgb24"),
        ((640, 480), np.uint16, "gray16le"),
        ((640, 480, 1), np.uint16, "gray16le"),
        ((640, 480, 3), np.uint16, "rgb48le"),
    ),
)
def test_ffmpegwriter_input_pix_fmt(shape, dtype, input_pix_fmt, tmp_path):
    """
    Infer input pix_fmt depending on the shape and dtype of the frame on first process call
    """
    frame = np.ones(shape, dtype)

    out_video = tmp_path / "video.mp4"
    node = FFMpegWriter(id="writer", fs=30, output_path=out_video, overwrite=True)
    node.init()
    try:
        node.process(frame=frame)
        assert node.input_params["-pix_fmt"] == input_pix_fmt
    finally:
        node.deinit()


@pytest.mark.parametrize(
    "shape,dtype",
    (
        ((640, 480), np.uint8),
        ((640, 480, 1), np.uint8),
        ((640, 480, 3), np.uint8),
        ((640, 480), np.uint16),
        ((640, 480, 1), np.uint16),
        ((640, 480, 3), np.uint16),
    ),
)
def test_ffmpegwriter_writes(shape, dtype, tmp_path):
    """
    We can write frames correctly with the ffmpeg writer.

    Test is weak to the content/quality of the write - we're just testing we can do it here.
    """
    n_frames = 10
    out_video = tmp_path / "video.mp4"
    frames = [np.ones(shape, dtype) for _ in range(n_frames)]
    node = FFMpegWriter(id="writer", fs=30, output_path=out_video, overwrite=True)
    node.init()
    try:
        for frame in frames:
            node.process(frame=frame)
    finally:
        node.deinit()

    reader = cv2.VideoCapture(str(out_video))
    read_frames = []
    for i in range(n_frames):
        ret, frame = reader.read()
        assert ret, f"Frame {i} could not be read!"
        read_frames.append(frame)

    for out_frame, in_frame in zip(frames, read_frames):
        if len(out_frame.shape) == 2:
            out_frame = np.stack([out_frame, out_frame, out_frame], axis=2)
        # since these are integers and we're writing videos that are pretty much just black frames,
        # absolute toleratnce can be 1 with no real problem here.
        assert np.allclose(out_frame, in_frame, atol=1)
