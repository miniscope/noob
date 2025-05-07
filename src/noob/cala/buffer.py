from collections import deque

import xarray as xr


class Buffer:
    def __init__(self, buffer_size: int) -> None:
        """
        Initialize the ring buffer with:
          - buffer_size: number of frames to store
          - dtype: data type of the frames (e.g. np.uint8 for typical video data)
        """
        self.buffer_size = buffer_size
        self.frame_axis = "frame"

        self.buffer: deque[xr.DataArray] = deque()

    def add_frame(self, frame: xr.DataArray) -> None:
        """
        Add a new frame to the ring buffer.
        """

        self.buffer.append(frame)
        if len(self.buffer) > self.buffer_size:
            self.buffer.popleft()

    def get_latest(self, n: int = 1) -> xr.DataArray:
        """Get n most recent frames.

        Returns:
            xr.DataArray: A 3D array containing the stacked frames.
        """
        if not self.is_ready(n):
            raise ValueError("Buffer does not have enough frames.")

        if n == 1:
            return self.buffer[-1]

        return xr.concat(list(self.buffer)[-n:], dim="frame")

    def get_earliest(self, n: int = 1) -> xr.DataArray:
        """Get n earliest frames.

        Returns:
            xr.DataArray: A 3D array containing the stacked frames.
        """
        if not self.is_ready(n):
            raise ValueError("Buffer does not have enough frames.")

        if n == 1:
            return self.buffer[0]

        return xr.concat(list(self.buffer)[:n], dim="frame")

    def is_ready(self, num_frames: int) -> bool:
        """Check if buffer has enough frames."""
        return len(self.buffer) >= num_frames

    def cleanup(self) -> None:
        self.buffer.clear()
        return None
