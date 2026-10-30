"""RealSense RGBD camera using multiprocessing."""

# pyright: reportAttributeAccessIssue=false

import ctypes
import multiprocessing as mp
import time
from dataclasses import dataclass
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray
from typing import Literal

import numpy as np


_MAX_W = 1920
_MAX_H = 1080


@dataclass
class RealsenseConfig:
    """RealSense camera config; `None` resolution uses the camera default (see `rs-enumerate-devices` for modes).

    >>> RealsenseConfig()
    RealsenseConfig(color_width=None, color_height=None, depth_width=None, depth_height=None, fps=30, serial_number=None, align_to='color')
    """

    color_width: int | None = None
    color_height: int | None = None
    depth_width: int | None = None
    depth_height: int | None = None
    fps: int = 30
    serial_number: str | None = None
    align_to: Literal["color", "depth"] = "color"


class Realsense:
    """RealSense RGBD camera that captures frames into shared arrays.

    >>> # Realsense(RealsenseConfig())  # doctest: +SKIP
    """

    def __init__(self, config: RealsenseConfig):
        self._config = config
        self._color = mp.Array(ctypes.c_uint8, _MAX_H * _MAX_W * 3)
        self._depth = mp.Array(ctypes.c_uint16, _MAX_H * _MAX_W)
        self._intrinsics = mp.Array(ctypes.c_double, 7)
        self._running = mp.Value(ctypes.c_bool, True)
        self._ready = mp.Value(ctypes.c_bool, False)
        self._process: mp.Process | None = None

    def start(self):
        self._process = mp.Process(
            target=Realsense._capture_loop,
            args=(self._config, self._color, self._depth, self._intrinsics, self._running, self._ready),
            daemon=True,
        )
        self._process.start()
        while not self._ready.value:
            if not self._process.is_alive():
                raise RuntimeError("RealSense capture process died — check stream config or camera connection")
            time.sleep(0.01)

    def stop(self):
        self._running.value = False
        if self._process:
            self._process.join(timeout=5.0)

    def get_resolution(self) -> tuple[int, int]:
        """Return active `(height, width)` after alignment."""
        return int(self._intrinsics[4]), int(self._intrinsics[5])

    def get_color(self) -> np.ndarray:
        """Return color image as `(H, W, 3)` uint8 BGR array."""
        h, w = self.get_resolution()
        return np.array(self._color[: h * w * 3], dtype=np.uint8).reshape(h, w, 3)

    def get_depth(self) -> np.ndarray:
        """Return depth image as `(H, W)` uint16 array (raw sensor units)."""
        h, w = self.get_resolution()
        return np.array(self._depth[: h * w], dtype=np.uint16).reshape(h, w)

    def get_intrinsics(self) -> np.ndarray:
        """Return `[fx, fy, ppx, ppy, height, width, depth_scale]` as float64."""
        return np.array(self._intrinsics[:])

    @staticmethod
    def _capture_loop(
        config: RealsenseConfig,
        color: SynchronizedArray,
        depth: SynchronizedArray,
        intrinsics: SynchronizedArray,
        running: Synchronized,
        ready: Synchronized,
    ) -> None:
        import pyrealsense2 as rs  # type: ignore[import-unresolved]

        rs_config = rs.config()
        if config.serial_number is not None:
            rs_config.enable_device(config.serial_number)
        if config.color_width is not None and config.color_height is not None:
            rs_config.enable_stream(
                rs.stream.color, config.color_width, config.color_height, rs.format.bgr8, config.fps
            )
        else:
            rs_config.enable_stream(rs.stream.color, rs.format.bgr8, config.fps)
        if config.depth_width is not None and config.depth_height is not None:
            rs_config.enable_stream(rs.stream.depth, config.depth_width, config.depth_height, rs.format.z16, config.fps)
        else:
            rs_config.enable_stream(rs.stream.depth, rs.format.z16, config.fps)

        pipeline = rs.pipeline()
        profile = pipeline.start(rs_config)

        align = rs.align(rs.stream.color if config.align_to == "color" else rs.stream.depth)

        # extract intrinsics from the align-target stream
        target_stream = rs.stream.color if config.align_to == "color" else rs.stream.depth
        stream_profile = profile.get_stream(target_stream)
        intr = stream_profile.as_video_stream_profile().get_intrinsics()
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        intrinsics[:] = [intr.fx, intr.fy, intr.ppx, intr.ppy, intr.height, intr.width, depth_scale]

        n_color = intr.height * intr.width * 3
        n_depth = intr.height * intr.width
        while running.value:
            frameset = align.process(pipeline.wait_for_frames())
            color[:n_color] = np.asarray(frameset.get_color_frame().get_data()).ravel()
            depth[:n_depth] = np.asarray(frameset.get_depth_frame().get_data()).ravel()
            ready.value = True

        pipeline.stop()
