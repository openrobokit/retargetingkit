"""ZED stereo camera using multiprocessing."""

# pyright: reportAttributeAccessIssue=false

import ctypes
import multiprocessing as mp
import time
from dataclasses import dataclass
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray

import numpy as np


_MAX_W = 1920
_MAX_H = 1080


@dataclass
class ZedConfig:
    """ZED stereo camera config; `resolution`/`depth_mode` must name an `sl.RESOLUTION`/`sl.DEPTH_MODE` member.

    >>> ZedConfig()
    ZedConfig(resolution='HD1080', fps=30, depth_mode='ULTRA', serial_number=None)
    """

    resolution: str = "HD1080"
    fps: int = 30
    depth_mode: str = "ULTRA"
    serial_number: int | None = None


class Zed:
    """ZED stereo camera that captures frames into shared arrays.

    >>> # Zed(ZedConfig())  # doctest: +SKIP
    """

    def __init__(self, config: ZedConfig):
        self._config = config
        self._color = mp.Array(ctypes.c_uint8, _MAX_H * _MAX_W * 3)
        self._depth = mp.Array(ctypes.c_float, _MAX_H * _MAX_W)
        self._intrinsics = mp.Array(ctypes.c_double, 6)
        self._running = mp.Value(ctypes.c_bool, True)
        self._ready = mp.Value(ctypes.c_bool, False)
        self._process: mp.Process | None = None

    def start(self):
        self._process = mp.Process(
            target=Zed._capture_loop,
            args=(self._config, self._color, self._depth, self._intrinsics, self._running, self._ready),
            daemon=True,
        )
        self._process.start()
        while not self._ready.value:
            if not self._process.is_alive():
                raise RuntimeError("ZED capture process died — check camera connection or SDK installation")
            time.sleep(0.01)

    def stop(self):
        self._running.value = False
        if self._process:
            self._process.join(timeout=5.0)

    def get_resolution(self) -> tuple[int, int]:
        """Return active `(height, width)`."""
        return int(self._intrinsics[4]), int(self._intrinsics[5])

    def get_color(self) -> np.ndarray:
        """Return color image as `(H, W, 3)` uint8 BGR array."""
        h, w = self.get_resolution()
        return np.array(self._color[: h * w * 3], dtype=np.uint8).reshape(h, w, 3)

    def get_depth(self) -> np.ndarray:
        """Return depth image as `(H, W)` float32 array (meters)."""
        h, w = self.get_resolution()
        return np.array(self._depth[: h * w], dtype=np.float32).reshape(h, w)

    def get_intrinsics(self) -> np.ndarray:
        """Return `[fx, fy, cx, cy, height, width]` as float64."""
        return np.array(self._intrinsics[:])

    @staticmethod
    def _capture_loop(
        config: ZedConfig,
        color: SynchronizedArray,
        depth: SynchronizedArray,
        intrinsics: SynchronizedArray,
        running: Synchronized,
        ready: Synchronized,
    ) -> None:
        from pyzed import sl  # type: ignore[import-unresolved]

        res_map = {
            "HD2K": sl.RESOLUTION.HD2K,
            "HD1080": sl.RESOLUTION.HD1080,
            "HD720": sl.RESOLUTION.HD720,
            "VGA": sl.RESOLUTION.VGA,
        }
        depth_map = {
            "NEURAL": sl.DEPTH_MODE.NEURAL,
            "NEURAL_PLUS": sl.DEPTH_MODE.NEURAL_PLUS,
            "ULTRA": sl.DEPTH_MODE.ULTRA,
            "QUALITY": sl.DEPTH_MODE.QUALITY,
            "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
            "NONE": sl.DEPTH_MODE.NONE,
        }

        init_params = sl.InitParameters()
        init_params.camera_resolution = res_map[config.resolution]
        init_params.camera_fps = config.fps
        init_params.depth_mode = depth_map[config.depth_mode]
        init_params.coordinate_units = sl.UNIT.METER
        if config.serial_number is not None:
            init_params.set_from_serial_number(config.serial_number)

        cam = sl.Camera()
        status = cam.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED camera open failed: {status}")

        cam_info = cam.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters.left_cam
        res = cam_info.camera_configuration.resolution
        w, h = res.width, res.height
        intrinsics[:] = [calib.fx, calib.fy, calib.cx, calib.cy, h, w]

        ready.value = True

        color_mat = sl.Mat()
        depth_mat = sl.Mat()
        runtime = sl.RuntimeParameters()
        n_color = h * w * 3
        n_depth = h * w

        while running.value:
            if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(color_mat, sl.VIEW.LEFT)
                color[:n_color] = color_mat.get_data()[:, :, :3].ravel()

                cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                depth[:n_depth] = depth_mat.get_data().ravel()

        cam.close()
