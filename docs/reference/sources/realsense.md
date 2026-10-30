# RealSense

Capture aligned RGB-D frames with `Realsense`. The driver runs in a background process and exposes BGR color, raw depth, and camera intrinsics. L515 is the tested model; this camera driver is separate from the canonical teleoperation sources.

## Check the camera

From the RetargetingKit repository:

```shell
uv sync --extra realsense
uv run --extra realsense python scripts/sources/realsense/print_intrinsics.py
uv run --extra realsense --with opencv-python \
    python scripts/sources/realsense/stream_rgbd.py
```

The intrinsics script prints image resolution, focal lengths, principal point, and depth scale. The viewer shows color beside a depth visualization; press Escape to close it.

The driver uses the first connected camera by default. Run `rs-enumerate-devices` to find a serial number, then pass `--serial-number <SERIAL>` to either script. The `realsense` extra pins `pyrealsense2<2.55` for L515 depth-stream compatibility.

## Read frames in Python

```python
from retargetingkit.sources.cameras.realsense import Realsense, RealsenseConfig

camera = Realsense(RealsenseConfig())
camera.start()
color = camera.get_color()          # (H, W, 3), uint8 BGR
depth = camera.get_depth()          # (H, W), uint16 sensor units
intrinsics = camera.get_intrinsics()
depth_meters = depth * intrinsics[6]
camera.stop()
```

`intrinsics` is `[fx, fy, ppx, ppy, height, width, depth_scale]`. Frames align to the color stream by default; set `RealsenseConfig(align_to="depth")` to align to depth. `start()` waits for the first captured frames.
