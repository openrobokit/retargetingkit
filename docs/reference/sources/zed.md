# ZED

Capture BGR color and metric depth with a Stereolabs ZED camera. `Zed` runs capture in a background process and exposes frames and intrinsics. It is a standalone camera driver and requires a CUDA-capable NVIDIA GPU.

## Install the SDK

Install the [ZED SDK](https://www.stereolabs.com/developers/release/) for the workstation's Ubuntu and CUDA versions. Stereolabs publishes one Python binding wheel per Python version (3.10–3.14) for **ZED SDK 5.2, Linux x86-64**; select the one matching the project's interpreter:

```shell
PY=$(uv run python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
PYZED="https://download.stereolabs.com/zedsdk/5.2/whl/linux_x86_64/pyzed-5.2-$PY-$PY-linux_x86_64.whl"
```

## Check the camera

```shell
uv run --with "$PYZED" --with opencv-python \
    python scripts/sources/zed/stream_rgbd.py
```

The viewer shows color beside a depth visualization; press Escape to close it. It defaults to `HD1080`, 30 FPS, and `ULTRA` depth mode. Set `--serial-number <SERIAL>` to select a camera, or use `--resolution`, `--fps`, and `--depth-mode` to change capture settings.

## Read frames in Python

```python
from retargetingkit.sources.cameras.zed import Zed, ZedConfig

camera = Zed(ZedConfig(resolution="HD1080", depth_mode="ULTRA"))
camera.start()
color = camera.get_color()          # (H, W, 3), uint8 BGR
depth = camera.get_depth()          # (H, W), float32 meters
intrinsics = camera.get_intrinsics()
camera.stop()
```

`intrinsics` is `[fx, fy, cx, cy, height, width]` for the left camera. Depth is already in meters and needs no scale conversion.
