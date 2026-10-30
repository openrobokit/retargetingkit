"""Print RealSense intrinsics and exit."""
# /// script
# requires-python = ">=3.10"
# dependencies = ["retargetingkit[realsense]"]
#
# [tool.uv.sources]
# retargetingkit = { path = "../../..", editable = true }
# ///

import argparse
import json
from pathlib import Path

from retargetingkit.sources.cameras.realsense import Realsense, RealsenseConfig


def main():
    parser = argparse.ArgumentParser(description="Print RealSense intrinsics")
    parser.add_argument("--serial-number", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save", action="store_true", help="Save intrinsics JSON next to this script")
    args = parser.parse_args()

    cam = Realsense(RealsenseConfig(fps=args.fps, serial_number=args.serial_number))
    cam.start()

    intrinsics = cam.get_intrinsics()
    fx, fy, ppx, ppy, height, width, depth_scale = intrinsics
    print(f"resolution:  {int(width)} x {int(height)}")
    print(f"fx, fy:      {fx:.4f}, {fy:.4f}")
    print(f"ppx, ppy:    {ppx:.4f}, {ppy:.4f}")
    print(f"depth_scale: {depth_scale}")

    if args.save:
        out = Path(__file__).parent / "intrinsics.json"
        out.write_text(json.dumps({"intrinsic": intrinsics.tolist()}, indent=2))
        print(f"saved to {out}")

    cam.stop()


if __name__ == "__main__":
    main()
