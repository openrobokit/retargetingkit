"""Stream ZED RGBD side-by-side via OpenCV."""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "retargetingkit",
#     "opencv-python",
# ]
#
# [tool.uv.sources]
# retargetingkit = { path = "../../../..", editable = true }
# ///

import argparse
import time

import cv2
import numpy as np

from retargetingkit.sources.cameras.zed import Zed, ZedConfig


def main():
    parser = argparse.ArgumentParser(description="Stream ZED RGBD")
    parser.add_argument("--serial-number", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", type=str, default="HD1080")
    parser.add_argument("--depth-mode", type=str, default="ULTRA")
    args = parser.parse_args()

    config = ZedConfig(
        resolution=args.resolution,
        fps=args.fps,
        depth_mode=args.depth_mode,
        serial_number=args.serial_number,
    )
    cam = Zed(config)
    cam.start()

    h, w = cam.get_resolution()
    cv2.namedWindow("RGBD", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RGBD", w * 2, h)

    while True:
        color = cam.get_color()
        depth = cam.get_depth()
        depth_vis = np.clip(depth, 0, 10.0)
        depth_vis = (depth_vis / 10.0 * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        combined = np.hstack((color, depth_color))
        cv2.imshow("RGBD", combined)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
        time.sleep(0.01)

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
