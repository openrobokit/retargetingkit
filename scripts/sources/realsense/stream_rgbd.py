"""Stream RealSense RGBD side-by-side via OpenCV."""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "retargetingkit[realsense]",
#     "opencv-python",
# ]
#
# [tool.uv.sources]
# retargetingkit = { path = "../../..", editable = true }
# ///

import argparse
import time

import cv2
import numpy as np

from retargetingkit.sources.cameras.realsense import Realsense, RealsenseConfig


def main():
    parser = argparse.ArgumentParser(description="Stream RealSense RGBD")
    parser.add_argument("--serial-number", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    config = RealsenseConfig(fps=args.fps, serial_number=args.serial_number)
    cam = Realsense(config)
    cam.start()

    h, w = cam.get_resolution()
    cv2.namedWindow("RGBD", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RGBD", w * 2, h)

    while True:
        color = cam.get_color()
        depth = cam.get_depth()
        depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
        combined = np.hstack((color, depth_color))
        cv2.imshow("RGBD", combined)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
        time.sleep(0.01)

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
