"""Move RBY1-A head to a target (yaw, pitch) in degrees.

Usage:
    uv run --extra rby1 python scripts/targets/rby1/move_head.py --yaw-deg 20 --pitch-deg 0
"""

import argparse
import signal
import sys
import time

import numpy as np

from retargetingkit.targets.humanoids.rby1 import RBY1AConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="10.2.1.200:50051")
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="target yaw deg (head_0)")
    parser.add_argument("--pitch-deg", type=float, default=0.0, help="target pitch deg (head_1)")
    parser.add_argument("--max-vel", type=float, default=1.0, help="max head velocity rad/s")
    parser.add_argument("--tolerance", type=float, default=0.02, help="convergence tolerance rad")
    args = parser.parse_args()

    config = RBY1AConfig(args.ip, head_max_velocity=np.full(2, args.max_vel))
    arm = config.build()
    time.sleep(0.5)

    signal.signal(signal.SIGINT, lambda *_: (arm.close(), sys.exit(1)))

    print(f"head home: {arm.head_home_qpos}")

    target = np.deg2rad(np.array([args.yaw_deg, args.pitch_deg]))
    print(f"target: {target}")
    arm.set_head_qpos(target)
    while True:
        time.sleep(0.2)
        err = np.max(np.abs(arm.get_head_qpos() - target))
        if err < args.tolerance:
            break

    print(f"reached: {arm.get_head_qpos()}")
    arm.close()
    print("done")


if __name__ == "__main__":
    main()
