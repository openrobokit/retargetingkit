"""Move RBY1-A to home joint configuration very slowly.

Usage:
    uv run --extra rby1 python scripts/targets/rby1/move_to_home.py
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
    parser.add_argument("--max-vel", type=float, default=0.5, help="max velocity rad/s")
    parser.add_argument("--tolerance", type=float, default=0.01, help="convergence tolerance rad")
    parser.add_argument("--zero", action="store_true", help="move to zero qpos instead of home")
    args = parser.parse_args()

    config = RBY1AConfig(args.ip, max_velocity=np.full(7, args.max_vel))
    arm = config.build()
    time.sleep(0.5)

    signal.signal(signal.SIGINT, lambda *_: (arm.close(), sys.exit(1)))

    target = np.zeros(7) if args.zero else config.home_qpos
    arm.set_qpos(target)
    print(f"moving to:      {target}")
    print(f"from:           {arm.get_qpos()}")

    while True:
        qpos = arm.get_qpos()
        err = np.max(np.abs(qpos - target))
        print(f"max joint error: {err:.4f} rad")
        if err < args.tolerance:
            break
        time.sleep(0.5)

    arm.close()
    print("done")


if __name__ == "__main__":
    main()
