"""Read and print RBY1-A right-arm joint positions.

Usage:
    uv run --extra rby1 python scripts/targets/rby1/read_qpos.py
"""

import argparse
import time

from retargetingkit.targets.humanoids.rby1 import RBY1AConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="10.2.1.200:50051")
    parser.add_argument("--hz", type=float, default=10.0)
    args = parser.parse_args()

    arm = RBY1AConfig(args.ip).build()

    dt = 1.0 / args.hz
    while True:
        print(arm.get_qpos())
        time.sleep(dt)


if __name__ == "__main__":
    main()
