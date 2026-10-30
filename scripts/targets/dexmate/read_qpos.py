"""Read and print Dexmate whole-body joint positions (torso, head, arms).

Usage:
    uv run --extra dexmate scripts/targets/dexmate/read_qpos.py
"""

import argparse
import atexit
import time

from retargetingkit.targets.humanoids.dexmate import DexmateConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hz", type=float, default=10.0)
    args = parser.parse_args()

    robot = DexmateConfig().build()
    atexit.register(robot.close)
    dt = 1.0 / args.hz
    while True:
        print(robot.get_qpos())
        time.sleep(dt)


if __name__ == "__main__":
    main()
