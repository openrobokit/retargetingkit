"""Read and print xArm6 joint positions."""

import argparse
import time

from retargetingkit.targets.arms.xarm6 import XArm6, XArm6Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="192.168.1.228")
    parser.add_argument("--hz", type=float, default=10.0)
    args = parser.parse_args()

    arm = XArm6(XArm6Config(args.ip))
    arm.start()

    dt = 1.0 / args.hz
    while True:
        print(arm.get_qpos())
        time.sleep(dt)


if __name__ == "__main__":
    main()
