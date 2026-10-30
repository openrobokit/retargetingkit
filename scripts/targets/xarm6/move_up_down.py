"""
Move xArm6 end-effector up and back using the XArm6 class.
Usage:
      uv run --extra xarm scripts/targets/xarm6/move_up_down.py
"""

import argparse
import signal
import sys
import time

from retargetingkit.targets.arms.xarm6 import XArm6, XArm6Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="192.168.1.228")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--distance", type=float, default=5.0, help="cm")
    args = parser.parse_args()

    arm = XArm6(XArm6Config(args.ip))
    arm.start()
    time.sleep(0.5)

    signal.signal(signal.SIGINT, lambda *_: (arm.close(), sys.exit(1)))

    home_pose = arm.get_ee_pose()
    up_pose = home_pose.copy()
    up_pose[2] += args.distance / 100  # cm to m

    print(f"home z: {home_pose[2]:.4f} m")
    print(f"up   z: {up_pose[2]:.4f} m")

    for i in range(args.reps):
        for label, action in [("up", lambda: arm.set_eepose(up_pose)), ("down", lambda: arm.set_qpos(arm.home_qpos))]:
            print(f"rep {i + 1}/{args.reps}: {label}")
            action()
            time.sleep(1.0)

    arm.close()
    print("done")


if __name__ == "__main__":
    main()
