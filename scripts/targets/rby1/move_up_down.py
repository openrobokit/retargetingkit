"""Move RBY1-A to home, then repeatedly raise and lower the end-effector.

Usage:
    uv run --extra rby1 python scripts/targets/rby1/move_up_down.py
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
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--distance", type=float, default=5.0, help="cm")
    parser.add_argument("--max-vel", type=float, default=1.5, help="max velocity rad/s")
    parser.add_argument("--tolerance", type=float, default=0.01, help="convergence tolerance rad")
    args = parser.parse_args()

    config = RBY1AConfig(args.ip, max_velocity=np.full(7, args.max_vel))
    arm = config.build()
    time.sleep(0.5)

    signal.signal(signal.SIGINT, lambda *_: (arm.close(), sys.exit(1)))

    # Move to home first
    arm.set_qpos(config.home_qpos)
    print(f"moving to home: {config.home_qpos}")
    while True:
        err = np.max(np.abs(arm.get_qpos() - config.home_qpos))
        print(f"max joint error: {err:.4f} rad")
        if err < args.tolerance:
            break
        time.sleep(0.5)

    home_pose = arm.get_ee_pose()
    up_pose = home_pose.copy()
    up_pose[2] += args.distance / 100  # cm to m

    print(f"home z: {home_pose[2]:.4f} m")
    print(f"up   z: {up_pose[2]:.4f} m")

    for i in range(args.reps):
        for label, action in [
            ("up", lambda: arm.set_eepose(up_pose)),
            ("down", lambda: arm.set_qpos(config.home_qpos)),
        ]:
            print(f"rep {i + 1}/{args.reps}: {label}")
            action()
            prev = arm.get_qpos()
            while True:
                time.sleep(0.5)
                qpos = arm.get_qpos()
                if np.max(np.abs(qpos - prev)) < args.tolerance:
                    break
                prev = qpos

    arm.close()
    print("done")


if __name__ == "__main__":
    main()
