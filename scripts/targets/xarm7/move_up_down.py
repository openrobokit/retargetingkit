"""Move xArm7 end-effector up and back using the XArm7 class."""

import argparse
import time
from contextlib import ExitStack

import numpy as np

from retargetingkit.targets.arms.xarm7 import NUM_JOINTS, XArm7, XArm7Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="192.168.1.242")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--distance", type=float, default=1.0, help="cm")
    parser.add_argument("--max-velocity", type=float, default=0.1, help="rad/s per joint")
    parser.add_argument("--settle", type=float, default=2.0)
    args = parser.parse_args()

    config = XArm7Config(args.ip, max_velocity=np.full(NUM_JOINTS, args.max_velocity))
    with ExitStack() as stack:
        arm = XArm7(config)
        arm.start()
        stack.callback(arm.close)
        time.sleep(0.5)
        home_qpos = arm.home_qpos
        home_pose = arm.get_ee_pose()
        up_pose = home_pose.copy()
        up_pose[2] += args.distance / 100
        print(f"home_pose={np.array2string(home_pose, precision=6)}")

        for i in range(args.reps):
            arm.set_eepose(up_pose)
            time.sleep(args.settle)
            feedback_pose = arm.get_ee_pose()
            print(
                f"rep={i + 1}/{args.reps} up "
                f"command={np.array2string(up_pose, precision=6)} "
                f"feedback={np.array2string(feedback_pose, precision=6)} "
                f"position_error={np.linalg.norm(up_pose[:3] - feedback_pose[:3]):.6f} m"
            )

            arm.set_qpos(home_qpos)
            time.sleep(args.settle)
            feedback_qpos = arm.get_qpos()
            print(
                f"rep={i + 1}/{args.reps} return "
                f"command={np.array2string(home_qpos, precision=6)} "
                f"feedback={np.array2string(feedback_qpos, precision=6)} "
                f"max_abs_error={np.max(np.abs(home_qpos - feedback_qpos)):.6f} rad"
            )
    print("done")


if __name__ == "__main__":
    main()
