"""Move both Dexmate arms to the rest pose; does NOT check self-collision, so keep the space clear and e-stop ready.

Usage:
    uv run --extra dexmate scripts/targets/dexmate/move_arms.py
"""

import argparse

import numpy as np

from retargetingkit.targets.humanoids.dexmate import DexmateConfig


# Reference bimanual rest pose [left(7), right(7)] from SharpaDexmateTeleop.
REST_QPOS = np.array([0.74, 0.51, 0.37, -1.30, -0.65, -0.29, -0.03, -0.74, -0.51, -0.37, -1.30, 0.65, 0.29, 0.03])
# REST_QPOS = np.array(
#     [0.84, 0.51, 0.37, -1.30, -0.65, -0.29, -0.03, -0.74, -0.51, -0.37, -1.30, 0.65 + np.pi / 2, 0.29, 0.03 - np.pi / 3]
# )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", type=float, default=3.0, help="seconds per move")
    args = parser.parse_args()

    robot = DexmateConfig().build()
    qpos = robot.get_qpos()
    qpos[robot.joint_names.index("L_arm_j1") :] = REST_QPOS  # torso/head hold their measured pose
    print("moving to rest pose")
    robot.set_qpos(qpos, wait_time=args.wait)
    robot.close()


if __name__ == "__main__":
    main()
