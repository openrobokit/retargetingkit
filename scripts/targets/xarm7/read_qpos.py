"""Read and print xArm7 joint positions."""

import argparse
import sys
import time

import numpy as np
from jaxtyping import Float64

from retargetingkit.targets.arms.xarm7 import NUM_JOINTS, XARM7_JOINT_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="192.168.1.242")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--samples", type=int, default=0)
    args = parser.parse_args()

    from xarm import XArmAPI  # type: ignore[import-unresolved]

    arm = XArmAPI(args.ip, is_radian=True)
    dt = 1.0 / args.hz
    print(f"joint_names={XARM7_JOINT_NAMES}")
    for _ in range(args.samples or sys.maxsize):
        code, state = arm.get_joint_states(is_radian=True)
        assert code == 0, f"get_joint_states failed: code={code}"
        qpos: Float64[np.ndarray, "7"] = np.asarray(state[0])
        assert qpos.shape == (NUM_JOINTS,), f"qpos shape must be ({NUM_JOINTS},), got {qpos.shape}"
        assert qpos.dtype == np.float64, f"qpos dtype must be float64, got {qpos.dtype}"
        assert np.isfinite(qpos).all(), "qpos must be finite"
        print(f"qpos_rad={qpos.tolist()} shape={qpos.shape} dtype={qpos.dtype} finite=True")
        time.sleep(dt)
    arm.disconnect()


if __name__ == "__main__":
    main()
