"""Read and print Sharpa hand joint positions and velocities.

Usage:
    uv run scripts/targets/sharpa_hand/read_qpos.py
    uv run scripts/targets/sharpa_hand/read_qpos.py --hz 50
"""

import argparse
import time

import numpy as np

from retargetingkit.targets.hands.sharpa_hand import JOINT_NAMES, SharpaHand


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--samples", type=int, default=0)
    args = parser.parse_args()

    hand = SharpaHand(control_mode="floating")
    try:
        print("FLOATING mode -- move the hand freely. Ctrl+C to quit.\n")
        dt = 1.0 / args.hz
        count = 0
        while args.samples == 0 or count < args.samples:
            angles, _, _ = hand.get_state()
            parts = [f"{name}: {np.degrees(angle):7.2f}deg" for name, angle in zip(JOINT_NAMES, angles)]
            print(" | ".join(parts[:5]))
            print(" | ".join(parts[5:9]))
            print(" | ".join(parts[9:13]))
            print(" | ".join(parts[13:17]))
            print(" | ".join(parts[17:]))
            print()
            count += 1
            time.sleep(dt)
    finally:
        hand.close()


if __name__ == "__main__":
    main()
