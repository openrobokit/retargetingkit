"""Move one Sharpa joint by a small relative amount and return.

Usage:
    uv run scripts/targets/sharpa_hand/wave_example.py
"""

import argparse
import time

import numpy as np

from retargetingkit.targets.hands.sharpa_hand import (
    JOINT_LIMIT_LOWER,
    JOINT_LIMIT_UPPER,
    JOINT_NAMES,
    NUM_JOINTS,
    SharpaHand,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", type=int, default=7)
    parser.add_argument("--amplitude-deg", type=float, default=3.0)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--speed", type=float, default=0.1)
    parser.add_argument("--current", type=float, default=0.3)
    parser.add_argument("--settle", type=float, default=2.0)
    args = parser.parse_args()
    assert 0 <= args.joint < NUM_JOINTS

    hand = SharpaHand(speed_coeff=args.speed, current_coeff=args.current)
    try:
        home = hand.get_qpos()
        moved = home.copy()
        moved[args.joint] += np.radians(args.amplitude_deg)
        assert JOINT_LIMIT_LOWER[args.joint] <= moved[args.joint] <= JOINT_LIMIT_UPPER[args.joint]
        print(f"joint={args.joint} name={JOINT_NAMES[args.joint]} start={home[args.joint]:.6f} rad")
        for cycle in range(args.cycles):
            for label, command in (("move", moved), ("return", home)):
                hand.set_qpos(command)
                time.sleep(args.settle)
                feedback = hand.get_qpos()
                print(
                    f"cycle={cycle + 1}/{args.cycles} {label} "
                    f"command={np.array2string(command, precision=6)} "
                    f"feedback={np.array2string(feedback, precision=6)} "
                    f"max_abs_error={np.max(np.abs(command - feedback)):.6f} rad"
                )
    finally:
        hand.close()


if __name__ == "__main__":
    main()
