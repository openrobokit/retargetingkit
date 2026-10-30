"""Move one XHand joint by a small relative amount and return."""

import argparse
import time

import numpy as np

from retargetingkit.targets.hands.xhand import JOINT_LIMIT_LOWER, JOINT_LIMIT_UPPER, JOINT_NAMES, XHand, XHandConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-port", default="/dev/ttyUSB0")
    parser.add_argument("--joint", type=int, default=4)
    parser.add_argument("--delta", type=float, default=0.05, help="Relative motion in radians")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--settle", type=float, default=0.5)
    args = parser.parse_args()
    assert 0 <= args.joint < len(JOINT_NAMES)

    config = XHandConfig(serial_port=args.serial_port)
    hand = XHand(config)
    hand.start()
    try:
        time.sleep(0.5)
        start = hand.get_qpos()
        moved = start.copy()
        moved[args.joint] = np.clip(
            moved[args.joint] + args.delta,
            JOINT_LIMIT_LOWER[args.joint],
            JOINT_LIMIT_UPPER[args.joint],
        )
        print(f"joint={args.joint} name={JOINT_NAMES[args.joint]} start={start[args.joint]:.6f} rad")
        for i in range(args.reps):
            for label, target in (("move", moved), ("return", start)):
                steps = max(1, round(args.duration / config.dt))
                for qpos in np.linspace(hand.get_qpos(), target, steps + 1)[1:]:
                    hand.set_qpos(qpos)
                    time.sleep(config.dt)
                time.sleep(args.settle)
                feedback = hand.get_qpos()
                print(
                    f"rep={i + 1}/{args.reps} {label} "
                    f"command={np.array2string(target, precision=6)} "
                    f"feedback={np.array2string(feedback, precision=6)} "
                    f"max_abs_error={np.max(np.abs(target - feedback)):.6f} rad"
                )
    finally:
        hand.close()
    print("done")


if __name__ == "__main__":
    main()
