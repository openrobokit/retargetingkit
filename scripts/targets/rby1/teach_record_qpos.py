"""Drag-to-teach RBY1-A right arm ('c' captures qpos, 'q' saves and quits); support the arm before quitting or it drops.

Usage:
    uv run --extra rby1 python scripts/targets/rby1/teach_record_qpos.py
"""

import argparse
import signal
import sys
import termios
import tty

import numpy as np

from retargetingkit.targets.humanoids.rby1 import NUM_ARM_JOINTS, RBY1AConfig, RBY1AMode


def _getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    ch = sys.stdin.read(1)
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="10.2.1.200:50051")
    parser.add_argument("--out", default="recorded_qpos.npy")
    args = parser.parse_args()

    input("Press Enter to enable gravity compensation (arm becomes weightless). ")
    arm = RBY1AConfig(args.ip, mode=RBY1AMode.TEACHING).build()
    signal.signal(signal.SIGINT, lambda *_: (arm.close(), sys.exit(1)))

    poses: list[np.ndarray] = []
    print("Drag the arm. 'c' = capture, 'q' = save & quit.")
    while True:
        ch = _getch()
        if ch == "c":
            q = arm.get_qpos()
            poses.append(q)
            print(f"[{len(poses)}] {np.array2string(q, precision=4)}")
        elif ch == "q":
            break

    input("Support the arm, then press Enter to disable control (arm goes limp). ")
    arm.close()
    arr = np.stack(poses) if poses else np.zeros((0, NUM_ARM_JOINTS))
    np.save(args.out, arr)
    print(f"saved {arr.shape} to {args.out}")


if __name__ == "__main__":
    main()
