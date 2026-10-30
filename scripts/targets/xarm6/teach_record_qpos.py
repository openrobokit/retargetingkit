"""Drag-to-teach xArm6 in TEACHING mode ('c' captures qpos, 'q' saves and quits); support the arm before quitting.

Usage:
    uv run --extra xarm scripts/targets/xarm6/teach_record_qpos.py
"""

import argparse
import signal
import sys
import termios
import tty
from pathlib import Path

import numpy as np

from retargetingkit.targets.arms.xarm6 import NUM_JOINTS, XArm6, XArm6Config, XArmMode


def _getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    ch = sys.stdin.read(1)
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="192.168.1.228")
    parser.add_argument("--out", default=str(Path(__file__).parent / "recorded_qpos.npy"))
    args = parser.parse_args()

    arm = XArm6(XArm6Config(args.ip, mode=XArmMode.TEACHING))
    input("Press Enter to enable manual mode (arm becomes draggable). ")
    arm.start()
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

    input("Support the arm, then press Enter to disable control. ")
    arm.close()
    arr = np.stack(poses) if poses else np.zeros((0, NUM_JOINTS))
    np.save(args.out, arr)
    print(f"saved {arr.shape} to {args.out}")


if __name__ == "__main__":
    main()
