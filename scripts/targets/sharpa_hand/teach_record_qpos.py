"""Drag-to-teach Sharpa hand: press 'c' to capture qpos, 'q' to save and quit.

Usage:
    uv run scripts/targets/sharpa_hand/teach_record_qpos.py
"""

import argparse
import signal
import sys
from pathlib import Path

import numpy as np

from retargetingkit.targets.hands.sharpa_hand import NUM_JOINTS, SharpaHand


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-sn")
    parser.add_argument("--out", default=str(Path(__file__).parent / "recorded_qpos.npy"))
    args = parser.parse_args()

    hand = SharpaHand(device_sn=args.device_sn)
    input("Press Enter to enable floating mode (hand becomes draggable). ")
    hand.set_control_mode("floating")
    signal.signal(signal.SIGINT, lambda *_: (hand.close(), sys.exit(1)))

    poses: list[np.ndarray] = []
    print("Move the hand freely. 'c' + Enter = capture, 'q' + Enter = save & quit.")
    while (command := input("> ")) != "q":
        if command == "c":
            qpos = hand.get_qpos()
            poses.append(qpos)
            print(f"[{len(poses)}] {np.array2string(qpos, precision=4)}")

    hand.close()
    targets = np.stack(poses) if poses else np.zeros((0, NUM_JOINTS))
    np.save(args.out, targets)
    print(f"saved {targets.shape} target qpos to {args.out}")


if __name__ == "__main__":
    main()
