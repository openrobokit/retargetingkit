"""Open and close the Ability Hand repeatedly."""

import argparse
import signal
import sys
import time

import numpy as np

from retargetingkit.targets.hands.ability_hand import AbilityHand, AbilityHandConfig


OPEN = np.deg2rad([9, 9, 9, 9, 9, -9]).astype(np.float64)  # 9 deg
CLOSED = np.deg2rad([70, 70, 70, 70, 70, -70]).astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", default="/dev/serial/by-id/usb-Prolific_Technology_Inc._USB-Serial_Controller_BUBZb11A922-if00-port0"
    )
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    hand = AbilityHand(AbilityHandConfig(args.port))
    hand.start()
    time.sleep(0.5)

    signal.signal(signal.SIGINT, lambda *_: (hand.close(), sys.exit(1)))

    for i in range(args.reps):
        for label, target in [("close", CLOSED), ("open", OPEN)]:
            print(f"rep {i + 1}/{args.reps}: {label}")
            hand.set_qpos(target)
            t0 = time.time()
            while time.time() - t0 < 3.0:
                if np.max(np.abs(hand.get_qpos() - target)) < 0.1:
                    break
                time.sleep(0.02)

    hand.close()
    print("done")


if __name__ == "__main__":
    main()
