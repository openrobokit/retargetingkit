"""Read and print Ability Hand joint positions."""

import argparse
import time

from retargetingkit.targets.hands.ability_hand import AbilityHand, AbilityHandConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", default="/dev/serial/by-id/usb-Prolific_Technology_Inc._USB-Serial_Controller_BUBZb11A922-if00-port0"
    )
    parser.add_argument("--hz", type=float, default=10.0)
    args = parser.parse_args()

    hand = AbilityHand(AbilityHandConfig(args.port))
    hand.start()

    dt = 1.0 / args.hz
    while True:
        print(hand.get_qpos())
        time.sleep(dt)


if __name__ == "__main__":
    main()
