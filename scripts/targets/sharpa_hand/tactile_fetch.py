"""Fetch and print tactile frames from Sharpa hand (from SDK example).

Usage:
    uv run scripts/targets/sharpa_hand/tactile_fetch.py
    uv run scripts/targets/sharpa_hand/tactile_fetch.py --channel 0
"""

import argparse
import sys
import time

from retargetingkit.targets.hands.sharpa_hand import NUM_TACTILE_CHANNELS, SharpaHand


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=int, default=-1, help="single channel (0-4), or -1 for all")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    hand = SharpaHand()
    assert hand.is_tactile_ready(), "tactile subsystem not ready"
    if args.calibrate:
        assert hand.calib_tactile(), "tactile calibration failed"
        print("tactile calibrated")

    channels = [args.channel] if args.channel >= 0 else list(range(NUM_TACTILE_CHANNELS))
    dt = 1.0 / args.hz
    try:
        for _ in range(args.samples or sys.maxsize):
            for ch in channels:
                frame = hand.fetch_tactile_frame(ch, timeout=1.0)
                if frame is None:
                    continue
                content = frame["content"]
                parts = [f"ch{ch}({FINGER_NAMES[ch]}) ts={frame['ts']:.3f}"]
                if content.get("RAW") is not None:
                    parts.append(f"RAW={content['RAW'].shape}")
                if content.get("DEFORM") is not None:
                    parts.append(f"DEFORM={content['DEFORM'].shape}")
                if content.get("F6") is not None:
                    parts.append(f"F6=[{','.join(f'{value:.2f}' for value in content['F6'])}]")
                print("  ".join(parts))
            time.sleep(dt)
        print(hand.tactile_summary())
    finally:
        hand.close()


if __name__ == "__main__":
    main()
