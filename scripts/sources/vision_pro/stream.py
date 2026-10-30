"""Print Vision Pro tracking data in a loop. Ctrl-C to stop."""

import argparse
import time

from retargetingkit.sources.vr.vision_pro import VisionPro


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream Vision Pro tracking data")
    parser.add_argument("--ip", type=str)
    parser.add_argument("--recording", type=str, default=None, help="Pickle file for playback")
    args = parser.parse_args()

    vp = VisionPro(ip=args.ip, recording_path=args.recording)

    while True:
        frame = vp.get_all()
        head = frame["head"]
        lp = frame["left_pinch_distance"]
        rp = frame["right_pinch_distance"]
        pos = head[0, :3, 3]
        print(
            f"\rhead: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]  pinch L={lp:.3f} R={rp:.3f}\033[K",
            end="",
            flush=True,
        )
        time.sleep(0.05)


if __name__ == "__main__":
    main()
