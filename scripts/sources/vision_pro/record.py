"""Record Vision Pro tracking frames to a pickle file."""

import argparse
import copy
import pickle
import time
from datetime import datetime
from pathlib import Path

from retargetingkit.sources.vr.vision_pro import VisionPro


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Vision Pro tracking frames")
    parser.add_argument("--ip", required=True)
    parser.add_argument("--duration", type=float, default=15.0, help="Recording duration in seconds")
    parser.add_argument("--rate", type=float, default=50.0, help="Polling rate in Hz")
    parser.add_argument("--output", type=Path, help="Output .pkl path")
    args = parser.parse_args()

    output = args.output or Path("data") / f"avp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    output.parent.mkdir(parents=True, exist_ok=True)
    vision = VisionPro(ip=args.ip)
    frames: list[dict] = []
    period = 1.0 / args.rate
    count = round(args.duration * args.rate)

    for index in range(count):
        started = time.monotonic()
        frames.append(copy.deepcopy(vision.get_all()))
        print(f"\rRecording {index + 1}/{count}", end="", flush=True)
        time.sleep(max(0.0, period - (time.monotonic() - started)))

    with output.open("wb") as file:
        pickle.dump(frames, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved {len(frames)} frames to {output}")


if __name__ == "__main__":
    main()
