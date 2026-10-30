"""Inspect Quest3 controller poses from NVIDIA Isaac Teleop.

Usage:
    uv run python scripts/sources/quest3/stream.py --side right
    uv run python scripts/sources/quest3/stream.py --side both
"""

import argparse
import time

from retargetingkit.sources.vr.quest3 import LEFT, RIGHT, Quest3Config
from robokit.xform.numpy import matrix_to_quaternion


def main() -> None:
    parser = argparse.ArgumentParser(description="Quest3 Isaac Teleop stream inspector")
    parser.add_argument("--side", choices=(LEFT, RIGHT, "both"), default=RIGHT, help="Controller side to print")
    parser.add_argument("--app-name", type=str, default="RetargetingKitQuest3Stream", help="OpenXR app name")
    parser.add_argument("--rate", type=float, default=10.0, help="Print rate in Hz")
    args = parser.parse_args()

    sides = (LEFT, RIGHT) if args.side == "both" else (args.side,)
    quest = Quest3Config(app_name=args.app_name, sides=sides).build()
    dt = 1.0 / args.rate
    print("Streaming Quest3 controller poses from Isaac Teleop. Ctrl+C to stop.\n")
    while True:
        frame = quest.get_frame()
        age_ms = (time.time_ns() - frame.timestamp_ns) / 1e6
        parts = [f"age_ms={age_ms:.1f}"]
        for side in sides:
            if side not in frame.controllers:
                parts.append(f"{side}=invalid")
                continue
            pose = frame.controllers[side].pose
            qw, qx, qy, qz = matrix_to_quaternion(pose[:3, :3])
            parts.append(
                "%s xyz=(%+.3f,%+.3f,%+.3f) quat_wxyz=(%+.3f,%+.3f,%+.3f,%+.3f)"
                % (
                    side,
                    pose[0, 3],
                    pose[1, 3],
                    pose[2, 3],
                    qw,
                    qx,
                    qy,
                    qz,
                )
            )
        print(" | ".join(parts))
        time.sleep(dt)


if __name__ == "__main__":
    main()
