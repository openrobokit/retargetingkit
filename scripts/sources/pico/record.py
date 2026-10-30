"""Record a PICO body-tracking session to .npz so `BodyFileSource` can replay it offline without the headset.

Output .npz fields:
  poses        : (N, 24, 7) float32  -- raw SDK output [x,y,z, qx,qy,qz,qw], Y-up
  timestamps_s : (N,)       float64  -- seconds since recording start (wall clock)
  metadata     : str (JSON) -- recording info (date, duration, fps, joint names)

Usage:
    uv run python scripts/sources/pico/record.py
    uv run python scripts/sources/pico/record.py --duration 30 --out my_walk.npz
    uv run python scripts/sources/pico/record.py --rate 60
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from retargetingkit.sources.vr.pico import BODY_JOINT_NAMES as JOINT_NAMES
from retargetingkit.sources.vr.pico import Pico


def main() -> int:
    parser = argparse.ArgumentParser(description="Record PICO body stream to .npz")
    parser.add_argument("--duration", type=float, default=15.0, help="Recording length (s)")
    parser.add_argument("--rate", type=float, default=50.0, help="Poll rate (Hz)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("assets/pico_recordings/default.npz"),
        help="Output .npz path",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    pico = Pico()

    print("Waiting for body data (up to 10 s)…")
    t0 = time.monotonic()
    while not pico.is_body_available():
        if time.monotonic() - t0 > 10.0:
            print("[ERROR] Body data did not become available within 10 s.", file=sys.stderr)
            pico.close()
            return 1
        time.sleep(0.05)
    print(f"[OK] Body data available after {time.monotonic() - t0:.1f} s.\n")

    period = 1.0 / args.rate
    poses_list: list[np.ndarray] = []
    timestamps: list[float] = []

    print(f"Recording {args.duration:.0f} s @ {args.rate:.0f} Hz → {args.out}")

    start = time.monotonic()
    next_tick = start
    last_print = start
    while True:
        now = time.monotonic()
        if now - start >= args.duration:
            break
        if now < next_tick:
            time.sleep(min(next_tick - now, 0.005))
            continue
        poses = pico.get_body_poses()
        if poses.shape != (24, 7):
            print(f"[WARN] Unexpected pose shape {poses.shape}, skipping", file=sys.stderr)
            next_tick += period
            continue
        poses_list.append(poses)
        timestamps.append(now - start)
        next_tick += period

        if now - last_print >= 1.0:
            elapsed = now - start
            print(f"  [{elapsed:5.1f}s] {len(poses_list)} frames captured")
            last_print = now

    pico.close()

    if not poses_list:
        print("[ERROR] No frames captured.", file=sys.stderr)
        return 1

    poses_arr = np.stack(poses_list, axis=0).astype(np.float32)
    ts_arr = np.asarray(timestamps, dtype=np.float64)

    measured_fps = (len(ts_arr) - 1) / (ts_arr[-1] - ts_arr[0]) if len(ts_arr) > 1 else 0.0
    metadata = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "duration_s": float(ts_arr[-1] - ts_arr[0]),
        "num_frames": int(len(poses_arr)),
        "measured_fps": float(measured_fps),
        "joint_names": JOINT_NAMES,
        "format": "raw_sdk_yup_xyzw",
        "shape": "(N, 24, 7)",
    }

    np.savez_compressed(
        args.out,
        poses=poses_arr,
        timestamps_s=ts_arr,
        metadata=json.dumps(metadata),
    )
    print(f"\nSaved {len(poses_arr)} frames ({ts_arr[-1]:.2f} s, {measured_fps:.1f} Hz) → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
