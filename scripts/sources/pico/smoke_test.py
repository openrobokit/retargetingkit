"""PICO SDK smoke test: validates connectivity and body tracking via XRoboToolkit PC Service with the Pico wrapper.

Two modes:
  --no-body   Connectivity only: prints headset + controller poses.
  (default)   Full body: waits for body data, streams 5 seconds.

Usage:
    uv run python scripts/sources/pico/smoke_test.py
    uv run python scripts/sources/pico/smoke_test.py --no-body
"""

import argparse
import time

import numpy as np

from retargetingkit.sources.vr.pico import BODY_JOINT_NAMES as JOINT_NAMES
from retargetingkit.sources.vr.pico import Pico


def test_connectivity(pico: Pico, duration_s: float = 3.0) -> None:
    print("── Connectivity test ─────────────────────────────────────")
    deadline = time.monotonic() + duration_s
    tick = 0
    while time.monotonic() < deadline:
        head = pico.get_headset_pose()
        left = pico.get_left_controller_pose()
        right = pico.get_right_controller_pose()
        print(
            f"[{tick:02d}] "
            f"head xyz=({head[0]:+.2f},{head[1]:+.2f},{head[2]:+.2f}) "
            f"L xyz=({left[0]:+.2f},{left[1]:+.2f},{left[2]:+.2f}) "
            f"R xyz=({right[0]:+.2f},{right[1]:+.2f},{right[2]:+.2f})"
        )
        tick += 1
        time.sleep(0.3)

    head = pico.get_headset_pose()
    if np.allclose(head, 0.0):
        print("\n[WARN] Headset pose is all zeros — PICO app may not be connected.")
    else:
        print("\n[OK] Headset pose has non-zero values — SDK link is alive.")


def test_body_stream(pico: Pico, duration_s: float = 5.0, rate_hz: float = 10.0) -> None:
    print("── Body stream test ──────────────────────────────────────")
    print("Waiting for body tracking data (up to 10 s)…")

    t0 = time.monotonic()
    while not pico.is_body_available():
        if time.monotonic() - t0 > 10.0:
            raise SystemExit("[ERROR] Body data not available within 10 s.")
        time.sleep(0.05)

    print(f"[OK] Body data available after {time.monotonic() - t0:.1f} s.\n")

    period = 1.0 / rate_hz
    pelvis_history: list[np.ndarray] = []
    deadline = time.monotonic() + duration_s

    while time.monotonic() < deadline:
        poses = pico.get_body_poses()
        pelvis = poses[0]
        pelvis_history.append(pelvis[:3].copy())
        print(
            f"pelvis xyz=({pelvis[0]:+.3f},{pelvis[1]:+.3f},{pelvis[2]:+.3f}) "
            f"qxyzw=({pelvis[3]:+.3f},{pelvis[4]:+.3f},{pelvis[5]:+.3f},{pelvis[6]:+.3f})"
        )
        time.sleep(period)

    history = np.stack(pelvis_history)
    motion_range = history.max(axis=0) - history.min(axis=0)
    print(f"\n[INFO] Pelvis motion range: dx={motion_range[0]:.3f} dy={motion_range[1]:.3f} dz={motion_range[2]:.3f} m")
    if motion_range.max() < 1e-3:
        print("[WARN] Pelvis did not move.")
    else:
        print("[OK] Live stream confirmed.")

    poses = pico.get_body_poses()
    print("\n── Final snapshot (24 joints, Y-up) ──")
    for i, name in enumerate(JOINT_NAMES):
        x, y, z = poses[i, :3]
        print(f"  {i:2d}  {name:<16s}  xyz=({x:+.2f}, {y:+.2f}, {z:+.2f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="PICO smoke test")
    parser.add_argument("--no-body", action="store_true", help="Connectivity only")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--rate", type=float, default=10.0)
    args = parser.parse_args()

    pico = Pico()
    if args.no_body:
        test_connectivity(pico)
    else:
        test_connectivity(pico, duration_s=1.5)
        print()
        test_body_stream(pico, duration_s=args.duration, rate_hz=args.rate)
    pico.close()
    print("\n[OK] Done.")


if __name__ == "__main__":
    main()
