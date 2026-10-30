"""Exercise every `Pico` API; needs XRoboToolkit PC Service (Full Body mode + ankle trackers for body tests).

Usage:
    uv run python scripts/sources/pico/test_all_apis.py
    uv run python scripts/sources/pico/test_all_apis.py --skip-body   # controllers + buttons only
    uv run python scripts/sources/pico/test_all_apis.py --skip-hands  # skip hand tracking
"""

import argparse
import time

import numpy as np

from retargetingkit.sources.vr.pico import Pico


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def test_headset_and_controllers(pico: Pico, duration: float = 2.0) -> None:
    section("Headset + Controllers (poses)")
    print("Move headset and controllers around.\n")
    deadline = time.monotonic() + duration
    samples = 0
    while time.monotonic() < deadline:
        head = pico.get_headset_pose()
        left = pico.get_left_controller_pose()
        right = pico.get_right_controller_pose()
        assert head.shape == (7,), f"headset pose shape: {head.shape}"
        assert left.shape == (7,), f"left controller shape: {left.shape}"
        assert right.shape == (7,), f"right controller shape: {right.shape}"
        assert head.dtype == np.float32
        if samples % 5 == 0:
            print(
                f"  head xyz=({head[0]:+.3f},{head[1]:+.3f},{head[2]:+.3f})  "
                f"L=({left[0]:+.3f},{left[1]:+.3f},{left[2]:+.3f})  "
                f"R=({right[0]:+.3f},{right[1]:+.3f},{right[2]:+.3f})"
            )
        samples += 1
        time.sleep(0.1)
    print(f"\n[OK] {samples} samples. Poses are (7,) float32.")


def test_triggers_and_grips(pico: Pico, duration: float = 3.0) -> None:
    section("Triggers + Grips (analog 0.0-1.0)")
    print("Squeeze triggers and grips on both controllers.\n")
    deadline = time.monotonic() + duration
    samples = 0
    max_lt, max_rt, max_lg, max_rg = 0.0, 0.0, 0.0, 0.0
    while time.monotonic() < deadline:
        lt = pico.get_left_trigger()
        rt = pico.get_right_trigger()
        lg = pico.get_left_grip()
        rg = pico.get_right_grip()
        assert isinstance(lt, float)
        assert isinstance(rt, float)
        assert isinstance(lg, float)
        assert isinstance(rg, float)
        max_lt = max(max_lt, lt)
        max_rt = max(max_rt, rt)
        max_lg = max(max_lg, lg)
        max_rg = max(max_rg, rg)
        if samples % 5 == 0:
            print(f"  L_trig={lt:.2f}  R_trig={rt:.2f}  L_grip={lg:.2f}  R_grip={rg:.2f}")
        samples += 1
        time.sleep(0.1)
    print(f"\n  Max values: L_trig={max_lt:.2f}  R_trig={max_rt:.2f}  L_grip={max_lg:.2f}  R_grip={max_rg:.2f}")
    if max(max_lt, max_rt, max_lg, max_rg) < 0.01:
        print("  [WARN] No trigger/grip input detected.")
    else:
        print("  [OK] Trigger/grip input detected.")


def test_buttons(pico: Pico, duration: float = 3.0) -> None:
    section("Buttons (digital)")
    button_names = ["A", "B", "X", "Y", "left_menu", "right_menu", "left_axis_click", "right_axis_click"]
    print(f"Press buttons: {', '.join(button_names)}\n")
    pressed: set[str] = set()
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        for name in button_names:
            val = pico.get_button(name)
            assert isinstance(val, bool), f"button {name} returned {type(val)}"
            if val and name not in pressed:
                print(f"  [{name}] pressed!")
                pressed.add(name)
        time.sleep(0.05)
    print(f"\n  Buttons detected: {sorted(pressed) if pressed else '(none)'}")
    if not pressed:
        print("  [WARN] No button presses detected.")
    else:
        print(f"  [OK] {len(pressed)}/{len(button_names)} buttons tested.")


def test_joysticks(pico: Pico, duration: float = 2.0) -> None:
    section("Joysticks (x, y)")
    print("Move joysticks on both controllers.\n")
    deadline = time.monotonic() + duration
    max_deflection_l, max_deflection_r = 0.0, 0.0
    samples = 0
    while time.monotonic() < deadline:
        lj = pico.get_left_joystick()
        rj = pico.get_right_joystick()
        assert lj.shape == (2,), f"left joystick shape: {lj.shape}"
        assert rj.shape == (2,), f"right joystick shape: {rj.shape}"
        assert lj.dtype == np.float32
        max_deflection_l = max(max_deflection_l, float(np.linalg.norm(lj)))
        max_deflection_r = max(max_deflection_r, float(np.linalg.norm(rj)))
        if samples % 5 == 0:
            print(f"  L=({lj[0]:+.3f},{lj[1]:+.3f})  R=({rj[0]:+.3f},{rj[1]:+.3f})")
        samples += 1
        time.sleep(0.1)
    print(f"\n  Max deflection: L={max_deflection_l:.3f}  R={max_deflection_r:.3f}")
    if max(max_deflection_l, max_deflection_r) < 0.01:
        print("  [WARN] No joystick movement detected.")
    else:
        print("  [OK] Joystick input detected.")


def test_timestamp(pico: Pico) -> None:
    section("Timestamp")
    ts1 = pico.get_timestamp_ns()
    time.sleep(0.1)
    ts2 = pico.get_timestamp_ns()
    assert isinstance(ts1, int)
    assert isinstance(ts2, int)
    delta_ms = (ts2 - ts1) / 1e6
    print(f"  ts1={ts1}  ts2={ts2}  delta={delta_ms:.1f} ms")
    if delta_ms < 1.0:
        print("  [WARN] Timestamp delta < 1 ms — clock may not be advancing.")
    else:
        print("  [OK] Timestamp advancing.")


def test_motion_trackers(pico: Pico) -> None:
    section("Motion Trackers (ankle trackers etc.)")
    trackers = pico.get_motion_trackers()
    if not trackers:
        print("  No motion trackers detected. Skipping.")
        print("  (Pair ankle trackers in XRoboToolKit app to test this.)")
        return
    print(f"  Found {len(trackers)} tracker(s):\n")
    for serial, data in trackers.items():
        pose = data["pose"]
        vel = data["velocity"]
        acc = data["acceleration"]
        assert pose.shape == (7,), f"tracker {serial} pose shape: {pose.shape}"
        assert vel.shape == (6,), f"tracker {serial} velocity shape: {vel.shape}"
        assert acc.shape == (6,), f"tracker {serial} acceleration shape: {acc.shape}"
        print(f"  [{serial}] pose xyz=({pose[0]:+.3f},{pose[1]:+.3f},{pose[2]:+.3f})")
    print(f"\n  [OK] {len(trackers)} tracker(s) reporting.")


def test_body_tracking(pico: Pico, duration: float = 3.0) -> None:
    section("Body Tracking (24 SMPL joints)")
    print("Waiting for body data (up to 10 s)…")
    t0 = time.monotonic()
    while not pico.is_body_available():
        if time.monotonic() - t0 > 10.0:
            print("  [FAIL] Body data not available within 10 s.")
            print("         Wear headset, enable Full Body, pair ankle trackers.")
            return
        time.sleep(0.05)
    print(f"  [OK] Body data available after {time.monotonic() - t0:.1f} s.\n")

    print("Streaming body poses + velocities + accelerations…\n")
    deadline = time.monotonic() + duration
    samples = 0
    while time.monotonic() < deadline:
        poses = pico.get_body_poses()
        vels = pico.get_body_velocities()
        accs = pico.get_body_accelerations()
        assert poses.shape == (24, 7), f"body poses shape: {poses.shape}"
        assert vels.shape == (24, 6), f"body velocities shape: {vels.shape}"
        assert accs.shape == (24, 6), f"body accelerations shape: {accs.shape}"
        assert poses.dtype == np.float32
        assert vels.dtype == np.float32
        assert accs.dtype == np.float32
        assert np.all(np.isfinite(poses)), "body poses contain non-finite values"
        pelvis = poses[0]
        q_norm = float(np.linalg.norm(pelvis[3:]))
        if samples % 5 == 0:
            print(
                f"  pelvis xyz=({pelvis[0]:+.3f},{pelvis[1]:+.3f},{pelvis[2]:+.3f}) "
                f"quat_norm={q_norm:.4f}  vel_norm={float(np.linalg.norm(vels[0])):.3f}"
            )
        samples += 1
        time.sleep(0.1)
    print(f"\n  [OK] {samples} body frames. Shape (24,7) poses + (24,6) vel + (24,6) acc.")


def test_hand_tracking(pico: Pico, duration: float = 3.0) -> None:
    section("Hand Tracking (27 joints per hand)")
    print("Hold hands in front of headset cameras.\n")
    deadline = time.monotonic() + duration
    left_seen, right_seen = False, False
    samples = 0
    while time.monotonic() < deadline:
        l_active = pico.is_left_hand_active()
        r_active = pico.is_right_hand_active()
        assert isinstance(l_active, bool)
        assert isinstance(r_active, bool)

        left = pico.get_left_hand()
        right = pico.get_right_hand()

        status_parts = []
        if left is not None:
            assert left.shape == (27, 7), f"left hand shape: {left.shape}"
            assert left.dtype == np.float32
            left_seen = True
            wrist = left[0]
            status_parts.append(f"L wrist=({wrist[0]:+.3f},{wrist[1]:+.3f},{wrist[2]:+.3f})")
        else:
            status_parts.append("L inactive")

        if right is not None:
            assert right.shape == (27, 7), f"right hand shape: {right.shape}"
            assert right.dtype == np.float32
            right_seen = True
            wrist = right[0]
            status_parts.append(f"R wrist=({wrist[0]:+.3f},{wrist[1]:+.3f},{wrist[2]:+.3f})")
        else:
            status_parts.append("R inactive")

        if samples % 5 == 0:
            print(f"  {'  '.join(status_parts)}")
        samples += 1
        time.sleep(0.1)

    print(f"\n  Left seen: {left_seen}  Right seen: {right_seen}")
    if not left_seen and not right_seen:
        print("  [WARN] No hand tracking detected. Hold hands in camera view.")
    else:
        print("  [OK] Hand tracking working.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive PICO SDK test")
    parser.add_argument("--skip-body", action="store_true", help="Skip body tracking test")
    parser.add_argument("--skip-hands", action="store_true", help="Skip hand tracking test")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration per test section (s)")
    args = parser.parse_args()

    print("Initializing PICO device…")
    pico = Pico()

    test_headset_and_controllers(pico, duration=args.duration)
    test_triggers_and_grips(pico, duration=args.duration)
    test_buttons(pico, duration=args.duration)
    test_joysticks(pico, duration=args.duration)
    test_timestamp(pico)
    test_motion_trackers(pico)

    if not args.skip_body:
        test_body_tracking(pico, duration=args.duration)

    if not args.skip_hands:
        test_hand_tracking(pico, duration=args.duration)

    pico.close()

    section("DONE")
    print("All API tests completed. Check [WARN] items above for issues.\n")


if __name__ == "__main__":
    main()
