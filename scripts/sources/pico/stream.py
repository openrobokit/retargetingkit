"""Live PICO data stream — prints all active channels in a compact real-time display.

Usage:
    uv run python scripts/sources/pico/stream.py
    uv run python scripts/sources/pico/stream.py --rate 30
    uv run python scripts/sources/pico/stream.py --no-body --no-hands
"""

import argparse
import time

import numpy as np

from retargetingkit.sources.vr.pico import BUTTON_NAMES, Pico


def fmt_xyz(arr: np.ndarray) -> str:
    return f"({arr[0]:+6.3f},{arr[1]:+6.3f},{arr[2]:+6.3f})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Live PICO data stream")
    parser.add_argument("--rate", type=float, default=20.0, help="Print rate (Hz)")
    parser.add_argument("--no-body", action="store_true", help="Skip body tracking")
    parser.add_argument("--no-hands", action="store_true", help="Skip hand tracking")
    args = parser.parse_args()

    pico = Pico()
    period = 1.0 / args.rate
    tick = 0

    print("Streaming PICO data. Ctrl+C to stop.\n")

    while True:
        t0 = time.monotonic()
        lines: list[str] = []
        lines.append(f"── tick {tick:06d} ──")

        # Headset + controllers
        head = pico.get_headset_pose()
        lc = pico.get_left_controller_pose()
        rc = pico.get_right_controller_pose()
        lines.append(f"Head  {fmt_xyz(head[:3])}  L_ctrl {fmt_xyz(lc[:3])}  R_ctrl {fmt_xyz(rc[:3])}")

        # Triggers + grips
        lt = pico.get_left_trigger()
        rt = pico.get_right_trigger()
        lg = pico.get_left_grip()
        rg = pico.get_right_grip()
        lines.append(f"Trig  L={lt:.2f} R={rt:.2f}  Grip  L={lg:.2f} R={rg:.2f}")

        # Joysticks
        lj = pico.get_left_joystick()
        rj = pico.get_right_joystick()
        lines.append(f"Joy   L=({lj[0]:+.2f},{lj[1]:+.2f})  R=({rj[0]:+.2f},{rj[1]:+.2f})")

        # Buttons — only show pressed ones
        pressed = [name for name in BUTTON_NAMES if pico.get_button(name)]
        lines.append(f"Btns  [{', '.join(pressed) if pressed else '-'}]")

        # Motion trackers
        trackers = pico.get_motion_trackers()
        if trackers:
            parts = [f"{s[-6:]}={fmt_xyz(d['pose'][:3])}" for s, d in trackers.items()]
            lines.append(f"Track {' '.join(parts)}")

        # Body
        if not args.no_body and pico.is_body_available():
            poses = pico.get_body_poses()
            pelvis = poses[0]
            lw = poses[20]
            rw = poses[21]
            lines.append(f"Body  pelvis{fmt_xyz(pelvis[:3])}  L_wrist{fmt_xyz(lw[:3])}  R_wrist{fmt_xyz(rw[:3])}")

        # Hands
        if not args.no_hands:
            lh = pico.get_left_hand()
            rh = pico.get_right_hand()
            lh_str = fmt_xyz(lh[0, :3]) if lh is not None else "inactive"
            rh_str = fmt_xyz(rh[0, :3]) if rh is not None else "inactive"
            lines.append(f"Hands L_wrist={lh_str}  R_wrist={rh_str}")

        # Print with clear
        output = "\n".join(lines)
        n_lines = len(lines)
        print(output)
        # Move cursor up to overwrite on next iteration
        print(f"\033[{n_lines}A", end="")

        tick += 1
        elapsed = time.monotonic() - t0
        if elapsed < period:
            time.sleep(period - elapsed)


if __name__ == "__main__":
    main()
