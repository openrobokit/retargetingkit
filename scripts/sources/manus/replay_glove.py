"""Replay recorded Manus glove data with viser visualization."""

import argparse
import time
from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf

from retargetingkit.assets import calib
from retargetingkit.retargeting.hand.neural import NeuralHandRetargeterConfig
from retargetingkit.retargeting.hand.presets import HANDS
from retargetingkit.sources.glove.manus_conversion import (
    apply_glove_corrections,
    load_glove_calibration,
    manus_positions_to_fixed_wrist_local,
    manus_to_mano_keypoints,
)
from retargetingkit.utils.viz import add_hand_urdf, draw_manus_skeleton


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay recorded Manus glove data")
    parser.add_argument("recording", type=Path)
    parser.add_argument("--hand", choices=["left", "right"], default=None, help="Which hand to replay")
    parser.add_argument("--checkpoint", type=Path, help="Show the retargeted Sharpa hand")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--manus-calib", type=Path)
    args = parser.parse_args()

    data = np.load(args.recording)
    available_sides = [side for side in ("left", "right") if f"{side}_positions" in data]
    assert len(available_sides) > 0, "No hand data found in recording"
    if args.hand:
        assert args.hand in available_sides, f"{args.hand} hand not in recording (available: {available_sides})"
        sides = [args.hand]
    else:
        sides = available_sides

    for side in sides:
        n = len(data[f"{side}_timestamps"])
        duration = data[f"{side}_timestamps"][-1]
        print(f"  {side}: {n} frames, {duration:.1f}s")

    calibration_path = calib.get("manus_calib") if args.manus_calib is None else args.manus_calib
    glove_calib = load_glove_calibration(calibration_path)
    print(f"Loaded glove calibration from: {calibration_path}")
    retargeter = (
        NeuralHandRetargeterConfig(args.checkpoint, HANDS["sharpa"]).build(args.device)
        if args.checkpoint is not None
        else None
    )

    server = viser.ViserServer()
    server.scene.add_grid("/grid", 2.0, 2.0)
    viser_urdfs: dict[str, ViserUrdf] = {}
    viser_joint_indices: dict[str, tuple[int, ...]] = {}

    x_offset = 0.0
    for side in sides:
        server.scene.add_frame(f"/{side}_keypoints", axes_length=0.02, axes_radius=0.001, position=(x_offset, 0.0, 0.0))
        if retargeter is not None:
            root_name = f"/{side}_hand"
            server.scene.add_frame(root_name, axes_length=0.03, axes_radius=0.001, position=(x_offset + 0.3, 0.0, 0.0))
            viser_urdf, indices = add_hand_urdf(server, HANDS["sharpa"], root_name)
            viser_urdfs[side] = viser_urdf
            viser_joint_indices[side] = indices
        x_offset += 0.8

    primary_side = sides[0]
    n_frames = len(data[f"{primary_side}_timestamps"])
    frame_slider = server.gui.add_slider("Frame", min=0, max=n_frames - 1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("Playing", initial_value=True)

    print(f"Viser: {server.request_share_url()}")
    print("Looping playback at 1x speed. Press Ctrl+C to stop.\n")

    frame_idx = 0
    last_time = time.monotonic()
    while True:
        now = time.monotonic()
        dt = now - last_time
        last_time = now
        timestamps = data[f"{primary_side}_timestamps"]
        if playing.value:
            frame_idx = int(np.searchsorted(timestamps, timestamps[frame_idx] + dt * 0.5))
            if frame_idx >= n_frames:
                frame_idx = 0
            frame_slider.value = frame_idx
        else:
            frame_idx = int(frame_slider.value)

        for side in sides:
            side_idx = min(frame_idx, len(data[f"{side}_timestamps"]) - 1)
            positions = data[f"{side}_positions"][side_idx]
            rotations = data[f"{side}_rotations"][side_idx]
            wrist_local = manus_positions_to_fixed_wrist_local(positions, rotations)
            wrist_local = apply_glove_corrections(
                wrist_local,
                glove_calib["thumb_rotation"],
                glove_calib["thumb_translation"],
                glove_calib["finger_scales"],
            )
            with server.atomic():
                draw_manus_skeleton(server, wrist_local, f"/{side}_keypoints")

            if retargeter is not None and side in viser_urdfs:
                mano_kps = manus_to_mano_keypoints(positions, rotations, glove_calib)
                joint_angles = retargeter.retarget(mano_kps)
                joint_cfg = np.asarray([joint_angles[index] for index in viser_joint_indices[side]], dtype=np.float32)
                viser_urdfs[side].update_cfg(joint_cfg)

        time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
