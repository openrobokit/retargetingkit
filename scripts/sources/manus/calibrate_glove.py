"""Calibrate Manus glove thumb correction and per-finger scales."""

import argparse
import threading
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from retargetingkit.sources.glove._manus_sdk import ManusClient, resolve_manus_sdk_library_path
from retargetingkit.sources.glove.manus_conversion import (
    FINGER_NAMES,
    MANUS_TIP_INDICES,
    apply_glove_corrections,
    manus_positions_to_fixed_wrist_local,
)
from retargetingkit.sources.glove.manus_glove import DualManusGloveAggregator, parse_glove_id
from retargetingkit.utils.viz import draw_manus_skeleton


CALIBRATION_POSES: list[dict[str, object]] = [
    {"name": "flat_hand", "description": "Hold hand flat and open", "type": "flat"},
    {
        "name": "pinch_index",
        "description": "Pinch: thumb tip to index fingertip",
        "type": "contact",
        "thumb_joint": 4,
        "target_joint": 9,
        "weight": 4.0,
    },
    {
        "name": "pinch_middle",
        "description": "Pinch: thumb tip to middle fingertip",
        "type": "contact",
        "thumb_joint": 4,
        "target_joint": 14,
        "weight": 4.0,
    },
    {
        "name": "pinch_ring",
        "description": "Pinch: thumb tip to ring fingertip",
        "type": "contact",
        "thumb_joint": 4,
        "target_joint": 19,
        "weight": 4.0,
    },
    {
        "name": "pinch_pinky",
        "description": "Pinch: thumb tip to pinky fingertip",
        "type": "contact",
        "thumb_joint": 4,
        "target_joint": 24,
        "weight": 4.0,
    },
]


def collect_frames(
    aggregator: DualManusGloveAggregator, side: str, num_frames: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    all_positions: list[np.ndarray] = []
    all_rotations: list[np.ndarray] = []
    for _ in range(num_frames):
        left_frame, right_frame = aggregator.latest()
        frame = left_frame if side == "left" else right_frame
        assert frame is not None, f"No glove data for {side} hand"
        all_positions.append(frame.positions.copy())
        all_rotations.append(frame.rotations.copy())
        time.sleep(0.033)
    avg_positions = np.mean(all_positions, axis=0).astype(np.float32)
    avg_rotations = np.mean(all_rotations, axis=0).astype(np.float32)
    avg_rotations /= np.maximum(np.linalg.norm(avg_rotations, axis=1, keepdims=True), 1e-8)
    return avg_positions, avg_rotations


def optimize_glove_corrections(contact_data: list[dict[str, object]]) -> dict[str, object]:
    def loss_function(params: np.ndarray) -> float:
        thumb_rot = params[0:3]
        thumb_trans = params[3:6]
        finger_scales = np.exp(params[6:11])
        total = 0.0
        for datum in contact_data:
            wrist_local = np.asarray(datum["wrist_local"], dtype=np.float64)
            thumb_joint = int(datum["thumb_joint"])
            target_joint = int(datum["target_joint"])
            weight = float(datum["weight"])
            corrected = apply_glove_corrections(wrist_local, thumb_rot, thumb_trans, finger_scales)
            total += weight * np.sum((corrected[thumb_joint] - corrected[target_joint]) ** 2)
        total += 0.01 * (
            5.0 * np.sum(params[0:3] ** 2) + 500.0 * np.sum(params[3:6] ** 2) + 20.0 * np.sum(params[6:11] ** 2)
        )
        return float(total)

    result = minimize(
        loss_function,
        np.zeros(11, dtype=np.float64),
        method="Powell",
        options={"maxiter": 10000, "ftol": 1e-12},
    )
    return {
        "thumb_rotation": result.x[0:3].astype(np.float32),
        "thumb_translation": result.x[3:6].astype(np.float32),
        "finger_scales": np.exp(result.x[6:11]).astype(np.float32),
        "loss": float(result.fun),
        "iterations": int(result.nit),
        "success": bool(result.success),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Manus glove calibration")
    parser.add_argument("--hand", choices=["left", "right"], default="right")
    parser.add_argument("--manus-left-id", type=str, default=None)
    parser.add_argument("--manus-right-id", type=str, default=None)
    parser.add_argument("--name", type=str, default="manus_glove_calib", help="Calibration file name")
    parser.add_argument("--output", type=str, default=None, help="Full output path")
    parser.add_argument("--num-frames", type=int, default=30)
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else Path(f"{args.name}.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    library_path = resolve_manus_sdk_library_path()
    print(f"Loading Manus client from: {library_path}")
    client = ManusClient(library_path)
    aggregator = DualManusGloveAggregator(
        left_override=parse_glove_id(args.manus_left_id),
        right_override=parse_glove_id(args.manus_right_id),
    )
    client.start(aggregator.update)
    print("Waiting for glove data...")

    while True:
        left_frame, right_frame = aggregator.latest()
        frame = left_frame if args.hand == "left" else right_frame
        if frame is not None:
            print(f"Glove detected: id=0x{frame.glove_id:08X}")
            break
        time.sleep(0.1)

    import viser

    server = viser.ViserServer()
    server.scene.add_grid("/grid", 2.0, 2.0)
    server.scene.add_frame("/raw_hand", axes_length=0.02, axes_radius=0.001)
    print(f"Viser: {server.request_share_url()}")

    preview_active = threading.Event()
    preview_active.set()

    def preview_loop() -> None:
        while preview_active.is_set():
            left, right = aggregator.latest()
            glove_frame = left if args.hand == "left" else right
            if glove_frame is not None:
                wrist_local = manus_positions_to_fixed_wrist_local(glove_frame.positions, glove_frame.rotations)
                with server.atomic():
                    draw_manus_skeleton(server, wrist_local, "/raw_hand")
            time.sleep(0.033)

    threading.Thread(target=preview_loop, daemon=True).start()

    collected_wrist_locals: dict[str, np.ndarray] = {}
    for pose_idx, pose in enumerate(CALIBRATION_POSES):
        print(f"\n{'=' * 60}")
        print(f"[{pose_idx + 1}/{len(CALIBRATION_POSES)}] {pose['description']}")
        print("Hold the pose steady, check viser, press Enter to capture...")
        input()
        print(f"Capturing {args.num_frames} frames...")
        positions, rotations = collect_frames(aggregator, args.hand, num_frames=args.num_frames)
        wrist_local = manus_positions_to_fixed_wrist_local(positions, rotations)
        collected_wrist_locals[str(pose["name"])] = wrist_local
        if pose["type"] == "contact":
            thumb_joint = int(pose["thumb_joint"])
            target_joint = int(pose["target_joint"])
            dist_mm = np.linalg.norm(wrist_local[thumb_joint] - wrist_local[target_joint]) * 1000
            print(f"  Contact distance: {dist_mm:.1f} mm")
        else:
            for finger_idx, finger_name in enumerate(FINGER_NAMES):
                tip_dist_mm = np.linalg.norm(wrist_local[MANUS_TIP_INDICES[finger_idx]] - wrist_local[0]) * 1000
                print(f"  {finger_name} tip: {tip_dist_mm:.1f} mm from wrist")

    preview_active.clear()
    print(f"\n{'=' * 60}")
    print("Optimizing glove corrections...")

    contact_data = [
        {
            "wrist_local": collected_wrist_locals[str(pose["name"])],
            "thumb_joint": pose["thumb_joint"],
            "target_joint": pose["target_joint"],
            "weight": pose["weight"],
            "name": pose["name"],
        }
        for pose in CALIBRATION_POSES
        if pose["type"] == "contact"
    ]

    for datum in contact_data:
        wrist_local = np.asarray(datum["wrist_local"], dtype=np.float32)
        raw_dist = (
            np.linalg.norm(wrist_local[int(datum["thumb_joint"])] - wrist_local[int(datum["target_joint"])]) * 1000
        )
        print(f"  {datum['name']}: {raw_dist:.1f} mm (raw)")

    optimized = optimize_glove_corrections(contact_data)
    print(f"\nOptimization {'succeeded' if optimized['success'] else 'completed'}: loss={optimized['loss']:.8f}")

    for datum in contact_data:
        wrist_local = np.asarray(datum["wrist_local"], dtype=np.float32)
        corrected = apply_glove_corrections(
            wrist_local,
            np.asarray(optimized["thumb_rotation"], dtype=np.float32),
            np.asarray(optimized["thumb_translation"], dtype=np.float32),
            np.asarray(optimized["finger_scales"], dtype=np.float32),
        )
        corr_dist = np.linalg.norm(corrected[int(datum["thumb_joint"])] - corrected[int(datum["target_joint"])]) * 1000
        print(f"  {datum['name']}: {corr_dist:.1f} mm (corrected)")

    print(f"\nFinger scales: {optimized['finger_scales']}")
    save_dict = {
        "thumb_rotation": np.asarray(optimized["thumb_rotation"], dtype=np.float32),
        "thumb_translation": np.asarray(optimized["thumb_translation"], dtype=np.float32),
        "finger_scales": np.asarray(optimized["finger_scales"], dtype=np.float32),
    }
    for pose in CALIBRATION_POSES:
        save_dict[f"wrist_local_{pose['name']}"] = collected_wrist_locals[str(pose["name"])]
    np.savez(output_path, **save_dict)
    print(f"\nCalibration saved to: {output_path}")

    server.scene.add_frame("/corrected_hand", axes_length=0.02, axes_radius=0.001, position=(0.3, 0.0, 0.0))
    server.scene.add_label("/raw_label", "Raw", position=(0.0, 0.0, 0.15))
    server.scene.add_label("/corrected_label", "Calibrated", position=(0.3, 0.0, 0.15))
    print(f"\n{'=' * 60}")
    print("Live comparison mode. Press Ctrl+C to exit.\n")

    last_print_time = time.monotonic()
    while True:
        left_frame, right_frame = aggregator.latest()
        frame = left_frame if args.hand == "left" else right_frame
        if frame is None:
            time.sleep(0.01)
            continue
        wrist_local = manus_positions_to_fixed_wrist_local(frame.positions, frame.rotations)
        corrected = apply_glove_corrections(
            wrist_local,
            np.asarray(optimized["thumb_rotation"], dtype=np.float32),
            np.asarray(optimized["thumb_translation"], dtype=np.float32),
            np.asarray(optimized["finger_scales"], dtype=np.float32),
        )
        with server.atomic():
            draw_manus_skeleton(server, wrist_local, "/raw_hand")
            draw_manus_skeleton(server, corrected, "/corrected_hand")
        now = time.monotonic()
        if now - last_print_time >= 3.0:
            pinch_lines = []
            for finger_idx in range(1, 5):
                tip = MANUS_TIP_INDICES[finger_idx]
                raw_dist = np.linalg.norm(wrist_local[4] - wrist_local[tip]) * 1000
                corr_dist = np.linalg.norm(corrected[4] - corrected[tip]) * 1000
                pinch_lines.append(f"{FINGER_NAMES[finger_idx]}: raw={raw_dist:.0f}mm corr={corr_dist:.0f}mm")
            print("  " + "  |  ".join(pinch_lines))
            last_print_time = now
        time.sleep(0.016)


if __name__ == "__main__":
    main()
