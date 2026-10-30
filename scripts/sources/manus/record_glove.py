"""Record Manus glove frames to an npz file."""

import argparse
import threading
import time
from pathlib import Path

import numpy as np

from retargetingkit.assets import calib
from retargetingkit.sources.glove._manus_sdk import ManusClient, resolve_manus_sdk_library_path
from retargetingkit.sources.glove.manus_conversion import (
    apply_glove_corrections,
    load_glove_calibration,
    manus_positions_to_fixed_wrist_local,
)
from retargetingkit.sources.glove.manus_glove import (
    DualManusGloveAggregator,
    parse_glove_id,
)
from retargetingkit.utils.viz import draw_manus_skeleton


DEFAULT_OUTPUT_PATH = Path("recording.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Manus glove data")
    parser.add_argument("--hand", choices=["left", "right", "both"], default="right")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--no-viz", action="store_true", help="Disable live viser preview")
    parser.add_argument("--manus-left-id", type=str, default=None)
    parser.add_argument("--manus-right-id", type=str, default=None)
    parser.add_argument("--manus-calib", type=Path)
    parser.add_argument("--frames", type=int, default=0, help="Continuously capture this many frames; 0 is interactive")
    parser.add_argument("--rate", type=float, default=30.0, help="Continuous capture rate in Hz")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    library_path = resolve_manus_sdk_library_path()
    print(f"Loading Manus client from: {library_path}")
    client = ManusClient(library_path)
    aggregator = DualManusGloveAggregator(
        left_override=parse_glove_id(args.manus_left_id),
        right_override=parse_glove_id(args.manus_right_id),
    )
    client.start(aggregator.update)
    print("Waiting for glove data...")

    sides = ["left", "right"] if args.hand == "both" else [args.hand]
    while True:
        left_frame, right_frame = aggregator.latest()
        all_ready = all((left_frame if side == "left" else right_frame) is not None for side in sides)
        if all_ready:
            for side in sides:
                frame = left_frame if side == "left" else right_frame
                assert frame is not None
                print(f"  {side} glove detected: id=0x{frame.glove_id:08X}")
            break
        time.sleep(0.1)

    calib_path = calib.get("manus_calib") if args.manus_calib is None else args.manus_calib
    glove_calib = load_glove_calibration(calib_path)
    print(f"Loaded glove calibration from: {calib_path}")

    server = None
    viz_stop = threading.Event()
    viz_thread = None
    if not args.no_viz:
        import viser

        server = viser.ViserServer()
        server.scene.add_grid("/grid", 2.0, 2.0)
        for side in sides:
            server.scene.add_frame(f"/{side}_hand", axes_length=0.02, axes_radius=0.001)
        print(f"Viser: {server.request_share_url()}")

        def viz_loop() -> None:
            assert server is not None
            while not viz_stop.is_set():
                left_f, right_f = aggregator.latest()
                for side in sides:
                    frame = left_f if side == "left" else right_f
                    if frame is None:
                        continue
                    wrist_local = manus_positions_to_fixed_wrist_local(frame.positions, frame.rotations)
                    wrist_local = apply_glove_corrections(
                        wrist_local,
                        glove_calib["thumb_rotation"],
                        glove_calib["thumb_translation"],
                        glove_calib["finger_scales"],
                    )
                    with server.atomic():
                        draw_manus_skeleton(server, wrist_local, f"/{side}_hand")
                time.sleep(1.0 / 30.0)

        viz_thread = threading.Thread(target=viz_loop, daemon=True)
        viz_thread.start()

    buffers: dict[str, dict[str, list[int] | list[float] | list[np.ndarray]]] = {
        side: {
            "timestamps": [],
            "publish_time": [],
            "recv_ns": [],
            "glove_id": [],
            "side_code": [],
            "positions": [],
            "rotations": [],
        }
        for side in sides
    }
    if args.frames:
        print(f"\nRecording {args.frames} {args.hand} frame(s) at {args.rate:g} Hz.\n")
    else:
        print(f"\nRecording {args.hand} hand(s), one frame per Enter press.")
        print("Type q then Enter to stop and save.\n")
    start_time = time.monotonic()
    frame_index = 0
    while args.frames == 0 or frame_index < args.frames:
        if args.frames == 0:
            response = input(f"  [frame {frame_index}] Press Enter to capture, q to stop...")
            if response.strip().lower() in ("q", "quit", "done"):
                break
        left_frame, right_frame = aggregator.latest()
        t = time.monotonic() - start_time
        for side in sides:
            frame = left_frame if side == "left" else right_frame
            assert frame is not None
            buffers[side]["timestamps"].append(t)
            buffers[side]["publish_time"].append(frame.publish_time)
            buffers[side]["recv_ns"].append(frame.recv_ns)
            buffers[side]["glove_id"].append(frame.glove_id)
            buffers[side]["side_code"].append(frame.side_code)
            buffers[side]["positions"].append(frame.positions.copy())
            buffers[side]["rotations"].append(frame.rotations.copy())
        frame_index += 1
        print(f"    captured frame {frame_index}")
        if args.frames:
            time.sleep(1.0 / args.rate)

    if viz_thread is not None:
        viz_stop.set()
        viz_thread.join()

    save_dict: dict[str, np.ndarray] = {}
    for side in sides:
        buf = buffers[side]
        n = len(buf["timestamps"])
        assert n > 0
        save_dict[f"{side}_timestamps"] = np.array(buf["timestamps"], dtype=np.float64)
        save_dict[f"{side}_publish_time"] = np.array(buf["publish_time"], dtype=np.uint64)
        save_dict[f"{side}_recv_ns"] = np.array(buf["recv_ns"], dtype=np.int64)
        save_dict[f"{side}_glove_id"] = np.array(buf["glove_id"], dtype=np.uint32)
        save_dict[f"{side}_side_code"] = np.array(buf["side_code"], dtype=np.int32)
        save_dict[f"{side}_positions"] = np.stack(buf["positions"]).astype(np.float32)
        save_dict[f"{side}_rotations"] = np.stack(buf["rotations"]).astype(np.float32)
        print(f"  {side}: {n} frames")

    client.stop()
    np.savez(args.output, **save_dict)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
