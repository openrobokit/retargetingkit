"""Record continuous right-hand Manus motion as unsupervised data.

Usage:
    uv run --extra manus scripts/teleop/neural_hand/record_motion.py --output motion.npz
"""

import argparse
import threading
import time
from pathlib import Path

import numpy as np

from retargetingkit.sources.glove._manus_sdk import ManusClient, resolve_manus_sdk_library_path
from retargetingkit.sources.glove.manus_conversion import manus_positions_to_fixed_wrist_local
from retargetingkit.sources.glove.manus_glove import (
    DualManusGloveAggregator,
    parse_glove_id,
)
from retargetingkit.utils.viz import draw_manus_skeleton


def main() -> None:
    parser = argparse.ArgumentParser(description="Record continuous right-hand Manus motion")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--manus-right-id")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    client = ManusClient(resolve_manus_sdk_library_path())
    aggregator = DualManusGloveAggregator(right_override=parse_glove_id(args.manus_right_id))
    client.start(aggregator.update)
    print("Waiting for right glove data...")
    while aggregator.latest()[1] is None:
        time.sleep(0.1)

    stop = threading.Event()
    server = None
    if not args.no_viz:
        import viser

        server = viser.ViserServer()
        server.scene.add_grid("/grid", 2.0, 2.0)
        server.scene.add_frame("/right_hand", axes_length=0.02, axes_radius=0.001)
        print(f"viser={server.get_port()}")

        def viz_loop() -> None:
            assert server is not None
            while not stop.is_set():
                frame = aggregator.latest()[1]
                assert frame is not None
                wrist_local = manus_positions_to_fixed_wrist_local(frame.positions, frame.rotations)
                with server.atomic():
                    draw_manus_skeleton(server, wrist_local, "/right_hand")
                time.sleep(1.0 / 30.0)

        threading.Thread(target=viz_loop, daemon=True).start()

    timestamps: list[float] = []
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []

    def record_loop() -> None:
        start = time.monotonic()
        period = 1.0 / args.rate
        while not stop.is_set():
            frame = aggregator.latest()[1]
            assert frame is not None
            timestamps.append(time.monotonic() - start)
            positions.append(frame.positions.copy())
            rotations.append(frame.rotations.copy())
            time.sleep(period)

    input("Press Enter to START recording...")
    record_thread = threading.Thread(target=record_loop, daemon=True)
    record_thread.start()
    input("Press Enter to STOP and save...")
    stop.set()
    record_thread.join()

    assert timestamps
    np.savez(
        args.output,
        right_timestamps=np.asarray(timestamps, dtype=np.float64),
        right_positions=np.stack(positions).astype(np.float32),
        right_rotations=np.stack(rotations).astype(np.float32),
    )
    print(f"saved={args.output} frames={len(timestamps)} duration={timestamps[-1]:.1f}s")


if __name__ == "__main__":
    main()
