"""Visualize Manus glove keypoints in viser.

Example:
    uv run --extra manus scripts/sources/manus/visualize_keypoints.py --side both
"""

import argparse
import signal
import sys
import time

import numpy as np
import viser
from manus import Manus

from retargetingkit.sources.glove.manus_glove import parse_glove_id


MANUS_BONES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (0, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (0, 15),
    (15, 16),
    (16, 17),
    (17, 18),
    (18, 19),
    (0, 20),
    (20, 21),
    (21, 22),
    (22, 23),
    (23, 24),
)
FINGER_COLORS = np.array(
    [(220, 220, 220)]
    + [(255, 100, 100)] * 4
    + [(255, 180, 50)] * 5
    + [(255, 255, 80)] * 5
    + [(180, 255, 90)] * 5
    + [(100, 210, 120)] * 5,
    dtype=np.uint8,
)


def run(args: argparse.Namespace) -> None:
    """Stream selected Manus hands into a viser scene.

    >>> run.__doc__ is not None
    True
    """
    sides = ("left", "right") if args.side == "both" else (args.side,)

    if args.check_config:
        print(f"side={args.side}")
        print(f"viser_port={args.viser_port}")
        print(f"rate={args.rate}")
        return

    manus = Manus(
        left_glove_id=parse_glove_id(args.manus_left_id),
        right_glove_id=parse_glove_id(args.manus_right_id),
    )
    server = viser.ViserServer(port=args.viser_port)
    server.scene.add_grid("/grid", width=2.0, height=2.0)

    bone_starts = np.asarray([start for start, _ in MANUS_BONES], dtype=np.int64)
    bone_ends = np.asarray([end for _, end in MANUS_BONES], dtype=np.int64)
    offsets = {"left": (-0.25, 0.0, 0.0), "right": (0.25, 0.0, 0.0)}
    if args.side != "both":
        offsets[args.side] = (0.0, 0.0, 0.0)

    point_handles = {}
    bone_handles = {}
    label_handles = {}
    zeros = np.zeros((25, 3), dtype=np.float32)
    zero_bones = np.zeros((len(MANUS_BONES), 2, 3), dtype=np.float32)
    for side in sides:
        root = f"/{side}_manus"
        server.scene.add_frame(root, show_axes=True, axes_length=0.04, axes_radius=0.0015, position=offsets[side])
        point_handles[side] = server.scene.add_point_cloud(
            f"{root}/keypoints",
            zeros,
            colors=FINGER_COLORS,
            point_size=0.004,
            point_shape="circle",
            visible=False,
        )
        bone_handles[side] = server.scene.add_line_segments(
            f"{root}/bones",
            zero_bones,
            colors=(210, 210, 210),
            thickness=2.5,
            thickness_units="screen",
            visible=False,
        )
        label_handles[side] = []
        for keypoint_index in range(25):
            label = server.scene.add_label(
                f"{root}/label_{keypoint_index}",
                str(keypoint_index),
                position=(0.0, 0.0, 0.0),
                visible=False,
            )
            label_handles[side].append(label)

    def _stop(_signum: int, _frame: object) -> None:
        manus.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    manus.start()
    print(f"Viser ready: http://localhost:{args.viser_port}")
    print(f"Streaming Manus keypoints for: {', '.join(sides)}")

    dt = 1.0 / args.rate
    while True:
        loop_start = time.monotonic()
        for side in sides:
            points = manus.latest_wrist_local(side)
            if points is not None:
                point_handles[side].points = points
                point_handles[side].visible = True
                bone_handles[side].points = np.stack([points[bone_starts], points[bone_ends]], axis=1)
                bone_handles[side].visible = True
                if args.show_labels:
                    for keypoint_index, label in enumerate(label_handles[side]):
                        label.position = tuple(points[keypoint_index].tolist())
                        label.visible = True

        sleep_time = dt - (time.monotonic() - loop_start)
        if sleep_time > 0:
            time.sleep(sleep_time)


def main() -> None:
    """Parse CLI args and run the Manus keypoint visualizer.

    >>> main.__doc__ is not None
    True
    """
    parser = argparse.ArgumentParser(description="Visualize Manus glove keypoints in viser")
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--manus-left-id", type=str, default=None)
    parser.add_argument("--manus-right-id", type=str, default=None)
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--show-labels", action="store_true")
    parser.add_argument("--check-config", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
