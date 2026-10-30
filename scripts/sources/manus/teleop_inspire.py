"""Manus glove -> Inspire hand retargeting with Manus + Inspire visualization.

Run visual-only:
    uv run --extra manus scripts/sources/manus/teleop_inspire.py --side left

Run with G1 Inspire ZMQ server:
    uv run --extra manus scripts/sources/manus/teleop_inspire.py --side both
"""

import argparse
import json
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp
import zmq
from manus import Manus

from retargetingkit.sources.glove.manus_glove import parse_glove_id
from robokit.assets.robots.end_effectors import inspire_hand
from robokit.helpers.hand_retargeting import HandRetargeterOnline
from robokit.helpers.hand_retargeting.presets import inspire
from robokit.robo import Robot


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


def get_local_inspire_urdf_path(side: str) -> Path:
    """Return the repo-local Inspire hand URDF path."""
    return inspire_hand.URDF_PATH_RIGHT if side == "right" else inspire_hand.URDF_PATH_LEFT


def run(args: argparse.Namespace) -> None:
    """Run Manus -> Inspire retargeting."""
    sides = ("left", "right") if args.side == "both" else (args.side,)
    print(f"side={args.side}")
    print(f"viser={not args.no_visualize}")
    print(f"zmq=tcp://{args.zmq_host}:{args.zmq_port} topic={args.topic}")
    print(f"rate={args.rate}")
    for side in sides:
        print(f"{side}_urdf={get_local_inspire_urdf_path(side)}")
    if args.check_config:
        return

    from retargetingkit.sources.glove.manus_conversion import hand_pose_from_glove

    wp.init()
    manus = Manus(
        left_glove_id=parse_glove_id(args.manus_left_id),
        right_glove_id=parse_glove_id(args.manus_right_id),
    )

    retargeters = {}
    for side in sides:
        is_right = side == "right"
        collision_path = inspire_hand.COLLISION_SPHERE_PATH_RIGHT if is_right else None
        robot = Robot.load(
            get_local_inspire_urdf_path(side),
            load_collision_spheres=collision_path is not None,
            collision_spheres_path=collision_path,
            self_collision_ignore_path=inspire_hand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT if is_right else None,
        )
        retargeter = HandRetargeterOnline(
            config=inspire.online if is_right else inspire.online_left,
            robot=robot,
            spec=replace(inspire.spec if is_right else inspire.spec_left, floating_base=False),
        )
        retargeter.warmup(1)
        retargeters[side] = retargeter

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://{args.zmq_host}:{args.zmq_port}")
    topic_bytes = args.topic.encode()
    print(f"ZMQ PUB bound tcp://{args.zmq_host}:{args.zmq_port} [{args.topic}]", flush=True)

    server = None
    point_handles = {}
    bone_handles = {}
    label_handles = {}
    inspire_handles = {}
    if not args.no_visualize:
        import viser
        from viser.extras import ViserUrdf

        server = viser.ViserServer(port=args.viser_port)
        server.scene.add_grid("/grid", width=2.0, height=2.0)
        bone_starts = np.asarray([start for start, _ in MANUS_BONES], dtype=np.int64)
        bone_ends = np.asarray([end for _, end in MANUS_BONES], dtype=np.int64)
        zeros = np.zeros((25, 3), dtype=np.float32)
        zero_bones = np.zeros((len(MANUS_BONES), 2, 3), dtype=np.float32)
        side_offsets = {"left": (-0.35, 0.0, 0.0), "right": (0.35, 0.0, 0.0)}
        if args.side != "both":
            side_offsets[args.side] = (0.0, 0.0, 0.0)
        for side in sides:
            manus_root = f"/{side}_manus"
            inspire_root = f"/{side}_inspire"
            server.scene.add_frame(
                manus_root,
                show_axes=True,
                axes_length=0.04,
                axes_radius=0.0015,
                position=side_offsets[side],
            )
            server.scene.add_frame(
                inspire_root,
                show_axes=False,
                position=(side_offsets[side][0], side_offsets[side][1] + 0.24, side_offsets[side][2]),
            )
            point_handles[side] = server.scene.add_point_cloud(
                f"{manus_root}/keypoints",
                zeros,
                colors=FINGER_COLORS,
                point_size=args.point_size,
                point_shape="circle",
                visible=False,
            )
            bone_handles[side] = {
                "handle": server.scene.add_line_segments(
                    f"{manus_root}/bones",
                    zero_bones,
                    colors=(210, 210, 210),
                    thickness=2.0,
                    thickness_units="screen",
                    visible=False,
                ),
                "starts": bone_starts,
                "ends": bone_ends,
            }
            label_handles[side] = []
            for keypoint_index in range(25):
                label = server.scene.add_label(
                    f"{manus_root}/label_{keypoint_index}",
                    str(keypoint_index),
                    position=(0.0, 0.0, 0.0),
                    visible=False,
                )
                label_handles[side].append(label)
            inspire_handles[side] = ViserUrdf(
                server,
                get_local_inspire_urdf_path(side),
                root_node_name=f"{inspire_root}/urdf",
            )
            inspire_handles[side].update_cfg(np.zeros(6))
        print(f"Viser ready: http://localhost:{args.viser_port}", flush=True)

    def _stop(_signum: int, _frame: object) -> None:
        manus.stop()
        pub.close()
        ctx.term()
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    manus.start()
    print("Waiting for Manus glove data...", flush=True)
    last_status_time = 0.0
    for side in sides:
        while manus.latest(side) is None:
            now = time.monotonic()
            if now - last_status_time >= 1.0:
                left_frame, right_frame = manus.aggregator.latest()
                left_text = (
                    "none" if left_frame is None else f"0x{left_frame.glove_id:08X}/side_code={left_frame.side_code}"
                )
                right_text = (
                    "none" if right_frame is None else f"0x{right_frame.glove_id:08X}/side_code={right_frame.side_code}"
                )
                print(
                    f"waiting_for={side} seen_gloves={manus.aggregator.glove_ids()} left={left_text} right={right_text}",
                    flush=True,
                )
                last_status_time = now
            time.sleep(0.01)

    print("Retargeting running. Press Ctrl-C to stop.", flush=True)

    dt = 1.0 / args.rate
    left_angles = np.zeros(6)
    right_angles = np.zeros(6)
    while True:
        loop_start = time.monotonic()
        for side in sides:
            frame = manus.latest(side)
            if frame is not None:
                hand_pose = hand_pose_from_glove(frame.positions, frame.rotations)
                angles = retargeters[side].solve_numpy(hand_pose.positions)[0, 7:]
                if side == "left":
                    left_angles = angles
                else:
                    right_angles = angles

                if server is not None:
                    points = manus.latest_wrist_local(side)
                    if points is not None:
                        point_handles[side].points = points
                        point_handles[side].visible = True
                        bone_info = bone_handles[side]
                        bone_info["handle"].points = np.stack(
                            [points[bone_info["starts"]], points[bone_info["ends"]]], axis=1
                        )
                        bone_info["handle"].visible = True
                        if args.show_labels:
                            for keypoint_index, label in enumerate(label_handles[side]):
                                label.position = tuple(points[keypoint_index].tolist())
                                label.visible = True
                    inspire_handles[side].update_cfg(angles)

        payload = json.dumps(
            {
                "positions": np.concatenate([left_angles, right_angles]).tolist(),
                "timestamp": time.time(),
            }
        )
        pub.send_multipart([topic_bytes, payload.encode()])

        sleep_time = dt - (time.monotonic() - loop_start)
        if sleep_time > 0:
            time.sleep(sleep_time)


def main() -> None:
    """Parse CLI args and run Manus -> Inspire retargeting."""
    parser = argparse.ArgumentParser(description="Manus glove -> Inspire hand retargeting")
    parser.add_argument("--side", choices=("left", "right", "both"), default="right")
    parser.add_argument("--manus-left-id", type=str, default=None)
    parser.add_argument("--manus-right-id", type=str, default=None)
    parser.add_argument("--zmq-host", type=str, default="0.0.0.0")
    parser.add_argument("--zmq-port", type=int, default=5560)
    parser.add_argument("--topic", type=str, default="hand")
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--point-size", type=float, default=0.004)
    parser.add_argument("--show-labels", action="store_true")
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--check-config", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
