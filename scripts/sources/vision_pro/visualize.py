"""Visualize Vision Pro hand keypoints in viser. Ctrl-C to stop."""

import argparse
import time

import numpy as np
import viser
from viser import transforms as tf

from retargetingkit.sources.vr.vision_pro import VISION_PRO_HAND_EDGES, VisionPro


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream Vision Pro tracking data")
    parser.add_argument("--ip", type=str)
    parser.add_argument("--recording", type=str, default=None, help="Pickle file for playback")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    args = parser.parse_args()

    vp = VisionPro(ip=args.ip, recording_path=args.recording)

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/grid", width=2.0, height=2.0)

    bone_starts = np.array([b[0] for b in VISION_PRO_HAND_EDGES])
    bone_ends = np.array([b[1] for b in VISION_PRO_HAND_EDGES])

    left_pts = server.scene.add_point_cloud(
        "/left_hand/joints",
        np.zeros((25, 3), dtype=np.float32),
        colors=(0, 120, 255),
        point_size=0.008,
        point_shape="circle",
    )
    right_pts = server.scene.add_point_cloud(
        "/right_hand/joints",
        np.zeros((25, 3), dtype=np.float32),
        colors=(255, 120, 0),
        point_size=0.008,
        point_shape="circle",
    )
    left_bones = server.scene.add_line_segments(
        "/left_hand/bones",
        np.zeros((len(VISION_PRO_HAND_EDGES), 2, 3), dtype=np.float32),
        colors=(0, 120, 255),
        thickness=2.0,
        thickness_units="screen",
    )
    right_bones = server.scene.add_line_segments(
        "/right_hand/bones",
        np.zeros((len(VISION_PRO_HAND_EDGES), 2, 3), dtype=np.float32),
        colors=(255, 120, 0),
        thickness=2.0,
        thickness_units="screen",
    )
    head_frame = server.scene.add_frame("/head", axes_length=0.08, axes_radius=0.004)

    print(f"Viser ready — open http://localhost:{args.port}")

    while True:
        frame = vp.get_all()
        lw = frame["left_wrist"][0]
        rw = frame["right_wrist"][0]
        lf = (lw @ frame["left_fingers"])[:, :3, 3]
        rf = (rw @ frame["right_fingers"])[:, :3, 3]
        head = frame["head"]

        left_pts.points = lf.astype(np.float32)
        right_pts.points = rf.astype(np.float32)

        left_bones.points = np.stack([lf[bone_starts], lf[bone_ends]], axis=1).astype(np.float32)
        right_bones.points = np.stack([rf[bone_starts], rf[bone_ends]], axis=1).astype(np.float32)

        head_mat = head[0]
        head_frame.position = head_mat[:3, 3]
        head_frame.wxyz = tf.SO3.from_matrix(head_mat[:3, :3]).wxyz

        lp = frame["left_pinch_distance"]
        rp = frame["right_pinch_distance"]
        pos = head_mat[:3, 3]
        print(
            f"\rhead: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]  pinch L={lp:.3f} R={rp:.3f}\033[K",
            end="",
            flush=True,
        )
        time.sleep(0.05)


if __name__ == "__main__":
    main()
