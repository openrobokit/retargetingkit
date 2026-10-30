"""Floating-hand DexYCB online retargeting using the same data and solver as the gallery.

Usage: ``uv run python -m retargetingkit.examples.hand_optimization.06_dexycb_online``.
"""

import argparse
import time
from pathlib import Path
from typing import cast

import numpy as np
import trimesh
import viser
import warp as wp
from viser.extras import ViserUrdf

from robokit.assets.body_models import mano
from robokit.assets.datasets import dexycb
from robokit.gallery.hand_retargeting import MOTION_CAMERA, PRESETS, WITH_WRIST
from robokit.helpers.hand_retargeting import HandRetargeterOnline
from robokit.helpers.hand_retargeting.dexycb_utils import load_dexycb_keypoints
from robokit.robo import Robot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ee", choices=tuple(PRESETS), default="paxini")
    parser.add_argument("--capture-index", type=int, default=0)
    parser.add_argument("--mano-root", default="")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--loops", type=int, default=0, help="Stop after this many loops; 0 plays continuously")
    args = parser.parse_args()

    wp.init()
    preset = PRESETS[args.ee]
    robot = Robot.load(
        str(preset.urdf),
        load_collision_spheres=preset.spheres is not None,
        collision_spheres_path=preset.spheres,
        self_collision_ignore_path=preset.ignore,
    )
    assert preset.online_wrist is not None
    solver = HandRetargeterOnline(preset.online_wrist, robot, preset.spec, device="cuda:0")
    solver.warmup(1)
    solver.reset()
    data = load_dexycb_keypoints(
        dexycb_dir=dexycb.DIR,
        mano_root=Path(args.mano_root) if args.mano_root else mano.DIR,
        capture_index=args.capture_index,
        is_right=True,
        device="cuda:0",
    )
    points = data["keypoints"].astype(np.float32)
    root_index = preset.spec.target_names.index(preset.spec.root_target_name)

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.set_up_direction("+z")
    server.initial_camera.position, server.initial_camera.look_at = MOTION_CAMERA[WITH_WRIST]
    frame = server.scene.add_frame("/robot", show_axes=False)
    urdf = ViserUrdf(server, preset.urdf, root_node_name="/robot/urdf")
    hand = server.scene.add_mesh_simple(
        "/target_mano",
        data["mano_vertices"][0],
        data["mano_faces"],
        color=(230, 190, 150),
        opacity=0.35,
        side="double",
    )
    mesh = cast(trimesh.Trimesh, trimesh.load(data["object_mesh_path"], process=False, force="mesh"))
    object_handle = server.scene.add_mesh_simple(
        "/dexycb_object",
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
        color=(150, 210, 255),
    )
    print(f"[dexycb-online] ee={args.ee} sequence={data['capture_name']} frames={len(points)}")
    print(f"[dexycb-online] Open: http://localhost:{server.get_port()}")

    loop = 0
    max_base_offset = 0.0
    while args.loops == 0 or loop < args.loops:
        for index, target in enumerate(points):
            packed = solver.solve_numpy(target)[0]
            frame.position = packed[:3]
            frame.wxyz = packed[3:7]
            urdf.update_cfg(packed[7:])
            hand.vertices = data["mano_vertices"][index]
            object_handle.position = data["object_pos"][index]
            object_handle.wxyz = data["object_quat_wxyz"][index]
            max_base_offset = max(max_base_offset, float(np.linalg.norm(packed[:3] - target[root_index])))
            time.sleep(1.0 / args.fps)
        loop += 1
    print(f"[dexycb-online] completed loops={loop} frames={loop * len(points)} max_base_offset_m={max_base_offset:.8f}")
    server.stop()


if __name__ == "__main__":
    main()
