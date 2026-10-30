"""Visualize generated YCB grasp specs or captured contact-supervision data.

Usage:
    uv run --extra contact --extra vis scripts/teleop/neural_hand/contact_visualize.py
"""

import argparse
import time
from pathlib import Path

import numpy as np
import viser

from retargetingkit.assets import calib
from retargetingkit.retargeting.hand.presets import HANDS
from retargetingkit.sources.glove.manus_conversion import manus_positions_to_fixed_wrist_local
from retargetingkit.training.hand.contact import N_FINGERS, load_ycb_mesh
from retargetingkit.training.hand.data import ContactCaptureData, GraspSpecData
from retargetingkit.utils.viz import add_contact_spheres, add_hand_urdf, add_object_nodes, draw_manus_skeleton
from robokit.xform.numpy import axis_angle_to_quaternion


HUMAN_OFFSET = (0.0, 0.3, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize YCB contact-alignment grasps.")
    parser.add_argument("--data", type=Path, default=calib.get("sharpa_contact_grasp_ycb"))
    parser.add_argument("--captured", action="store_true")
    args = parser.parse_args()

    capture = ContactCaptureData.load(args.data) if args.captured else None
    grasps = capture.grasps if capture is not None else GraspSpecData.load(args.data)
    names = [str(name) for name in grasps.object_names]
    object_wxyz = axis_angle_to_quaternion(grasps.object_rotvec)
    print(f"frames={len(grasps)} captured={capture is not None}", flush=True)
    meshes = {}
    for name in sorted(set(names)):
        mesh = load_ycb_mesh(name)
        meshes[name] = (np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int32))

    server = viser.ViserServer()
    server.scene.add_grid("/grid", 1.0, 1.0)
    robot_urdf, viser_joint_indices = add_hand_urdf(server, HANDS["sharpa"], "/robot")
    object_nodes = add_object_nodes(server, "/robot", meshes)
    contact_nodes = add_contact_spheres(server, "/robot")
    if capture is not None:
        server.scene.add_frame("/human", position=HUMAN_OFFSET, show_axes=False)
    frame_text = server.gui.add_text("Frame", initial_value=f"0/{len(grasps) - 1}", disabled=True)
    status = server.gui.add_text("Status", initial_value="", disabled=True)

    def update(frame: int) -> None:
        frame_text.value = f"{frame}/{len(grasps) - 1}"
        name = names[frame]
        robot_urdf.update_cfg(
            np.asarray([grasps.qpos[frame, index] for index in viser_joint_indices], dtype=np.float32)
        )
        with server.atomic():
            for other, node in object_nodes.items():
                node.visible = other == name
            object_nodes[name].position = tuple(grasps.object_translation[frame].tolist())
            object_nodes[name].wxyz = tuple(object_wxyz[frame].tolist())
            for finger in range(N_FINGERS):
                contact_nodes[finger].visible = bool(grasps.contact_mask[frame, finger])
                contact_nodes[finger].position = tuple(grasps.contact_points[frame, finger].tolist())
            if capture is not None:
                wrist_local = manus_positions_to_fixed_wrist_local(
                    capture.positions[frame],
                    capture.rotations[frame],
                )
                draw_manus_skeleton(server, wrist_local, "/human")
        grip = "".join("TIMRP"[finger] for finger in range(N_FINGERS) if grasps.contact_mask[frame, finger])
        status.value = f"frame={frame}/{len(grasps) - 1} object={name} grip={grip}"

    current = [0]

    def step(delta: int) -> None:
        current[0] = (current[0] + delta) % len(grasps)
        update(current[0])

    server.gui.add_button("Previous Frame").on_click(lambda _: step(-1))
    server.gui.add_button("Next Frame").on_click(lambda _: step(1))
    update(0)
    print(f"viser={server.get_port()}", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
