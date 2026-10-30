"""Hand-contact gallery: select an arm-hand, DexYCB clip, and contact refinement mode."""

import argparse
import importlib
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType

import numpy as np
import trimesh
import viser
import warp as wp
from viser.extras import ViserUrdf

from robokit.assets.datasets import dexycb
from robokit.geom import MeshGeom, WarpScene
from robokit.helpers.hand_retargeting import (
    HandRetargeterOffline,
    HandRetargeterOfflineConfig,
    HandSpec,
    contact_config,
    refine_config,
)
from robokit.helpers.hand_retargeting.dexycb_utils import DexYCBVideoDataset, load_dexycb_grasp
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy import quaternion_apply, quaternion_invert, quaternion_multiply


ROBOTS = ("xarm7_xhand", "xarm7_ability")
CONTACT_MESH_LINKS = {
    "xarm7_xhand": (
        "right_hand_thumb_rota_tip",
        "right_hand_index_rota_tip",
        "right_hand_mid_tip",
        "right_hand_ring_tip",
        "right_hand_pinky_tip",
    ),
    "xarm7_ability": ("thumb_L2", "index_L2", "middle_L2", "ring_L2", "pinky_L2"),
}
BASE_POSE = np.array([0.9, 0.9, 0.0, 0.70710678, 0.0, 0.0, -0.70710678], np.float32)
CONTACT_START_DISTANCE = 0.020
CONTACT_FULL_DISTANCE = 0.006
CONTACT_EMBED_DEPTH = 0.005
CONTACT_MARKER_RADIUS = 0.004


@dataclass(frozen=True)
class Solved:
    name: str
    urdf_path: Path
    joints: np.ndarray
    T_world_base: np.ndarray
    object_pos: np.ndarray
    object_quat_wxyz: np.ndarray
    object_mesh: trimesh.Trimesh
    mano_vertices: np.ndarray
    mano_faces: np.ndarray
    contact_points_world: np.ndarray
    contact_mask: np.ndarray
    contact_enabled: bool


def _load_config(name: str) -> tuple[ModuleType, HandSpec, HandRetargeterOfflineConfig]:
    asset = importlib.import_module(f"robokit.assets.robots.assembly.{name}")
    preset = importlib.import_module(f"robokit.helpers.hand_retargeting.presets.{name}")
    hand = importlib.import_module(f"robokit.helpers.hand_retargeting.presets.{name.removeprefix('xarm7_')}")
    return asset, replace(preset.spec, root_link_coord_spec=hand.spec.root_link_coord_spec), preset.offline


def _solve(robot_name: str, capture_index: int, contact: bool, device: str, max_frames: int) -> Solved:
    start = time.time()
    asset, spec, defaults = _load_config(robot_name)
    robot = Robot.load(
        str(asset.URDF_PATH),
        load_meshes=True,
        load_collision_spheres=True,
        collision_spheres_path=asset.COLLISION_SPHERE_PATH,
    )
    data = load_dexycb_grasp(capture_index, device=device)
    keypoints = data["keypoints"]
    object_pos = data["object_pos"]
    object_quat_wxyz = data["object_quat_wxyz"]
    mano_vertices = data["mano_vertices"]
    if max_frames:
        keypoints = keypoints[:max_frames]
        object_pos = object_pos[:max_frames]
        object_quat_wxyz = object_quat_wxyz[:max_frames]
        mano_vertices = mano_vertices[:max_frames]
    num_frames = len(keypoints)

    contact_target_indices = np.asarray(
        [spec.target_names.index(name) for name in spec.contact_target_names], dtype=np.int32
    )
    contact_link_indices = np.asarray(
        [robot.spec.link_names.index(spec.target_link_names[name]) for name in spec.contact_target_names],
        dtype=np.int32,
    )
    q_wxyz_object_world = quaternion_invert(object_quat_wxyz)
    keypoints_object = quaternion_apply(q_wxyz_object_world[:, None], keypoints - object_pos[:, None]).astype(
        np.float32
    )
    fingertips_object = keypoints_object[:, contact_target_indices]
    closest, distance, _ = trimesh.proximity.closest_point(data["object_mesh"], fingertips_object.reshape(-1, 3))
    contact_points_object = closest.reshape(fingertips_object.shape).astype(np.float32)
    contact_mask = distance.reshape(fingertips_object.shape[:2]) < CONTACT_START_DISTANCE
    contact_points_world = object_pos[:, None] + quaternion_apply(object_quat_wxyz[:, None], contact_points_object)
    direct_config = contact_config(defaults)
    direct_config.task("collision").weight = 0.0
    direct = HandRetargeterOffline(config=direct_config, robot=robot, spec=spec, device=device)
    # Start the full trajectory on the first pose's IK branch, with the actual wrist orientation.
    q_init = np.tile(robot.spec.zero_q.astype(np.float32), (1, 3, 1))
    joint_index = {name: index for index, name in enumerate(robot.spec.actuated_joint_names)}
    for name, value in spec.init_q_by_name.items():
        q_init[:, :, joint_index[name]] = value
    first_state = robot.state(
        q=wp.from_numpy(q_init, dtype=wp.float32, device=device),
        T_world_base=wp.from_numpy(np.tile(BASE_POSE, (1, 3, 1)), dtype=wp_vec7, device=device),
    )
    first_pose = direct.solve_numpy(
        np.repeat(keypoints[None, :1], 3, axis=1),
        np.repeat(data["wrist_quat_wxyz"][None, :1], 3, axis=1),
        first_state,
    )
    initial = robot.state(
        q=wp.from_numpy(np.tile(first_pose[:, :1, 7:], (1, num_frames, 1)), dtype=wp.float32, device=device),
        T_world_base=wp.from_numpy(np.tile(BASE_POSE, (1, num_frames, 1)), dtype=wp_vec7, device=device),
    )
    packed = direct.solve_numpy(keypoints[None], data["wrist_quat_wxyz"][None, :num_frames], initial)[0]

    if contact:
        T_object_base = np.empty((num_frames, 7), dtype=np.float32)
        T_object_base[:, :3] = quaternion_apply(q_wxyz_object_world, BASE_POSE[:3] - object_pos)
        T_object_base[:, 3:] = quaternion_multiply(q_wxyz_object_world, BASE_POSE[3:])
        init_state = robot.state(
            q=wp.from_numpy(packed[None, :, 7:], dtype=wp.float32, device=device),
            T_world_base=wp.from_numpy(T_object_base[None], dtype=wp_vec7, device=device),
        )
        robot.forward_kinematics(init_state)
        T_object_all_links = init_state.T_world_link.numpy()[0]
        T_object_link = T_object_all_links[:, contact_link_indices]
        contact_outward = fingertips_object - contact_points_object
        contact_outward /= np.maximum(np.linalg.norm(contact_outward, axis=-1, keepdims=True), 1e-8)
        contact_inward_local = quaternion_apply(quaternion_invert(T_object_link[..., 3:]), -contact_outward)
        local_contact_points = np.zeros_like(contact_points_object)
        for contact_index, mesh_link_name in enumerate(CONTACT_MESH_LINKS[robot_name]):
            mesh_link_index = robot.spec.link_names.index(mesh_link_name)
            link_mesh = robot.spec.link_visual_geometries[mesh_link_name].to_mesh()
            vertices_mesh = np.asarray(link_mesh.vertices, np.float32)
            T_object_mesh = T_object_all_links[0, mesh_link_index]
            T_object_target = T_object_link[0, contact_index]
            vertices_object = T_object_mesh[:3] + quaternion_apply(T_object_mesh[3:], vertices_mesh)
            vertices_target = quaternion_apply(
                quaternion_invert(T_object_target[3:]), vertices_object - T_object_target[:3]
            )
            local_contact_points[:, contact_index] = vertices_target[
                np.argmax(contact_inward_local[:, contact_index] @ vertices_target.T, axis=1)
            ]

        root_link_index = robot.spec.link_names.index(spec.target_link_names[spec.root_target_name])
        arm_joint_mask = robot.spec.link_ancestor_joints_mask[root_link_index]
        arm_dof_mask = np.any(robot.spec.joints_to_actuated_mapping[arm_joint_mask] != 0.0, axis=0)
        hand_joint_names = tuple(np.asarray(robot.spec.actuated_joint_names)[~arm_dof_mask])
        identity = wp.from_numpy(np.eye(4, dtype=np.float32)[None], dtype=wp.mat44, device=device)
        scene = WarpScene(1, device).add(
            MeshGeom([data["object_mesh"]], np.array([0, 1], dtype=np.int32), poses=identity)
        )
        refinement = refine_config(
            replace(
                defaults,
                optimize_base=False,
                active_joint_names=hand_joint_names,
                locked_prefix_frames=min(2, num_frames),
            )
        )
        refinement.task("contact").weight = 300.0
        refinement.task("contact").contact_margin = 0.0  # pyright: ignore[reportAttributeAccessIssue]
        refinement.task("anchor").dense_task.weight = 10.0
        refinement.task("velocity").weight = 30.0
        refinement.task("acceleration").weight = 100.0
        refinement.task("collision").weight = 100.0
        refinement.task("collision").margin = 0.0  # pyright: ignore[reportAttributeAccessIssue]
        contact_points_robot = T_object_link[..., :3] + quaternion_apply(T_object_link[..., 3:], local_contact_points)
        contact_strength = np.clip(
            (CONTACT_START_DISTANCE - distance.reshape(contact_mask.shape))
            / (CONTACT_START_DISTANCE - CONTACT_FULL_DISTANCE),
            0.0,
            1.0,
        ).astype(np.float32)
        contact_strength = contact_strength * contact_strength * (3.0 - 2.0 * contact_strength)
        contact_points_target = contact_points_object - CONTACT_EMBED_DEPTH * contact_outward
        contact_points_target = contact_points_robot + contact_strength[..., None] * (
            contact_points_target - contact_points_robot
        )
        packed = HandRetargeterOffline(
            config=refinement, robot=robot, spec=spec, scene=scene, device=device
        ).solve_with_contact_numpy(
            keypoints_object,
            contact_points_target,
            contact_mask,
            init_state=init_state,
            local_contact_points=local_contact_points,
        )
        joints, T_object_base = packed[:, 7:], packed[:, :7]
        T_world_base = np.empty_like(T_object_base)
        T_world_base[:, :3] = object_pos + quaternion_apply(object_quat_wxyz, T_object_base[:, :3])
        T_world_base[:, 3:] = quaternion_multiply(object_quat_wxyz, T_object_base[:, 3:])
    else:
        joints = packed[:, 7:]
        T_world_base = np.tile(BASE_POSE, (num_frames, 1))

    print(
        f"hand-contact gallery: robot={robot_name} clip={capture_index} contact={contact} "
        f"frames={num_frames} touching={int(contact_mask.any(1).sum())} time={time.time() - start:.1f}s"
    )
    return Solved(
        str(data["capture_name"]),
        asset.URDF_PATH,
        joints,
        T_world_base,
        object_pos,
        object_quat_wxyz,
        data["object_mesh"],
        mano_vertices,
        data["mano_faces"],
        contact_points_world,
        contact_mask,
        contact,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=ROBOTS, default=ROBOTS[0])
    parser.add_argument("--capture", type=int, default=0)
    parser.add_argument("--no-contact", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    dataset = DexYCBVideoDataset(dexycb.DIR)
    captures = tuple(str(index) for index in range(len(dataset)))
    if args.headless:
        _solve(args.robot, args.capture, not args.no_contact, device, args.max_frames)
        return

    server = viser.ViserServer()
    server.scene.add_grid("/grid", 1.0, 1.0, position=(0.5, 0.5, 0.0))
    robot_frame = server.scene.add_frame("/robot", show_axes=False)
    object_frame = server.scene.add_frame("/object", show_axes=False)
    with server.gui.add_folder("Source"):
        robot_dd = server.gui.add_dropdown("Robot", ROBOTS, initial_value=args.robot)
        capture_dd = server.gui.add_dropdown("DexYCB clip", captures, initial_value=str(args.capture))
        contact_cb = server.gui.add_checkbox("Contact refinement", not args.no_contact)
    playing = server.gui.add_checkbox("Play", True)
    show_contacts = server.gui.add_checkbox("Show contacts", True)
    frame_slider = server.gui.add_slider("Frame", 0, 1, 1, 0)
    status = server.gui.add_text("Status", initial_value="Solving...", disabled=True)
    spheres = [
        server.scene.add_icosphere(f"/contact_{index}", radius=CONTACT_MARKER_RADIUS, color=(255, 40, 40))
        for index in range(5)
    ]

    cache: dict[tuple[str, int, bool], Solved] = {}
    urdf_vis = None
    object_mesh = None
    mano = None
    current = None
    built = ("", -1, False)
    while True:
        selection = (robot_dd.value, int(capture_dd.value), contact_cb.value)
        if selection != built:
            status.value = f"Solving clip {selection[1]} on {selection[0]}..."
            if selection not in cache:
                cache[selection] = _solve(*selection, device, args.max_frames)
            current = cache[selection]
            if urdf_vis is not None:
                urdf_vis.remove()
            if object_mesh is not None:
                object_mesh.remove()
            if mano is not None:
                mano.remove()
            urdf_vis = ViserUrdf(server, current.urdf_path, root_node_name="/robot")
            object_mesh = server.scene.add_mesh_simple(
                "/object/mesh",
                current.object_mesh.vertices,
                current.object_mesh.faces,
                color=(180, 180, 190),
                opacity=0.6,
            )
            mano = server.scene.add_mesh_simple(
                "/mano",
                current.mano_vertices[0],
                current.mano_faces,
                color=(120, 200, 130),
                opacity=0.35,
            )
            frame_slider.max = len(current.joints) - 1
            frame_slider.value = 0
            status.value = f"{current.name} / {selection[0]} / contact={selection[2]}"
            built = selection

        assert current is not None and urdf_vis is not None and mano is not None
        if playing.value:
            frame_slider.value = (frame_slider.value + 1) % len(current.joints)
        frame = min(frame_slider.value, len(current.joints) - 1)
        with server.atomic():
            robot_frame.position, robot_frame.wxyz = current.T_world_base[frame, :3], current.T_world_base[frame, 3:]
            urdf_vis.update_cfg(current.joints[frame])
            object_frame.position, object_frame.wxyz = current.object_pos[frame], current.object_quat_wxyz[frame]
            mano.vertices = current.mano_vertices[frame]
            for index, sphere in enumerate(spheres):
                sphere.visible = bool(
                    show_contacts.value and current.contact_enabled and current.contact_mask[frame, index]
                )
                sphere.position = current.contact_points_world[frame, index]
        time.sleep(0.03)


if __name__ == "__main__":
    main()
