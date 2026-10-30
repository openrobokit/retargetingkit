"""Terrain gallery: select a climbing clip and humanoid, retarget, and scrub the result."""

import argparse
import time

import numpy as np
import trimesh
import viser
import warp as wp

from robokit.assets.datasets import climb
from robokit.geom import MeshGeom, PlaneGeom, WarpScene
from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnline, build_interaction_mesh_frame
from robokit.helpers.humanoid_retargeting.loaders import load_mocap_clip
from robokit.helpers.humanoid_retargeting.presets.bhl_mocap import BHL_HEIGHT, bhl_mocap
from robokit.helpers.humanoid_retargeting.presets.g1_mocap import G1_HEIGHT, g1_mocap
from robokit.helpers.humanoid_retargeting.presets.gr3_mocap import GR3_HEIGHT, gr3_mocap
from robokit.helpers.humanoid_retargeting.presets.h1_2_mocap import H1_2_HEIGHT, h1_2_mocap
from robokit.robo import Robot
from robokit.utils.visualize_utils import ViserBatchUrdf
from robokit.utils.warp_utils import wp_vec7


ROBOTS = {  # preset and robot height
    "g1": (g1_mocap, G1_HEIGHT),
    "gr3": (gr3_mocap, GR3_HEIGHT),
    "h1_2": (h1_2_mocap, H1_2_HEIGHT),
    "bhl": (bhl_mocap, BHL_HEIGHT),
}
CLIPS = tuple(f"mocap_climb_seq_{index}" for index in range(5))
FPS = 30.0


def _solve(robot_name: str, clip_name: str, device: str, max_frames: int) -> tuple[Robot, trimesh.Trimesh, np.ndarray]:
    ref = climb.sequence_dir(clip_name)
    preset, robot_height = ROBOTS[robot_name]
    robot = preset.load_robot(load_meshes=True)
    T_world_human, _, interaction_scale = load_mocap_clip(
        str(sorted(ref.glob("*.npy"))[0]), preset.human_joint_names, robot_height=robot_height
    )
    if max_frames:
        T_world_human = T_world_human[:max_frames]

    terrain = trimesh.load(str(ref / "multi_boxes.obj"), force="mesh")
    assert isinstance(terrain, trimesh.Trimesh)
    terrain.apply_scale(interaction_scale)
    interaction_points, _ = trimesh.sample.sample_surface_even(terrain, 100, seed=42)
    scene = (
        WarpScene(1, device)
        .add(MeshGeom([terrain], np.array([0, 1], dtype=np.int32)))
        .add(PlaneGeom(np.array([0, 1], dtype=np.int32)))
    )
    retargeter = HumanoidRetargeterOnline(preset, robot=robot, scene=scene, device=device)
    key_indices = [retargeter.human_joint_names.index(entry.human_joint) for entry in preset.link_mapping.values()]
    terrain_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    retargeter.interaction_task.set_frame(
        *build_interaction_mesh_frame(T_world_human[0, key_indices, :3], interaction_points, terrain_pose)
    )
    retargeter.warmup(1)
    root = T_world_human[0, retargeter.human_joint_names.index(preset.human_root_name), :3]
    face_xy = interaction_points.mean(0)[:2]
    yaw = np.arctan2(face_xy[1] - root[1], face_xy[0] - root[0])
    T_world_base = np.array([*root, np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32)
    retargeter.reset(wp.from_numpy(T_world_base[None], dtype=wp_vec7, device=device))

    start = time.time()
    qpos = []
    for frame in T_world_human:
        retargeter.interaction_task.set_frame(
            *build_interaction_mesh_frame(frame[key_indices, :3], interaction_points, terrain_pose)
        )
        qpos.append(retargeter.solve_numpy(frame[None])[0])
    result = np.stack(qpos)
    print(f"terrain gallery: robot={robot_name} clip={clip_name} frames={len(result)} time={time.time() - start:.1f}s")
    return robot, terrain, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=tuple(ROBOTS), default="g1")
    parser.add_argument("--clip", choices=CLIPS, default=CLIPS[0])
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    if args.headless:
        _solve(args.robot, args.clip, device, args.max_frames)
        return

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=10, height=10)
    with server.gui.add_folder("Source"):
        robot_dd = server.gui.add_dropdown("Robot", tuple(ROBOTS), initial_value=args.robot)
        clip_dd = server.gui.add_dropdown("Terrain / Clip", CLIPS, initial_value=args.clip)
    frame_slider = server.gui.add_slider("Frame", 0, 1, 1, 0)
    playing = server.gui.add_checkbox("Play", True)
    status = server.gui.add_text("Status", initial_value="Solving...", disabled=True)

    cache: dict[tuple[str, str], tuple[Robot, trimesh.Trimesh, np.ndarray]] = {}
    robot_viz = None
    terrain_viz = None
    qpos = np.zeros((1, 8), dtype=np.float32)
    ndof = 1
    built = ("", "")
    while True:
        selection = (robot_dd.value, clip_dd.value)
        if selection != built:
            status.value = f"Solving {selection[1]} on {selection[0]}..."
            if selection not in cache:
                cache[selection] = _solve(*selection, device, args.max_frames)
            robot, terrain, qpos = cache[selection]
            if robot_viz is not None:
                robot_viz.remove()
            if terrain_viz is not None:
                terrain_viz.remove()
            robot_viz = ViserBatchUrdf(server, robot, batch_size=1, root_node_name="/robot")
            terrain_viz = server.scene.add_mesh_trimesh("/terrain", terrain)
            ndof = robot.spec.num_actuated_joints
            frame_slider.max = len(qpos) - 1
            frame_slider.value = 0
            status.value = f"{selection[0]} / {selection[1]}: {len(qpos)} frames"
            built = selection

        if playing.value:
            frame_slider.value = (frame_slider.value + 1) % len(qpos)
        frame = min(frame_slider.value, len(qpos) - 1)
        assert robot_viz is not None
        robot_viz.update_cfg(qpos[frame : frame + 1, 7 : 7 + ndof], T_world_base=qpos[frame : frame + 1, :7])
        time.sleep(1.0 / FPS)


if __name__ == "__main__":
    main()
