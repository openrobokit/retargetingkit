"""GigaHands hand-object gallery with a stable wrist during contact refinement."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import viser
import warp as wp
from viser.extras import ViserUrdf

from retargetingkit.assets import gigahands_hoi
from retargetingkit.retargeting.hand_object.offline import (
    CONTACT_LINK_TARGETS,
    MANO16_LINKS,
    HandObjectMotion,
    HandObjectRetargeter,
    hand_object_configs,
    load_hand_object_clip,
)
from retargetingkit.sources.frames import Side
from robokit.assets.robots.end_effectors import sharpa_hand
from robokit.helpers.hand_retargeting import HandSpec
from robokit.helpers.hand_retargeting.presets import sharpa
from robokit.robo import Robot
from robokit.utils.ee_coord_utils import HandCoordinateSpec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=0, help="maximum sampled frames; 0 keeps the full clip")
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    asset_root = gigahands_hoi.root()
    manifest = json.loads((asset_root / "manifest.json").read_text())
    clips = {f"{index:02d} {clip['label']}": clip["path"] for index, clip in enumerate(manifest["clips"], 1)}
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    urdf_paths: dict[Side, Path] = {"right": sharpa_hand.URDF_PATH_RIGHT, "left": sharpa_hand.URDF_PATH_LEFT}
    robots: dict[Side, Robot] = {
        side: Robot.load(str(path), load_meshes=True, mesh_dir=str(path.parent)) for side, path in urdf_paths.items()
    }
    specs: dict[Side, HandSpec] = {
        "right": sharpa.spec,
        "left": replace(
            sharpa.spec,
            target_link_names={
                name: link.replace("right_", "left_") for name, link in sharpa.spec.target_link_names.items()
            },
            root_link_coord_spec=HandCoordinateSpec("-z", "-y"),
        ),
    }
    for side, spec in specs.items():
        specs[side] = replace(
            spec,
            target_coord_spec=HandCoordinateSpec(
                **{name: axis.removeprefix("+") for name, axis in manifest["wrist_coordinate"][side].items()}
            ),
        )
    track, refine = hand_object_configs(
        replace(sharpa.offline, optimize_base=False, solver=replace(sharpa.offline.solver, max_iter=100))
    )
    for config in (track, refine):
        config.task("root_smoothness").base_weight[0:3] = 30.0  # pyright: ignore[reportAttributeAccessIssue]
        config.task("root_smoothness").base_weight[3:6] = 50.0  # pyright: ignore[reportAttributeAccessIssue]
    retargeter = HandObjectRetargeter(robots, specs, track, refine, device)

    labels = tuple(clips)

    def solve(label: str) -> HandObjectMotion:
        clip = load_hand_object_clip(asset_root / clips[label])
        if args.frames:
            indices = np.linspace(0, clip.num_frames - 1, min(args.frames, clip.num_frames)).round().astype(np.int32)
            clip = clip.select(indices)
        start = time.time()
        motion = retargeter.solve(clip)
        print(f"retargeted {label}: {clip.num_frames} frames in {time.time() - start:.2f}s on {device}")
        return motion

    if args.no_viewer:
        solve(labels[0])
        return

    contact_indices = [MANO16_LINKS.index(name) for name, _ in CONTACT_LINK_TARGETS]
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=3.0, height=3.0)
    with server.gui.add_folder("Source"):
        clip_dd = server.gui.add_dropdown("Clip", labels, initial_value=labels[0])
    slider = server.gui.add_slider("frame", 0, 1, 1, 0)
    playing = server.gui.add_checkbox("play", True)
    show_contacts = server.gui.add_checkbox("contacts", True)
    status = server.gui.add_text("status", initial_value="solving...", disabled=True)
    dirty = [True]
    clip_dd.on_update(lambda _: dirty.__setitem__(0, True))

    robot_frames = {side: server.scene.add_frame(f"/robot/{side}", show_axes=False, visible=False) for side in robots}
    robot_urdfs = {side: ViserUrdf(server, urdf_paths[side], root_node_name=f"/robot/{side}") for side in robots}
    contacts = {
        side: [
            server.scene.add_icosphere(f"/contacts/{side}/{slot}", radius=0.004, color=(40, 220, 80), visible=False)
            for slot in range(len(contact_indices))
        ]
        for side in robots
    }

    solved: dict[str, HandObjectMotion] = {}
    current: HandObjectMotion | None = None
    object_meshes = []
    contact_world: dict[Side, np.ndarray] = {}
    offset = np.zeros(3, dtype=np.float32)
    built = ""
    while True:
        if dirty[0]:
            dirty[0] = False
            selection = clip_dd.value
            if selection != built:
                status.value = f"solving {selection}..."
                if selection not in solved:
                    solved[selection] = solve(selection)
                current = solved[selection]
                clip = current.clip
                for side in robots:
                    robot_frames[side].visible = side in current.hands
                    for marker in contacts[side]:
                        marker.visible = False
                server.scene.remove_by_name("/object")
                object_meshes = [
                    server.scene.add_mesh_simple(
                        f"/object/{part.name}", part.vertices, part.faces, color=(210, 150, 70), opacity=0.8
                    )
                    for part in clip.object.parts
                ]
                contact_world = {side: clip.contacts_world(side)[:, contact_indices] for side in clip.hands}
                offset = np.array([0.0, 0.0, 1.0], np.float32) - clip.object.root_pose_world[0, :3]
                slider.value = 0
                slider.max = clip.num_frames - 1
                status.value = f"{selection}: {clip.num_frames} frames"
                built = selection

        if current is not None:
            clip = current.clip
            frame = min(slider.value, clip.num_frames - 1)
            if playing.value:
                frame = (frame + 1) % clip.num_frames
                slider.value = frame
            with server.atomic():
                for side, track in current.hands.items():
                    pose = track.base_pose_world[frame]
                    robot_frames[side].position = pose[:3] + offset
                    robot_frames[side].wxyz = pose[3:]
                    robot_urdfs[side].update_cfg(track.qpos[frame])
                    mask = clip.hands[side].contact_mask[frame, contact_indices]
                    for slot, handle in enumerate(contacts[side]):
                        handle.visible = bool(show_contacts.value and mask[slot])
                        handle.position = contact_world[side][frame, slot] + offset
                for part_index, mesh in enumerate(object_meshes):
                    pose = clip.object.part_pose_world[frame, part_index]
                    mesh.position, mesh.wxyz = pose[:3] + offset, pose[3:]
        time.sleep(1.0 / (current.clip.fps if current is not None else 30.0))


if __name__ == "__main__":
    main()
