"""ParaHome full-body gallery with optional grasp holding and scene clearance."""

import argparse
import time
import zlib
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import viser
import warp as wp

from retargetingkit.retargeting.full_body import FullBodyRetargeter
from robokit.assets.datasets import parahome
from robokit.helpers.full_body_retargeting import (
    FullBodyClip,
    HandTrack,
    solve_grasp_refine,
)
from robokit.helpers.full_body_retargeting.parahome_utils import (
    ParaHomeInterval,
    find_parahome_intervals,
    load_parahome_clip,
)
from robokit.utils.visualize_utils import ViserBatchUrdf, ViserHandSkeleton


MARKER_RADIUS = 0.004


class Solved(NamedTuple):
    """One solved interval and its per-side contact markers."""

    qpos: np.ndarray
    clip: FullBodyClip
    targets: Dict[str, np.ndarray]
    contacts: Optional[Dict[str, np.ndarray]]
    masks: Dict[str, np.ndarray]


def _world_points(points: np.ndarray, track: HandTrack) -> np.ndarray:
    """Bridge native-world points onto the scaled body world through the track's object pose."""
    frames = points.shape[0]
    return points - track.T_native_object[:frames, None, :3] + track.T_world_object[:frames, None, :3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=None, help="ParaHome data/ dir (default: fetch from HF)")
    parser.add_argument("--sequence", default="s1", help="sequence id, s1..s207")
    parser.add_argument("--index", type=int, default=0, help="which mined interval to retarget")
    parser.add_argument("--list", action="store_true", help="print the sequence's intervals and exit")
    parser.add_argument("--max_frames", type=int, default=0, help="0 keeps all frames")
    parser.add_argument("--grasp", action="store_true", help="optimize and hold grasps over contact runs")
    parser.add_argument("--no_scene", action="store_true", help="hide the furnished room")
    parser.add_argument("--scene_clearance", action="store_true", help="push the body out of penetrated scene parts")
    parser.add_argument("--headless", action="store_true", help="solve the first interval and exit")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else parahome.data_root()
    sequences = tuple(path.name for path in sorted((data_root / "seq").iterdir(), key=lambda path: int(path.name[1:])))
    intervals: Dict[str, List[ParaHomeInterval]] = {}

    def intervals_of(sequence: str) -> List[ParaHomeInterval]:
        if sequence not in intervals:
            intervals[sequence] = find_parahome_intervals(str(data_root), sequence)
        return intervals[sequence]

    def clips_of(sequence: str) -> Tuple[str, ...]:
        return tuple(f"{i}: {interval.label}"[:70] for i, interval in enumerate(intervals_of(sequence))) or ("(none)",)

    assert args.sequence in sequences, f"unknown sequence: {args.sequence}"
    initial_intervals = intervals_of(args.sequence)
    assert initial_intervals, f"no hand-engaged intervals in {args.sequence}"
    if args.list:
        for i, interval in enumerate(initial_intervals):
            print(f"[{i:2d}] {interval.start:5d}-{interval.end:5d} {interval.label}")
        return
    assert 0 <= args.index < len(initial_intervals), f"index must be in [0, {len(initial_intervals) - 1}]"
    initial_clip = clips_of(args.sequence)[args.index]

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    retargeter = FullBodyRetargeter(device=device, load_meshes=True)

    def solve(sequence: str, clip_name: str, scene_clearance: bool) -> Solved:
        print(f"retargeting parahome {sequence}/{clip_name} (grasp={args.grasp}, clearance={scene_clearance})")
        clip = load_parahome_clip(
            str(data_root),
            intervals_of(sequence)[int(clip_name.split(":", maxsplit=1)[0])],
            device=device,
            load_scene=not args.no_scene,
        )
        start = time.time()
        if args.grasp:
            result = solve_grasp_refine(clip, retargeter, max_frames=args.max_frames, scene_clearance=scene_clearance)
            solved = Solved(result.qpos, clip, result.targets_world, result.contacts_world, result.contact_masks)
        else:
            qpos = retargeter.solve(clip, max_frames=args.max_frames, scene_clearance=scene_clearance)
            frames = qpos.shape[0]
            targets = {side: _world_points(track.contact_points[:frames], track) for side, track in clip.hands.items()}
            masks = {side: track.contact_mask[:frames] for side, track in clip.hands.items()}
            solved = Solved(qpos, clip, targets, None, masks)
        print(f"solved {solved.qpos.shape[0]} frames in {time.time() - start:.1f}s")
        return solved

    if args.headless:
        solve(args.sequence, initial_clip, args.scene_clearance)
        return

    # --- viewer: source panel + playback controls (solves are cached per selection) ---
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=6, height=6)
    with server.gui.add_folder("Source"):
        sequence_dd = server.gui.add_dropdown("Sequence", sequences, initial_value=args.sequence)
        clip_dd = server.gui.add_dropdown("Interval", clips_of(args.sequence), initial_value=initial_clip)
        # Eject the core body from furniture the scaled robot over-penetrates (feet/hands stay pinned).
        scene_clearance_cb = server.gui.add_checkbox("scene clearance", args.scene_clearance)
    frame_slider = server.gui.add_slider("frame", min=0, max=1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("play", True)
    show_hand = server.gui.add_checkbox("human hands", True)
    show_contacts = server.gui.add_checkbox("grasp contacts" if args.grasp else "ParaHome contacts", True)
    show_scene = server.gui.add_checkbox("room", True)
    status = server.gui.add_text("status", initial_value="solving...", disabled=True)
    dirty = [True]

    def on_sequence(_: object) -> None:
        clip_dd.options = clips_of(sequence_dd.value)
        clip_dd.value = clip_dd.options[0]
        dirty[0] = True

    sequence_dd.on_update(on_sequence)
    clip_dd.on_update(lambda _: dirty.__setitem__(0, True))
    scene_clearance_cb.on_update(lambda _: dirty.__setitem__(0, True))

    robot_viz = ViserBatchUrdf(server, retargeter.body_robot, batch_size=1, root_node_name="/robot")
    ndof = retargeter.body_robot.spec.num_actuated_joints
    solved: Dict[Tuple[str, str, bool], Solved] = {}
    current: Optional[Solved] = None
    mano_world: Dict[str, np.ndarray] = {}
    hand_viz: Dict[str, ViserHandSkeleton] = {}
    object_handles: List[Tuple[int, viser.MeshHandle]] = []
    markers: Dict[str, List[Tuple[viser.IcosphereHandle, Optional[viser.IcosphereHandle]]]] = {}
    scene_handles: List[Tuple[int, viser.MeshHandle, bool]] = []
    built: Tuple[str, str, bool] = ("", "", False)
    while True:
        if dirty[0]:
            # Snapshot the dropdowns ONCE (callbacks fire on GUI worker threads at any time).
            dirty[0] = False
            selection = (sequence_dd.value, clip_dd.value, scene_clearance_cb.value)
            if selection != built and selection[1] in clips_of(selection[0]):
                if selection not in solved:
                    status.value = f"solving {clip_dd.value}..."
                    solved[selection] = solve(*selection)
                current = solved[selection]
                mano_world = {side: _world_points(track.mano, track) for side, track in current.clip.hands.items()}

                for node in ("/mano", "/contacts", "/object", "/scene"):
                    server.scene.remove_by_name(node)
                hand_viz = {
                    side: ViserHandSkeleton(
                        server, f"/mano/{side}", color=(220, 40, 40) if side == "right" else (40, 90, 220)
                    )
                    for side in current.clip.sides
                }
                object_handles = []
                for obj_i, name in enumerate(current.clip.object_names):
                    track = next(hand for hand in current.clip.hands.values() if hand.object_name == name)
                    assert track.object_mesh is not None
                    object_handles.append(
                        (
                            obj_i,
                            server.scene.add_mesh_simple(
                                f"/object/{name}",
                                track.object_mesh.vertices,
                                track.object_mesh.faces,
                                color=(230, 130, 40),
                                opacity=0.8,
                            ),
                        )
                    )
                markers = {
                    side: [
                        (
                            server.scene.add_icosphere(
                                f"/contacts/{side}/target_{slot}", radius=MARKER_RADIUS, color=(40, 220, 80)
                            ),
                            None
                            if current.contacts is None
                            else server.scene.add_icosphere(
                                f"/contacts/{side}/pad_{slot}", radius=MARKER_RADIUS * 0.7, color=(190, 50, 230)
                            ),
                        )
                        for slot in range(current.masks[side].shape[1])
                    ]
                    for side in current.masks
                }
                # The rest of the room (body world, same frame as qpos); only moving parts re-pose.
                scene_handles = []
                scene = current.clip.scene
                if scene is not None:
                    for part, (name, mesh) in enumerate(zip(scene.names, scene.meshes)):
                        shade = 120 + zlib.crc32(name.encode()) % 80
                        handle = server.scene.add_mesh_simple(
                            f"/scene/{name}", mesh.vertices, mesh.faces, color=(shade, shade - 20, shade - 40)
                        )
                        handle.position = scene.T_world_part[part, 0, :3]
                        handle.wxyz = scene.T_world_part[part, 0, 3:]
                        scene_handles.append((part, handle, bool(scene.static[part])))
                frame_slider.value = 0
                frame_slider.max = current.qpos.shape[0] - 1
                status.value = (
                    f"{selection[0]}/{selection[1]}: {current.qpos.shape[0]} frames, "
                    f"objects={current.clip.object_names}"
                )
                built = selection

        # --- viewer: playback (object poses are the qpos tail, one 7-pose per object) ---
        if current is not None:
            frame = min(frame_slider.value, current.qpos.shape[0] - 1)
            if playing.value:
                frame = (frame + 1) % current.qpos.shape[0]
                frame_slider.value = frame
            with server.atomic():
                robot_viz.update_cfg(
                    current.qpos[frame : frame + 1, 7 : 7 + ndof],
                    T_world_base=current.qpos[frame : frame + 1, :7],
                )
                for side, skeleton in hand_viz.items():
                    skeleton.update(mano_world[side][frame])
                    skeleton.visible = show_hand.value
                for obj_i, handle in object_handles:
                    base = 7 + ndof + 7 * obj_i
                    handle.position = current.qpos[frame, base : base + 3]
                    handle.wxyz = current.qpos[frame, base + 3 : base + 7]
                for side, slots in markers.items():
                    for slot, (target, pad) in enumerate(slots):
                        visible = bool(show_contacts.value and current.masks[side][frame, slot])
                        target.visible = visible
                        target.position = current.targets[side][frame, slot]
                        if pad is not None and current.contacts is not None:
                            pad.visible = visible
                            pad.position = current.contacts[side][frame, slot]
                for part, handle, is_static in scene_handles:
                    handle.visible = show_scene.value
                    if not is_static and current.clip.scene is not None:
                        handle.position = current.clip.scene.T_world_part[part, frame, :3]
                        handle.wxyz = current.clip.scene.T_world_part[part, frame, 3:]
        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
