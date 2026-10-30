"""Capture Manus motion for generated YCB contact-supervision grasp specs.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run --extra contact --extra manus --extra vis \
        scripts/teleop/neural_hand/contact_capture.py --output training_data.npz
"""

import argparse
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from retargetingkit.assets import calib
from retargetingkit.retargeting.hand.neural import select_mano_keypoints
from retargetingkit.retargeting.hand.presets import HANDS
from retargetingkit.sources.glove._manus_sdk import ManusClient, resolve_manus_sdk_library_path
from retargetingkit.sources.glove.manus_conversion import (
    FINGER_JOINT_RANGES,
    FINGERS_UP_ROT,
    MANUS_WRIST_INDEX,
    manus_positions_to_fixed_wrist_local,
    manus_to_mano_keypoints,
)
from retargetingkit.sources.glove.manus_glove import DualManusGloveAggregator, parse_glove_id
from retargetingkit.training.hand.contact import (
    N_FINGERS,
    YcbObjectModel,
    fabrik_reach,
    fit_object_translation,
    load_ycb_mesh,
    load_ycb_object,
)
from retargetingkit.training.hand.data import ContactCaptureData, GraspSpecData
from retargetingkit.utils.viz import (
    CONTACT_COLOR,
    add_contact_spheres,
    add_hand_urdf,
    add_object_nodes,
    draw_manus_skeleton,
)
from robokit.xform.numpy import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    matrix_to_quaternion,
    quaternion_to_matrix,
)


DEFAULT_CONFIG = Path(__file__).parent / "configs/contact/sharpa_ycb_contact.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture Manus motion for YCB contact-supervision specs.")
    parser.add_argument("--specs", type=Path, default=calib.get("sharpa_contact_grasp_ycb"))
    parser.add_argument("--output", type=Path, default=Path("training_data.npz"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hand", choices=["left", "right"], default="right")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--snap-in", type=float, default=0.01)
    parser.add_argument("--release", type=float, default=0.02)
    parser.add_argument("--fabrik-iterations", type=int, default=50)
    parser.add_argument("--manus-right-id")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    human_keypoint_indices = HANDS["sharpa"].human_keypoint_indices
    specs = GraspSpecData.load(args.specs)
    names = [str(name) for name in specs.object_names]
    masks = specs.contact_mask.astype(np.float32)
    grasp_penetration = float(config["grasp_penetration"])
    print(f"specs={len(specs)} grasp_penetration={grasp_penetration * 1000.0:.1f}mm", flush=True)

    device = torch.device(args.device)
    object_cache: dict[str, YcbObjectModel] = {}

    def get_object(name: str) -> YcbObjectModel:
        if name not in object_cache:
            object_cache[name] = load_ycb_object(
                name,
                device,
                float(config["sdf_voxel_size"]),
                float(config["sdf_padding"]),
            )
        return object_cache[name]

    import viser

    server = viser.ViserServer()
    server.scene.add_grid("/grid", 1.0, 1.0)
    side_offset = np.array([0.0, -0.15, 0.0], dtype=np.float32)
    server.scene.add_frame("/ref", position=(0.0, 0.15, 0.0), show_axes=False)
    robot_urdf, viser_joint_indices = add_hand_urdf(server, HANDS["sharpa"], "/ref/robot")
    side_rotation = axis_angle_to_matrix(np.array([0.0, 0.0, np.pi / 2.0], dtype=np.float32)).astype(np.float32)
    meshes = {}
    for name in sorted(set(names)):
        mesh = load_ycb_mesh(name)
        meshes[name] = (np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int32))
    object_nodes = add_object_nodes(server, "", meshes)
    side_object_nodes = add_object_nodes(server, "/side", meshes)
    reference_object_nodes = add_object_nodes(server, "/ref", meshes)
    contact_nodes = add_contact_spheres(server, "")
    side_contact_nodes = add_contact_spheres(server, "/side")
    reference_contact_nodes = add_contact_spheres(server, "/ref")
    spec_slider = server.gui.add_slider("Spec", min=0, max=len(specs) - 1, step=1, initial_value=0)

    client = ManusClient(resolve_manus_sdk_library_path())
    aggregator = DualManusGloveAggregator(right_override=parse_glove_id(args.manus_right_id))
    client.start(aggregator.update)
    print("Manus SDK client started; waiting for glove.", flush=True)

    def get_frame() -> tuple[np.ndarray, np.ndarray] | None:
        frame = aggregator.latest()[1] if args.hand == "right" else aggregator.latest()[0]
        return (frame.positions, frame.rotations) if frame is not None else None

    record_button = server.gui.add_button("Record sample")
    skip_button = server.gui.add_button("Skip spec")
    save_button = server.gui.add_button("Save contact-supervision npz")
    status = server.gui.add_text("Status", initial_value="", disabled=True)
    snap_toggle = server.gui.add_checkbox("Magnetic snap", initial_value=True)
    buffer: dict[str, list[Any]] = {
        key: [] for key in ("positions", "rotations", "qpos", "translation", "rotvec", "mask", "contact", "names")
    }
    if args.output.exists():
        old = ContactCaptureData.load(args.output)
        buffer["positions"] = list(old.positions)
        buffer["rotations"] = list(old.rotations)
        buffer["qpos"] = list(old.grasps.qpos)
        buffer["translation"] = list(old.grasps.object_translation)
        buffer["rotvec"] = list(old.grasps.object_rotvec)
        buffer["mask"] = list(old.grasps.contact_mask)
        buffer["contact"] = list(old.grasps.contact_points)
        buffer["names"] = [str(name) for name in old.grasps.object_names]
        print(f"resuming={args.output} samples={len(old)}", flush=True)

    state: dict[str, Any] = {
        "spec": -1,
        "object_translation": torch.zeros((1, 3), device=device),
        "live": None,
        "latched": np.zeros(N_FINGERS, dtype=bool),
    }

    def record_current_sample(advance: bool = False) -> None:
        live = state["live"]
        assert live is not None
        for key, value in zip(buffer, live, strict=True):
            buffer[key].append(value)
        if advance:
            spec_slider.value = min(int(spec_slider.value) + 1, len(specs) - 1)
        print(f"recorded={len(buffer['positions'])} object={live[-1]}", flush=True)

    @record_button.on_click
    def _record(_event: Any) -> None:
        record_current_sample()

    @skip_button.on_click
    def _skip(_event: Any) -> None:
        spec_slider.value = min(int(spec_slider.value) + 1, len(specs) - 1)

    @save_button.on_click
    def _save(_event: Any) -> None:
        assert buffer["positions"]
        grasps = GraspSpecData(
            qpos=np.stack(buffer["qpos"]).astype(np.float32),
            object_translation=np.stack(buffer["translation"]).astype(np.float32),
            object_rotvec=np.stack(buffer["rotvec"]).astype(np.float32),
            contact_mask=np.stack(buffer["mask"]).astype(bool),
            contact_points=np.stack(buffer["contact"]).astype(np.float32),
            object_names=np.asarray(buffer["names"]),
        )
        ContactCaptureData(
            grasps,
            np.stack(buffer["positions"]).astype(np.float32),
            np.stack(buffer["rotations"]).astype(np.float32),
        ).save(args.output)
        print(f"saved={args.output} samples={len(grasps)}", flush=True)

    def load_spec(index: int) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        name = names[index]
        rotation = torch.from_numpy(axis_angle_to_matrix(specs.object_rotvec[index]).astype(np.float32)).to(device)
        contact_offset = torch.from_numpy(specs.contact_points[index] - specs.object_translation[index]).to(device)
        mask = torch.from_numpy(masks[index]).to(device)
        state["object_translation"] = torch.zeros((1, 3), device=device)
        state["latched"][:] = False
        get_object(name)
        robot_urdf.update_cfg(np.asarray([specs.qpos[index, joint] for joint in viser_joint_indices], dtype=np.float32))
        object_wxyz = tuple(axis_angle_to_quaternion(specs.object_rotvec[index]).tolist())
        for other in object_nodes:
            visible = other == name
            object_nodes[other].visible = visible
            side_object_nodes[other].visible = visible
            reference_object_nodes[other].visible = visible
        reference_object_nodes[name].position = tuple(specs.object_translation[index].tolist())
        reference_object_nodes[name].wxyz = object_wxyz
        for finger in range(N_FINGERS):
            reference_contact_nodes[finger].visible = bool(masks[index, finger])
            side_contact_nodes[finger].visible = bool(masks[index, finger])
            reference_contact_nodes[finger].position = tuple(specs.contact_points[index, finger].tolist())
        return name, rotation.unsqueeze(0), contact_offset, mask

    def loop() -> None:
        period = 1.0 / args.rate
        name, object_rotation, contact_offset, mask = load_spec(0)
        state["spec"] = 0
        last_print = 0.0
        while True:
            start = time.monotonic()
            spec_index = int(spec_slider.value)
            if spec_index != state["spec"]:
                name, object_rotation, contact_offset, mask = load_spec(spec_index)
                state["spec"] = spec_index
            frame = get_frame()
            if frame is None:
                time.sleep(period)
                continue
            positions, rotations = frame
            mano_np = manus_to_mano_keypoints(positions, rotations)
            mano = torch.from_numpy(mano_np).to(device)
            tips = select_mano_keypoints(mano_np, human_keypoint_indices)
            object_translation, gap, targets = fit_object_translation(
                get_object(name),
                mano[list(human_keypoint_indices)],
                contact_offset,
                mask,
                object_rotation,
                state["object_translation"],
                mano.unsqueeze(0),
                0.0,
                grasp_penetration,
                args.iterations,
                args.learning_rate,
            )
            state["object_translation"] = object_translation
            translation = object_translation[0].cpu().numpy().astype(np.float32)
            object_wxyz = tuple(axis_angle_to_quaternion(specs.object_rotvec[spec_index]).tolist())
            side_wxyz = tuple(
                matrix_to_quaternion(side_rotation @ axis_angle_to_matrix(specs.object_rotvec[spec_index])).tolist()
            )
            wrist_local = manus_positions_to_fixed_wrist_local(positions, rotations)
            recorded_positions = positions.copy()
            contacted = np.zeros(N_FINGERS, dtype=bool)
            if snap_toggle.value:
                back_rotation = quaternion_to_matrix(rotations[MANUS_WRIST_INDEX][[3, 0, 1, 2]]) @ FINGERS_UP_ROT.T
                for finger in range(N_FINGERS):
                    if not masks[spec_index, finger]:
                        state["latched"][finger] = False
                        continue
                    distance = float(np.linalg.norm(tips[finger] - targets[finger]))
                    state["latched"][finger] = (
                        distance < args.snap_in if not state["latched"][finger] else distance <= args.release
                    )
                    if state["latched"][finger]:
                        chain = list(FINGER_JOINT_RANGES[finger])
                        wrist_local[chain] = fabrik_reach(wrist_local[chain], targets[finger], args.fabrik_iterations)
                        recorded_positions[chain] = positions[MANUS_WRIST_INDEX] + wrist_local[chain] @ back_rotation.T
                        contacted[finger] = True
            else:
                for finger in range(N_FINGERS):
                    contacted[finger] = (
                        bool(masks[spec_index, finger]) and np.linalg.norm(tips[finger] - targets[finger]) < 0.005
                    )

            with server.atomic():
                draw_manus_skeleton(server, wrist_local, "/manus")
                draw_manus_skeleton(server, wrist_local @ side_rotation.T + side_offset, "/manus_side")
                object_nodes[name].position = (
                    float(translation[0]),
                    float(translation[1]),
                    float(translation[2]),
                )
                object_nodes[name].wxyz = object_wxyz
                side_object_nodes[name].position = tuple((translation @ side_rotation.T + side_offset).tolist())
                side_object_nodes[name].wxyz = side_wxyz
                for finger in range(N_FINGERS):
                    contact_nodes[finger].visible = bool(masks[spec_index, finger])
                    contact_nodes[finger].position = tuple(targets[finger].tolist())
                    side_contact_nodes[finger].position = tuple(
                        (targets[finger] @ side_rotation.T + side_offset).tolist()
                    )
                    color = (80, 220, 120) if contacted[finger] else CONTACT_COLOR
                    contact_nodes[finger].color = color
                    side_contact_nodes[finger].color = color
                contact_fingers = [finger for finger in range(N_FINGERS) if masks[spec_index, finger]]
                fingers = (contact_fingers + [contact_fingers[0]] * N_FINGERS)[:N_FINGERS]
                side_tips = tips @ side_rotation.T + side_offset
                side_targets = targets @ side_rotation.T + side_offset
                segments = np.stack([[tips[finger], targets[finger]] for finger in fingers]).astype(np.float32)
                side_segments = np.stack([[side_tips[finger], side_targets[finger]] for finger in fingers]).astype(
                    np.float32
                )
                server.scene.add_line_segments(
                    "/links", segments, (40, 120, 255), thickness=3.0, thickness_units="screen"
                )
                server.scene.add_line_segments(
                    "/side/links", side_segments, (40, 120, 255), thickness=3.0, thickness_units="screen"
                )

            grip = "".join("TIMRP"[finger] for finger in range(N_FINGERS) if masks[spec_index, finger])
            state["live"] = (
                recorded_positions,
                rotations.copy(),
                specs.qpos[spec_index].copy(),
                specs.object_translation[spec_index].copy(),
                specs.object_rotvec[spec_index].copy(),
                specs.contact_mask[spec_index].copy(),
                specs.contact_points[spec_index].copy(),
                name,
            )
            status.value = (
                f"spec={spec_index}/{len(specs) - 1} object={name} grip={grip} "
                f"gap={gap * 1000.0:.1f}mm recorded={len(buffer['positions'])}"
            )
            if time.monotonic() - last_print >= 2.0:
                print(status.value, flush=True)
                last_print = time.monotonic()
            time.sleep(max(0.0, period - (time.monotonic() - start)))

    threading.Thread(target=loop, daemon=True).start()
    print(f"viser={server.get_port()}", flush=True)
    print("Press Enter to record and advance.", flush=True)
    while True:
        input()
        record_current_sample(advance=True)


if __name__ == "__main__":
    main()
