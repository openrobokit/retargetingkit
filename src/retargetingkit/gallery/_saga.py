"""Load SAGA FullGraspPose motion into RoboKit's current full-body clip schema."""

from pathlib import Path

import numpy as np
import trimesh

from robokit.assets.body_models import get_model_path
from robokit.helpers.full_body_retargeting.clip import (
    CONTACT_MANO_INDICES,
    FINGERS,
    FullBodyClip,
    HandTrack,
    mano_index,
)
from robokit.smplx import (
    SMPLX_JOINT_NAMES,
    SMPLX_STATIC_LANDMARK_NAMES,
    BodyModelSpecTensors,
    BodyModelState,
    body_lbs_warp,
    load_smplx,
)
from robokit.xform.numpy import axis_angle_to_matrix, matrix_to_quaternion


CONTACT_DISTANCE = 0.025
GRAB_FPS = 120


def _load_saga_clip(
    npz_path: str,
    mesh_dir: Path,
    robot_height: float = 1.32,
    fps: int = 30,
    device: str = "cuda:0",
) -> FullBodyClip:
    data = np.load(npz_path, allow_pickle=True)
    body = data["body"][()]
    gender = str(data["gender"])
    model_npz = get_model_path("smplx", gender)
    spec = load_smplx(str(model_npz), gender)  # type: ignore[arg-type]
    model_raw = np.load(str(model_npz))

    step = GRAB_FPS // fps
    rows = np.arange(0, body["transl"].shape[0], step)
    num_frames = len(rows)
    left_hand = (
        body["left_hand_pose"][rows] @ model_raw["hands_componentsl"][:24].astype(np.float32)
        + spec.metadata["hands_meanl"]
    )
    right_hand = (
        body["right_hand_pose"][rows] @ model_raw["hands_componentsr"][:24].astype(np.float32)
        + spec.metadata["hands_meanr"]
    )
    full_pose = np.concatenate(
        [
            body["global_orient"][rows],
            body["body_pose"][rows],
            body["jaw_pose"][rows],
            body["leye_pose"][rows],
            body["reye_pose"][rows],
            left_hand,
            right_hand,
        ],
        axis=1,
    ).astype(np.float32)
    state = BodyModelState(
        betas=np.repeat(data["betas"], num_frames, axis=0).astype(np.float32),
        full_pose_aa=full_pose.reshape(num_frames, 55, 3),
        transl=body["transl"][rows].astype(np.float32),
    )
    tensors = BodyModelSpecTensors(spec=spec, device=device)
    output = body_lbs_warp(tensors, state, return_landmarks=True)
    T_native_joint = output["T_world_joint"].numpy()
    landmarks = output["landmarks"].numpy()

    translation = data["transf_transl"][rows].astype(np.float32)
    joints = T_native_joint[..., :3] + translation[:, None]
    landmarks = landmarks + translation[:, None]
    object_vertices = data["verts_object"][rows].astype(np.float32)
    object_rotation = axis_angle_to_matrix(-data["global_orient_object"][rows]).astype(np.float32)
    object_translation = object_vertices.mean(axis=1) + translation
    object_name = Path(npz_path).name.split("_")[0]
    object_mesh = trimesh.load(str(mesh_dir / f"{object_name}.obj"), force="mesh")
    assert isinstance(object_mesh, trimesh.Trimesh)

    feet = [
        SMPLX_STATIC_LANDMARK_NAMES.index(name) for name in ("left_heel", "right_heel", "left_big_toe", "right_big_toe")
    ]
    floor_z = float(landmarks[:, feet, 2].min())
    tpose = BodyModelState(
        betas=data["betas"].astype(np.float32),
        full_pose_aa=np.zeros((1, 55, 3), dtype=np.float32),
        transl=np.zeros((1, 3), dtype=np.float32),
    )
    tpose_vertices = body_lbs_warp(tensors, tpose, return_landmarks=False)["vertices"].numpy()[0]
    human_height = float(tpose_vertices[:, 1].max() - tpose_vertices[:, 1].min())
    scale = robot_height / human_height

    T_native_object = np.concatenate([object_translation, matrix_to_quaternion(object_rotation)], axis=1).astype(
        np.float32
    )
    T_world_object = T_native_object.copy()
    T_world_object[:, :3] = (object_translation - [0.0, 0.0, floor_z]) * scale

    joint_indices, tip_indices = mano_index("right")
    mano = np.stack(
        [landmarks[:, tip] if joint < 0 else joints[:, joint] for joint, tip in zip(joint_indices, tip_indices)],
        axis=1,
    )
    vertices_world = object_vertices + translation[:, None]
    labels = data["contact_object"][rows]
    finger_label_ids = np.asarray(
        [[SMPLX_JOINT_NAMES.index(f"right_{finger}{joint}") + 1 for joint in (1, 2, 3)] for finger in FINGERS],
        dtype=labels.dtype,
    )
    contact_points = np.zeros((num_frames, len(CONTACT_MANO_INDICES), 3), dtype=np.float32)
    contact_normals = np.zeros_like(contact_points)
    contact_mask = np.zeros((num_frames, len(CONTACT_MANO_INDICES)), dtype=np.bool_)
    frames = np.arange(num_frames)
    for slot, mano_idx in enumerate(CONTACT_MANO_INDICES):
        labelled = np.isin(labels, finger_label_ids[slot // 2])
        deltas = vertices_world - mano[:, mano_idx, None]
        distance2 = np.einsum("fvi,fvi->fv", deltas, deltas)
        distance2[~labelled] = np.inf
        nearest = distance2.argmin(axis=1)
        valid = labelled.any(axis=1) & (np.linalg.norm(deltas[frames, nearest], axis=1) < CONTACT_DISTANCE)
        contact_points[valid, slot] = vertices_world[frames[valid], nearest[valid]]
        contact_mask[:, slot] = valid

    distance = np.linalg.norm(mano[:, CONTACT_MANO_INDICES] - contact_points, axis=-1)
    for slot in range(0, len(CONTACT_MANO_INDICES), 2):
        both = contact_mask[:, slot] & contact_mask[:, slot + 1]
        if both.any():
            first = np.median(distance[contact_mask[:, slot], slot])
            second = np.median(distance[contact_mask[:, slot + 1], slot + 1])
            contact_mask[both, slot + 1 if first <= second else slot] = False
    if contact_mask.any():
        local = np.einsum("fji,fsj->fsi", object_rotation, contact_points - object_translation[:, None])
        closest, _, triangles = trimesh.proximity.closest_point(object_mesh, local[contact_mask])
        contact_frames = np.nonzero(contact_mask)[0]
        contact_points[contact_mask] = (
            np.einsum("fij,fj->fi", object_rotation[contact_frames], closest) + object_translation[contact_frames]
        )
        contact_normals[contact_mask] = np.einsum(
            "fij,fj->fi", object_rotation[contact_frames], object_mesh.face_normals[triangles]
        )

    human_positions = joints - [0.0, 0.0, floor_z]
    wrist = SMPLX_JOINT_NAMES.index("right_wrist")
    return FullBodyClip(
        T_native_human=np.concatenate([human_positions, T_native_joint[..., 3:]], axis=-1).astype(np.float32),
        human_height=human_height,
        scale=scale,
        fps=fps,
        hands={
            "right": HandTrack(
                mano=mano.astype(np.float32),
                q_wxyz_world_wrist=T_native_joint[:, wrist, 3:].astype(np.float32),
                object_name=object_name,
                object_mesh=object_mesh,
                T_native_object=T_native_object,
                T_world_object=T_world_object,
                contact_points=contact_points,
                contact_normals=contact_normals,
                contact_mask=contact_mask,
            )
        },
    )
