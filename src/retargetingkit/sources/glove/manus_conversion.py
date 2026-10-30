"""Pure Manus glove calibration and MANO conversion."""

from pathlib import Path

import numpy as np

from retargetingkit.sources.frames import HandPose
from robokit.xform.numpy import axis_angle_to_matrix, quaternion_to_matrix


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_JOINT_RANGES = (range(1, 5), range(5, 10), range(10, 15), range(15, 20), range(20, 25))
MANUS_WRIST_INDEX = 0
MANUS_TIP_INDICES = (4, 9, 14, 19, 24)
MANUS_25_TO_MANO_21 = (0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24)
MANUS_SKELETON_EDGES = (
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
FINGERS_UP_ROT = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
MANUS_BENCHMARK_REORIENT_ROTATION = np.array(
    [
        [0.9915023, 0.05378835, 0.11844852],
        [-0.07747657, 0.9755818, 0.20551759],
        [-0.10450177, -0.21294814, 0.9714589],
    ],
    dtype=np.float32,
)


def manus_positions_to_fixed_wrist_local(positions: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """Transform Manus positions to wrist-local, fingers-up coordinates.

    >>> positions = np.zeros((25, 3), dtype=np.float32)
    >>> rotations = np.zeros((25, 4), dtype=np.float32); rotations[:, 3] = 1.0
    >>> manus_positions_to_fixed_wrist_local(positions, rotations).shape
    (25, 3)
    """
    centered = positions.astype(np.float32) - positions[MANUS_WRIST_INDEX]
    wrist_rotation = quaternion_to_matrix(rotations[MANUS_WRIST_INDEX][[3, 0, 1, 2]])  # xyzw -> wxyz
    return (FINGERS_UP_ROT @ wrist_rotation.T @ centered.T).T.astype(np.float32)


def load_glove_calibration(path: Path | None = None) -> dict[str, np.ndarray]:
    """Load Manus glove calibration from an NPZ file.

    >>> load_glove_calibration()  # doctest: +SKIP
    """
    if path is None:
        from retargetingkit.assets import calib

        path = calib.get("manus_calib")
    with np.load(path) as data:
        return {
            "thumb_rotation": data["thumb_rotation"].astype(np.float32),
            "thumb_translation": data["thumb_translation"].astype(np.float32),
            "finger_scales": data["finger_scales"].astype(np.float32),
        }


def apply_glove_corrections(
    wrist_local: np.ndarray,
    thumb_rotation: np.ndarray,
    thumb_translation: np.ndarray,
    finger_scales: np.ndarray,
) -> np.ndarray:
    """Apply the calibrated thumb transform and per-finger scales.

    >>> points = np.zeros((25, 3), dtype=np.float32)
    >>> apply_glove_corrections(points, np.zeros(3), np.zeros(3), np.ones(5)).shape
    (25, 3)
    """
    corrected = wrist_local.copy()
    rotation = axis_angle_to_matrix(thumb_rotation)
    for finger_index, joint_range in enumerate(FINGER_JOINT_RANGES):
        scale = float(finger_scales[finger_index])
        for joint_index in joint_range:
            if finger_index == 0:
                corrected[joint_index] = (
                    scale * (rotation @ wrist_local[joint_index].astype(np.float64)) + thumb_translation
                ).astype(np.float32)
            else:
                corrected[joint_index] = scale * wrist_local[joint_index]
    return corrected


def manus_to_mano_keypoints(
    manus_positions: np.ndarray,
    manus_rotations: np.ndarray,
    glove_calib: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Convert a Manus 25-joint frame to MANO-21 wrist-local keypoints.

    >>> positions = np.zeros((25, 3), dtype=np.float32)
    >>> rotations = np.zeros((25, 4), dtype=np.float32); rotations[:, 3] = 1.0
    >>> manus_to_mano_keypoints(positions, rotations).shape
    (21, 3)
    """
    wrist_local = manus_positions_to_fixed_wrist_local(manus_positions, manus_rotations)
    if glove_calib is not None:
        wrist_local = apply_glove_corrections(
            wrist_local,
            glove_calib["thumb_rotation"],
            glove_calib["thumb_translation"],
            glove_calib["finger_scales"],
        )
    mano = wrist_local[list(MANUS_25_TO_MANO_21)]
    mano = mano - mano[:1]
    if glove_calib is not None:
        mano = mano @ MANUS_BENCHMARK_REORIENT_ROTATION.T
    return mano.astype(np.float32)


def load_mano_keypoints_from_manus_npz(
    recording_path: Path,
    hand: str = "right",
    glove_calib: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Load a Manus recording and convert every frame to MANO keypoints.

    >>> load_mano_keypoints_from_manus_npz(Path("recording.npz"))  # doctest: +SKIP
    """
    with np.load(recording_path, allow_pickle=True) as data:
        positions = data[f"{hand}_positions"].astype(np.float32)
        rotations = data[f"{hand}_rotations"].astype(np.float32)
    return np.stack(
        [
            manus_to_mano_keypoints(position, rotation, glove_calib)
            for position, rotation in zip(positions, rotations, strict=True)
        ]
    )


def hand_pose_from_glove(
    positions: np.ndarray,
    rotations: np.ndarray,
    glove_calib: dict[str, np.ndarray] | None = None,
) -> HandPose:
    """Convert a Manus glove frame to a canonical MANO-21 hand pose.

    >>> positions = np.zeros((25, 3), dtype=np.float32)
    >>> rotations = np.zeros((25, 4), dtype=np.float32); rotations[:, 3] = 1.0
    >>> hand_pose_from_glove(positions, rotations).positions.shape
    (21, 3)
    """
    return HandPose(positions=manus_to_mano_keypoints(positions, rotations, glove_calib))
