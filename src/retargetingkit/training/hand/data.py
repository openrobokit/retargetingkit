"""Canonical neural-retargeting datasets, file schemas, and loaders."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from retargetingkit.retargeting.hand.neural import normalize_joints, select_mano_keypoints
from retargetingkit.retargeting.hand.presets import HandPreset
from retargetingkit.sources.glove.manus_conversion import load_mano_keypoints_from_manus_npz


@dataclass(frozen=True)
class PairSource:
    """Measured human/robot pose source with per-frame finger masks."""

    path: Path
    weight: float
    finger_mask: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class ContactSource:
    """Captured contact-alignment source with source-level weighting."""

    path: Path
    weight: float
    contact_only: bool


@dataclass(frozen=True)
class DataConfig:
    """Resolved training data and sampling configuration."""

    unsupervised_paths: tuple[Path, ...]
    pair_sources: tuple[PairSource, ...]
    contact_sources: tuple[ContactSource, ...]
    training_samples: int
    voxel_size_m: float


class HumanDataset(Dataset[torch.Tensor]):
    """SGD samples plus full frames used for human workspace support."""

    def __init__(self, samples: np.ndarray, support_frames: np.ndarray) -> None:
        self.samples = torch.from_numpy(samples.astype(np.float32))
        self.support_frames = torch.from_numpy(support_frames.astype(np.float32))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.samples[index]


@dataclass(frozen=True)
class JointSupervision:
    """Canonical masked joint targets shared by pair and contact supervision."""

    keypoints: torch.Tensor
    target_qpos: torch.Tensor
    joint_mask: torch.Tensor
    sample_weight: torch.Tensor

    def to(self, device: torch.device) -> "JointSupervision":
        """Move every supervision tensor to `device`."""
        return JointSupervision(
            self.keypoints.to(device),
            self.target_qpos.to(device),
            self.joint_mask.to(device),
            self.sample_weight.to(device),
        )


@dataclass(frozen=True)
class TrainingData:
    """All inputs consumed by one training run."""

    human: HumanDataset
    pair: JointSupervision
    contact: JointSupervision


@dataclass
class GraspSpecData:
    """Generated robot grasp specification stored before human capture."""

    qpos: np.ndarray
    object_translation: np.ndarray
    object_rotvec: np.ndarray
    contact_mask: np.ndarray
    contact_points: np.ndarray
    object_names: np.ndarray

    def __len__(self) -> int:
        return len(self.qpos)

    @classmethod
    def load(cls, path: str | Path) -> "GraspSpecData":
        """Load a generated grasp specification NPZ."""
        with np.load(Path(path), allow_pickle=False) as raw:
            return cls(
                qpos=raw["grasp_q"].astype(np.float32),
                object_translation=raw["object_t_init"].astype(np.float32),
                object_rotvec=raw["object_rotvec_fixed"].astype(np.float32),
                contact_mask=raw["contact_finger_mask"].astype(bool),
                contact_points=raw["contact_target_points"].astype(np.float32),
                object_names=raw["object_names"].astype(str),
            )

    def fields(self) -> dict[str, np.ndarray]:
        """Return the canonical flat NPZ fields."""
        return {
            "grasp_q": self.qpos.astype(np.float32),
            "object_t_init": self.object_translation.astype(np.float32),
            "object_rotvec_fixed": self.object_rotvec.astype(np.float32),
            "contact_finger_mask": self.contact_mask.astype(bool),
            "contact_target_points": self.contact_points.astype(np.float32),
            "object_names": self.object_names.astype(str),
        }

    def save(self, path: str | Path) -> None:
        """Save a generated grasp specification NPZ."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = self.fields()
        np.savez(
            path,
            grasp_q=fields["grasp_q"],
            object_t_init=fields["object_t_init"],
            object_rotvec_fixed=fields["object_rotvec_fixed"],
            contact_finger_mask=fields["contact_finger_mask"],
            contact_target_points=fields["contact_target_points"],
            object_names=fields["object_names"],
        )


@dataclass
class ContactCaptureData:
    """Captured Manus motion paired with generated robot grasp specs."""

    grasps: GraspSpecData
    positions: np.ndarray
    rotations: np.ndarray

    def __len__(self) -> int:
        return len(self.grasps)

    @classmethod
    def load(cls, path: str | Path) -> "ContactCaptureData":
        """Load a captured contact-alignment NPZ."""
        path = Path(path)
        grasps = GraspSpecData.load(path)
        with np.load(path, allow_pickle=False) as raw:
            positions = raw["right_positions"].astype(np.float32)
            rotations = raw["right_rotations"].astype(np.float32)
        assert len(grasps) == len(positions) == len(rotations)
        return cls(grasps, positions, rotations)

    def save(self, path: str | Path) -> None:
        """Save captured contact-alignment data."""
        assert len(self.grasps) == len(self.positions) == len(self.rotations)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = self.grasps.fields()
        np.savez(
            path,
            grasp_q=fields["grasp_q"],
            object_t_init=fields["object_t_init"],
            object_rotvec_fixed=fields["object_rotvec_fixed"],
            contact_finger_mask=fields["contact_finger_mask"],
            contact_target_points=fields["contact_target_points"],
            object_names=fields["object_names"],
            right_timestamps=np.arange(len(self), dtype=np.float64),
            right_positions=self.positions.astype(np.float32),
            right_rotations=self.rotations.astype(np.float32),
        )


def load_human_dataset(
    paths: Sequence[Path],
    spec: HandPreset,
    training_samples: int,
    voxel_size_m: float,
) -> HumanDataset:
    """Load full support frames and resample the fixed SGD dataset.

    >>> load_human_dataset(paths, spec, 20_000, 0.003)  # doctest: +SKIP
    """
    support_frames = load_human_frames(paths, spec)
    voxel_keys = np.round(support_frames.reshape(len(support_frames), -1) / voxel_size_m).astype(np.int64)
    _, unique_indices = np.unique(voxel_keys, axis=0, return_index=True)
    reduced = support_frames[np.sort(unique_indices)]
    samples = reduced[np.random.randint(0, len(reduced), training_samples)]
    print(f"Combined unsupervised data: frames={len(support_frames)}  samples={len(samples)}")
    return HumanDataset(samples, support_frames)


def load_human_frames(paths: Sequence[Path], spec: HandPreset) -> np.ndarray:
    """Load every human frame before deduplication or resampling.

    >>> load_human_frames(paths, spec).shape  # doctest: +SKIP
    """
    sequences: list[np.ndarray] = []
    for path in paths:
        if path.suffix == ".npy":
            raw = np.load(path).astype(np.float32)
            mano = raw - raw[:, :1]
        else:
            mano = load_mano_keypoints_from_manus_npz(path, hand="right")
        frames = select_mano_keypoints(mano, spec.human_keypoint_indices)
        sequences.append(frames)
        print(f"Loaded unsupervised asset: {path.name}  frames={len(frames)}")
    return np.concatenate(sequences).astype(np.float32)


def load_pair_supervision(sources: Sequence[PairSource], spec: HandPreset) -> JointSupervision:
    """Load measured pair-pose supervision.

    >>> load_pair_supervision(sources, spec).target_qpos.shape  # doctest: +SKIP
    """
    keypoints: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    lower, upper = torch.from_numpy(spec.joint_lower), torch.from_numpy(spec.joint_upper)
    for source in sources:
        mano = load_mano_keypoints_from_manus_npz(source.path, hand="right")
        with np.load(source.path) as raw:
            qpos = raw["qpos"].astype(np.float32)
        assert qpos.shape == (len(mano), spec.joint_count)
        finger_mask = np.asarray(source.finger_mask, dtype=np.float32)
        assert finger_mask.shape in ((1, len(spec.finger_joint_names)), (len(mano), len(spec.finger_joint_names)))
        finger_mask = np.broadcast_to(finger_mask, (len(mano), len(spec.finger_joint_names)))
        keypoints.append(select_mano_keypoints(mano, spec.human_keypoint_indices))
        targets.append(normalize_joints(torch.from_numpy(qpos), lower, upper).numpy())
        masks.append(joint_mask(finger_mask, spec))
        weights.append(np.full(len(mano), source.weight, dtype=np.float32))
        print(f"Loaded pair-pose asset: {source.path.name}  frames={len(mano)}")
    return joint_supervision(keypoints, targets, masks, weights, spec)


def load_contact_supervision(sources: Sequence[ContactSource], spec: HandPreset) -> JointSupervision:
    """Load captured contact-alignment supervision.

    >>> load_contact_supervision(sources, spec).sample_weight.shape  # doctest: +SKIP
    """
    keypoints: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    lower, upper = torch.from_numpy(spec.joint_lower), torch.from_numpy(spec.joint_upper)
    for source in sources:
        capture = ContactCaptureData.load(source.path)
        mano = load_mano_keypoints_from_manus_npz(source.path, hand="right")
        qpos = capture.grasps.qpos
        finger_mask = capture.grasps.contact_mask.astype(np.float32)
        assert qpos.shape == (len(mano), spec.joint_count)
        assert finger_mask.shape == (len(mano), len(spec.finger_joint_names))
        assert bool(np.all(qpos >= spec.joint_lower))
        assert bool(np.all(qpos <= spec.joint_upper))
        keypoints.append(select_mano_keypoints(mano, spec.human_keypoint_indices))
        targets.append(normalize_joints(torch.from_numpy(qpos), lower, upper).numpy())
        masks.append(
            joint_mask(finger_mask, spec)
            if source.contact_only
            else np.ones((len(mano), spec.joint_count), dtype=np.float32)
        )
        weights.append(np.full(len(mano), source.weight / len(mano), dtype=np.float32))
        print(
            f"Loaded contact-alignment asset: {source.path.parent.name}/{source.path.name}  "
            f"frames={len(mano)} source_weight={source.weight:g} contact_only={source.contact_only}"
        )
    return joint_supervision(keypoints, targets, masks, weights, spec)


def joint_mask(finger_mask: np.ndarray, spec: HandPreset) -> np.ndarray:
    """Expand a per-finger mask to one column per actuated joint.

    >>> joint_mask(np.ones((2, 5), np.float32), spec).shape  # doctest: +SKIP
    """
    mask = np.zeros((len(finger_mask), spec.joint_count), dtype=np.float32)
    start = 0
    for finger, names in enumerate(spec.finger_joint_names):
        mask[:, start : start + len(names)] = finger_mask[:, finger : finger + 1]
        start += len(names)
    return mask


def joint_supervision(
    keypoints: list[np.ndarray],
    targets: list[np.ndarray],
    masks: list[np.ndarray],
    weights: list[np.ndarray],
    spec: HandPreset,
) -> JointSupervision:
    """Concatenate loaded supervision arrays, or return canonical empty shapes.

    >>> joint_supervision([], [], [], [], spec).keypoints.shape  # doctest: +SKIP
    """
    if not keypoints:
        fingers = len(spec.human_keypoint_indices)
        return JointSupervision(
            torch.empty((0, fingers, 3)),
            torch.empty((0, spec.joint_count)),
            torch.empty((0, spec.joint_count)),
            torch.empty(0),
        )
    return JointSupervision(
        torch.from_numpy(np.concatenate(keypoints)),
        torch.from_numpy(np.concatenate(targets)),
        torch.from_numpy(np.concatenate(masks)),
        torch.from_numpy(np.concatenate(weights)),
    )


__all__ = [
    "DataConfig",
    "ContactCaptureData",
    "ContactSource",
    "GraspSpecData",
    "HumanDataset",
    "JointSupervision",
    "PairSource",
    "TrainingData",
    "joint_mask",
    "joint_supervision",
    "load_contact_supervision",
    "load_human_dataset",
    "load_human_frames",
    "load_pair_supervision",
]
