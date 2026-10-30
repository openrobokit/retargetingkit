"""Config-driven neural hand training and atomic checkpoint production."""

import hashlib
import os
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn
from torch.utils.data import DataLoader

from retargetingkit.assets import ASSETS_DIR, REVISION, calib
from retargetingkit.retargeting.hand.neural import IKModel
from retargetingkit.retargeting.hand.presets import HANDS, HandPreset
from retargetingkit.training.hand.data import (
    ContactSource,
    DataConfig,
    PairSource,
    TrainingData,
    load_contact_supervision,
    load_human_dataset,
    load_pair_supervision,
)
from retargetingkit.training.hand.kinematics import (
    load_hand_robot,
    make_collision_residual,
    robot_keypoints,
    sample_robot_workspace,
)
from retargetingkit.training.hand.metrics import COLLISION_MARGIN_M, build_motion_workspace
from retargetingkit.training.hand.objective import (
    LossTerms,
    ObjectiveConfig,
    ObjectiveContext,
    compute_objective,
)


@dataclass(frozen=True)
class OptimizationConfig:
    """Optimizer and schedule configuration."""

    epochs: int
    batch_size: int
    learning_rate: float
    schedule: Literal["const", "cosine"]
    warmup_epochs: int
    ema_decay: float
    seed: int


@dataclass(frozen=True)
class TrainConfig:
    """Fully resolved neural training configuration."""

    hand: str
    output: Path
    robot_samples: int
    data: DataConfig
    objective: ObjectiveConfig
    optimization: OptimizationConfig

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainConfig":
        """Resolve one canonical YAML config and its named assets."""
        raw = yaml.safe_load(path.read_text())
        hand = str(raw["hand"])
        spec = HANDS[hand]
        data = raw["data"]
        pair_sources = tuple(
            PairSource(
                path=calib.get(str(entry["asset"])),
                weight=float(entry["source_weight"]),
                finger_mask=tuple(tuple(float(value) for value in row) for row in entry["finger_mask"]),
            )
            for entry in data["pair"]
        )
        assert all(len(row) == len(spec.finger_joint_names) for source in pair_sources for row in source.finger_mask)
        contact_sources = tuple(
            ContactSource(
                path=calib.get(str(entry["asset"])),
                weight=float(entry["source_weight"]),
                contact_only=bool(entry["contact_only"]),
            )
            for entry in data["contact"]
        )
        objective = ObjectiveConfig.from_dict(raw["objective"])
        assert not objective.pair or pair_sources
        assert not objective.pair_position or pair_sources
        assert not objective.contact or contact_sources
        assert not objective.collision or spec.collision_spheres_path is not None
        return cls(
            hand=hand,
            output=Path(raw["output"]),
            robot_samples=int(raw["robot_samples"]),
            data=DataConfig(
                unsupervised_paths=tuple(calib.get(str(name)) for name in data["unsupervised"]),
                pair_sources=pair_sources,
                contact_sources=contact_sources,
                training_samples=int(data["training_samples"]),
                voxel_size_m=float(data["voxel_size_m"]),
            ),
            objective=objective,
            optimization=OptimizationConfig(**raw["optimization"]),
        )


def load_training_data(config: DataConfig, spec: HandPreset) -> TrainingData:
    """Load every training data role against one resolved hand.

    >>> load_training_data(config, spec)  # doctest: +SKIP
    """
    return TrainingData(
        human=load_human_dataset(
            config.unsupervised_paths,
            spec,
            config.training_samples,
            config.voxel_size_m,
        ),
        pair=load_pair_supervision(config.pair_sources, spec),
        contact=load_contact_supervision(config.contact_sources, spec),
    )


def train_from_config(config_path: Path, device: str | torch.device) -> Path:
    """Train one config into a new atomic checkpoint directory.

    >>> train_from_config(Path("train.yaml"), "cuda")  # doctest: +SKIP
    """
    config = TrainConfig.from_yaml(config_path)
    spec = HANDS[config.hand]
    assert not config.output.exists()
    temporary_output = config.output.with_name(f".{config.output.name}.tmp")
    assert not temporary_output.exists()
    config.output.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.optimization.seed)
    np.random.seed(config.optimization.seed)
    torch_device = torch.device(device)
    generator = torch.Generator(device=torch_device).manual_seed(config.optimization.seed)
    loader_generator = torch.Generator().manual_seed(config.optimization.seed)
    print(f"Device: {torch_device}")

    robot = load_hand_robot(spec)
    joint_limits = torch.from_numpy(robot.spec.actuated_joint_limits).float().to(torch_device)
    fingertip_link_indices = tuple(robot.spec.link_names.index(link) for link in spec.fingertip_links)
    data = load_training_data(config.data, spec)
    model = IKModel(spec.finger_joint_sizes).to(torch_device)
    robot_cloud = (
        sample_robot_workspace(robot, fingertip_link_indices, joint_limits, config.robot_samples)
        .transpose(0, 1)
        .to(torch_device)
    )
    motion_workspace = (
        build_motion_workspace(
            robot_cloud,
            data.human.support_frames.to(torch_device),
            config.objective.motion.directions,
            generator,
        )
        if config.objective.motion.formula == "motion_alignment"
        else None
    )
    if motion_workspace is not None:
        print(f"Motion workspace: robot_samples={robot_cloud.shape[1]} directions={len(motion_workspace.directions)}")
    finger_pairs = tuple(
        (left, right)
        for left in range(len(spec.finger_joint_names))
        for right in range(left + 1, len(spec.finger_joint_names))
    )
    context = ObjectiveContext(
        robot_keypoints=partial(robot_keypoints, robot, fingertip_link_indices, joint_limits),
        motion_workspace=motion_workspace,
        finger_joint_sizes=spec.finger_joint_sizes,
        finger_pairs=finger_pairs,
        collision_residual=(
            make_collision_residual(
                robot,
                joint_limits,
                spec.collision_max_active_pairs,
                COLLISION_MARGIN_M,
            )
            if config.objective.collision
            else None
        ),
        generator=generator,
    )
    pair = data.pair.to(torch_device)
    contact = data.contact.to(torch_device)
    loader = DataLoader(
        data.human,
        batch_size=config.optimization.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    optimizer = optim.AdamW(model.parameters(), lr=config.optimization.learning_rate)
    schedule = SequentialLR(
        optimizer,
        [
            LinearLR(
                optimizer,
                start_factor=1.0 / max(config.optimization.warmup_epochs, 1),
                total_iters=max(config.optimization.warmup_epochs - 1, 1),
            ),
            CosineAnnealingLR(optimizer, T_max=config.optimization.epochs - config.optimization.warmup_epochs),
        ],
        milestones=[config.optimization.warmup_epochs],
    )
    averaged = (
        AveragedModel(model, avg_fn=get_ema_avg_fn(config.optimization.ema_decay))
        if config.optimization.ema_decay
        else None
    )
    model.train()
    for epoch in range(config.optimization.epochs):
        totals = {field.name: 0.0 for field in fields(LossTerms)}
        for human_points in loader:
            terms = compute_objective(
                model,
                human_points.to(torch_device),
                pair,
                contact,
                context,
                config.objective,
            )
            optimizer.zero_grad()
            terms.weighted_sum(config.objective).backward()
            optimizer.step()
            if averaged is not None:
                averaged.update_parameters(model)
            for name in totals:
                totals[name] += float(getattr(terms, name).detach())
        if config.optimization.schedule == "cosine":
            schedule.step()
        print(f"Epoch {epoch}: " + " ".join(f"{name}={value / len(loader):.6f}" for name, value in totals.items()))
    if averaged is not None:
        model.load_state_dict(averaged.module.state_dict())
    model.eval()

    temporary_output.mkdir(parents=True)
    torch.save(model.state_dict(), temporary_output / "weights.pt")
    raw = yaml.safe_load(config_path.read_text())
    asset_root = ASSETS_DIR / REVISION
    asset_paths = (
        *config.data.unsupervised_paths,
        *(source.path for source in config.data.pair_sources),
        *(source.path for source in config.data.contact_sources),
    )
    raw["provenance"] = {
        "assets_revision": REVISION,
        "asset_sha256": {
            str(path.relative_to(asset_root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in asset_paths
        },
    }
    (temporary_output / "train.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    os.replace(temporary_output, config.output)
    print(f"Saved checkpoint: {config.output}")
    return config.output


__all__ = [
    "OptimizationConfig",
    "TrainConfig",
    "load_training_data",
    "train_from_config",
]
