"""Training objective configuration and assembly."""

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Literal

import torch

from retargetingkit.training.hand.data import JointSupervision
from retargetingkit.training.hand.metrics import (
    CURVATURE_STEP_M,
    INTERFINGER_MARGIN_SCALE,
    INTERFINGER_MIN_MARGIN_M,
    PINCH_THRESHOLD_M,
    MotionWorkspace,
    collision_loss,
    conformal_loss,
    cosine_motion_loss,
    curvature_loss,
    interfinger_loss,
    motion_alignment_loss,
    pinch_loss,
    sample_fixed_motion,
    sample_motion,
)


@dataclass(frozen=True)
class MotionConfig:
    """Local human/robot motion objective."""

    formula: Literal["motion_alignment", "cosine"]
    weight: float
    directions: int
    temperature: float
    perturb_min_m: float
    perturb_max_m: float


@dataclass(frozen=True)
class ObjectiveConfig:
    """Weights and motion parameters for the complete objective."""

    pair: float
    pair_position: float
    contact: float
    motion: MotionConfig
    conformal: float
    curvature: float
    pinch: float
    interfinger: float
    collision: float

    @classmethod
    def from_dict(cls, raw: dict) -> "ObjectiveConfig":
        """Parse the explicit objective section of a training config."""
        assert set(raw) == {field.name for field in fields(cls)}
        motion = MotionConfig(**raw["motion"])
        assert motion.formula in ("motion_alignment", "cosine")
        assert motion.directions % 2 == 0
        return cls(
            pair=float(raw["pair"]),
            pair_position=float(raw["pair_position"]),
            contact=float(raw["contact"]),
            motion=motion,
            conformal=float(raw["conformal"]),
            curvature=float(raw["curvature"]),
            pinch=float(raw["pinch"]),
            interfinger=float(raw["interfinger"]),
            collision=float(raw["collision"]),
        )


@dataclass(frozen=True)
class ObjectiveContext:
    """Fixed resources used by every training batch."""

    robot_keypoints: Callable[[torch.Tensor], torch.Tensor]
    motion_workspace: MotionWorkspace | None
    finger_joint_sizes: tuple[int, ...]
    finger_pairs: tuple[tuple[int, int], ...]
    collision_residual: Callable[[torch.Tensor], torch.Tensor] | None
    generator: torch.Generator


@dataclass(frozen=True)
class LossTerms:
    """Unweighted objective terms for one training batch."""

    pair: torch.Tensor
    pair_position: torch.Tensor
    contact: torch.Tensor
    motion: torch.Tensor
    conformal: torch.Tensor
    curvature: torch.Tensor
    pinch: torch.Tensor
    interfinger: torch.Tensor
    collision: torch.Tensor

    def weighted_sum(self, config: ObjectiveConfig) -> torch.Tensor:
        """Return the configured scalar training objective."""
        return (
            self.pair * config.pair
            + self.pair_position * config.pair_position
            + self.contact * config.contact
            + self.motion * config.motion.weight
            + self.conformal * config.conformal
            + self.curvature * config.curvature
            + self.pinch * config.pinch
            + self.interfinger * config.interfinger
            + self.collision * config.collision
        )


def masked_joint_loss(predicted: torch.Tensor, supervision: JointSupervision) -> torch.Tensor:
    """Return sample-weighted masked normalized-joint MSE.

    >>> masked_joint_loss(predicted, supervision)  # doctest: +SKIP
    """
    squared_error = (predicted - supervision.target_qpos).square() * supervision.joint_mask
    per_frame = squared_error.sum(dim=1) / supervision.joint_mask.sum(dim=1)
    return (per_frame * supervision.sample_weight).sum() / supervision.sample_weight.sum()


def masked_keypoint_loss(
    predicted: torch.Tensor,
    supervision: JointSupervision,
    robot_keypoints: Callable[[torch.Tensor], torch.Tensor],
    finger_joint_sizes: tuple[int, ...],
) -> torch.Tensor:
    """Return sample-weighted masked robot fingertip MSE.

    >>> masked_keypoint_loss(predicted, supervision, robot_keypoints, (4, 4, 4, 4))  # doctest: +SKIP
    """
    finger_mask = torch.stack(
        [part.amax(dim=1) for part in supervision.joint_mask.split(finger_joint_sizes, dim=1)],
        dim=1,
    )
    squared_error = (robot_keypoints(predicted) - robot_keypoints(supervision.target_qpos).detach()).square().sum(
        dim=2
    ) * finger_mask
    per_frame = squared_error.sum(dim=1) / finger_mask.sum(dim=1)
    return (per_frame * supervision.sample_weight).sum() / supervision.sample_weight.sum()


def compute_objective(
    model: Callable[[torch.Tensor], torch.Tensor],
    human_points: torch.Tensor,
    pair: JointSupervision,
    contact: JointSupervision,
    context: ObjectiveContext,
    config: ObjectiveConfig,
) -> LossTerms:
    """Compute the complete objective with explicit data scopes.

    >>> compute_objective(model, human, pair, contact, context, config)  # doctest: +SKIP
    """
    human_q = model(human_points)
    pair_q = model(pair.keypoints)
    contact_q = model(contact.keypoints)
    robot_points = context.robot_keypoints(human_q)
    zero = human_points.new_zeros(())

    pair_loss = masked_joint_loss(pair_q, pair) if config.pair else zero
    pair_position = (
        masked_keypoint_loss(pair_q, pair, context.robot_keypoints, context.finger_joint_sizes)
        if config.pair_position
        else zero
    )
    contact_loss = masked_joint_loss(contact_q, contact) if config.contact else zero

    curvature = zero
    if config.curvature:
        delta = sample_fixed_motion(human_points, CURVATURE_STEP_M, context.generator)
        curvature = curvature_loss(
            robot_points,
            context.robot_keypoints(model(human_points + delta)),
            context.robot_keypoints(model(human_points - delta)),
        )

    motion = zero
    conformal = zero
    if config.motion.weight or config.conformal:
        human_motion = sample_motion(
            human_points,
            config.motion.perturb_min_m,
            config.motion.perturb_max_m,
            context.generator,
        )
        robot_motion = context.robot_keypoints(model(human_points + human_motion)) - robot_points
        if config.motion.weight:
            if config.motion.formula == "motion_alignment":
                assert context.motion_workspace is not None
                motion = motion_alignment_loss(
                    human_points,
                    robot_points,
                    human_motion,
                    robot_motion,
                    context.motion_workspace,
                    config.motion.temperature,
                )
            else:
                motion = cosine_motion_loss(human_motion, robot_motion)
        if config.conformal:
            conformal = conformal_loss(human_motion, robot_motion)

    pinch = pinch_loss(human_points, robot_points, context.finger_pairs, PINCH_THRESHOLD_M) if config.pinch else zero
    interfinger = (
        interfinger_loss(
            human_points,
            robot_points,
            context.finger_pairs,
            INTERFINGER_MARGIN_SCALE,
            INTERFINGER_MIN_MARGIN_M,
        )
        if config.interfinger
        else zero
    )
    collision = collision_loss(context.collision_residual(human_q)) if context.collision_residual is not None else zero
    return LossTerms(
        pair_loss,
        pair_position,
        contact_loss,
        motion,
        conformal,
        curvature,
        pinch,
        interfinger,
        collision,
    )


__all__ = [
    "LossTerms",
    "MotionConfig",
    "ObjectiveConfig",
    "ObjectiveContext",
    "compute_objective",
    "masked_joint_loss",
    "masked_keypoint_loss",
]
