"""Shared mathematical metrics and sampling for training and evaluation."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


PROJECTION_BATCH_SIZE = 100_000
CURVATURE_STEP_M = 0.002
PINCH_THRESHOLD_M = 0.015
INTERFINGER_MARGIN_SCALE = 0.3
INTERFINGER_MIN_MARGIN_M = 0.001
COLLISION_MARGIN_M = 0.0


@dataclass(frozen=True)
class MotionWorkspace:
    """Per-keypoint human and robot support along fixed spatial directions."""

    directions: torch.Tensor
    robot_support: torch.Tensor
    robot_width: torch.Tensor
    human_support: torch.Tensor
    human_width: torch.Tensor


def build_motion_workspace(
    robot_cloud: torch.Tensor,
    human_frames: torch.Tensor,
    direction_count: int,
    generator: torch.Generator,
) -> MotionWorkspace:
    """Build workspace support from all human frames and sampled robot poses.

    >>> workspace = build_motion_workspace(robot_cloud, human_frames, 64, generator)  # doctest: +SKIP
    """
    assert direction_count % 2 == 0
    directions = F.normalize(
        torch.randn((direction_count // 2, 3), generator=generator, device=robot_cloud.device),
        dim=1,
    )
    directions = torch.cat((directions, -directions))
    robot_max: list[torch.Tensor] = []
    robot_min: list[torch.Tensor] = []
    for cloud in robot_cloud.split(PROJECTION_BATCH_SIZE, dim=1):
        projection = torch.einsum("knc,dc->knd", cloud, directions)
        robot_max.append(projection.amax(dim=1))
        robot_min.append(projection.amin(dim=1))
    human_projection = torch.einsum("nkc,dc->nkd", human_frames, directions)
    robot_support = torch.stack(robot_max).amax(dim=0)
    human_support = human_projection.amax(dim=0)
    return MotionWorkspace(
        directions=directions,
        robot_support=robot_support,
        robot_width=robot_support - torch.stack(robot_min).amin(dim=0),
        human_support=human_support,
        human_width=human_support - human_projection.amin(dim=0),
    )


def motion_alignment_loss(
    human_points: torch.Tensor,
    robot_points: torch.Tensor,
    human_motion: torch.Tensor,
    robot_motion: torch.Tensor,
    workspace: MotionWorkspace,
    temperature: float,
) -> torch.Tensor:
    """Return workspace-aware local motion alignment loss.

    >>> motion_alignment_loss(human, robot, human_delta, robot_delta, workspace, 0.05)  # doctest: +SKIP
    """
    human_direction = F.normalize(human_motion, dim=-1, eps=1.0e-5).unsqueeze(2)
    robot_direction = F.normalize(robot_motion, dim=-1, eps=1.0e-5).unsqueeze(2)
    human_projection = torch.einsum("nkc,dc->nkd", human_points, workspace.directions)
    robot_projection = torch.einsum("nkc,dc->nkd", robot_points, workspace.directions)
    human_remaining = (workspace.human_support - human_projection) / workspace.human_width
    robot_remaining = (workspace.robot_support - robot_projection) / workspace.robot_width
    error = (robot_remaining.unsqueeze(-1) * robot_direction - human_remaining.unsqueeze(-1) * human_direction).norm(
        dim=-1
    )
    return temperature * (torch.logsumexp(error / temperature, dim=2) - math.log(error.shape[2])).mean()


def cosine_motion_loss(human_motion: torch.Tensor, robot_motion: torch.Tensor) -> torch.Tensor:
    """Return negative cosine alignment between human and robot motion.

    >>> float(cosine_motion_loss(torch.tensor([[[1.0, 0, 0]]]), torch.tensor([[[1.0, 0, 0]]])))
    -1.0
    """
    return (
        -(F.normalize(human_motion, dim=-1, eps=1.0e-5) * F.normalize(robot_motion, dim=-1, eps=1.0e-5))
        .sum(dim=-1)
        .mean()
    )


def curvature_loss(center: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """Return local finite-difference curvature.

    >>> float(curvature_loss(torch.ones(1), torch.ones(1), torch.ones(1)))
    0.0
    """
    return (positive + negative - 2 * center).square().mean()


def conformal_loss(human_motion: torch.Tensor, robot_motion: torch.Tensor) -> torch.Tensor:
    """Return the global best-scale local conformal residual.

    >>> float(conformal_loss(torch.ones(1, 1, 3), torch.ones(1, 1, 3)))
    0.0
    """
    flat_human = human_motion.reshape(-1, 3)
    flat_robot = robot_motion.reshape(-1, 3)
    denominator = flat_human.square().sum().clamp_min(1.0e-9)
    scale = (flat_robot * flat_human).sum() / denominator
    return (flat_robot - scale * flat_human).square().sum() / denominator


def pinch_loss(
    human_points: torch.Tensor,
    robot_points: torch.Tensor,
    finger_pairs: tuple[tuple[int, int], ...],
    threshold_m: float,
) -> torch.Tensor:
    """Penalize robot gaps for human fingertip pairs within `threshold_m`.

    >>> pinch_loss(human, robot, ((0, 1),), 0.015)  # doctest: +SKIP
    """
    terms: list[torch.Tensor] = []
    for left, right in finger_pairs:
        human_gap = (human_points[:, left] - human_points[:, right]).norm(dim=-1)
        mask = (human_gap < threshold_m).float()
        robot_distance = (robot_points[:, left] - robot_points[:, right]).square().sum(dim=-1)
        terms.append((mask * robot_distance).sum() / (mask.sum() + 1.0e-7))
    return torch.stack(terms).sum()


def interfinger_loss(
    human_points: torch.Tensor,
    robot_points: torch.Tensor,
    finger_pairs: tuple[tuple[int, int], ...],
    margin_scale: float,
    min_margin_m: float,
) -> torch.Tensor:
    """Preserve the ordinal inter-finger distances across human samples.

    >>> interfinger_loss(human, robot, ((0, 1),), 0.3, 0.001)  # doctest: +SKIP
    """
    upper = torch.triu(
        torch.ones((len(human_points), len(human_points)), dtype=torch.bool, device=human_points.device),
        diagonal=1,
    )
    terms: list[torch.Tensor] = []
    for left, right in finger_pairs:
        human_gap = (human_points[:, left] - human_points[:, right]).norm(dim=-1)
        robot_gap = (robot_points[:, left] - robot_points[:, right]).norm(dim=-1)
        human_difference = human_gap[:, None] - human_gap[None]
        robot_difference = robot_gap[:, None] - robot_gap[None]
        margin = (margin_scale * human_difference.abs()).clamp_min(min_margin_m)
        terms.append(F.relu(margin - torch.sign(human_difference) * robot_difference)[upper].mean())
    return torch.stack(terms).sum()


def collision_loss(residual: torch.Tensor) -> torch.Tensor:
    """Aggregate per-pair collision residuals over a batch.

    >>> float(collision_loss(torch.tensor([[1.0, 2.0], [3.0, 4.0]])))
    5.0
    """
    return residual.sum(dim=-1).mean()


def sample_motion(
    points: torch.Tensor,
    minimum_m: float,
    maximum_m: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample isotropic per-keypoint motion with explicit metric bounds.

    >>> sample_motion(torch.zeros(2, 5, 3), 0.001, 0.011, torch.Generator()).shape
    torch.Size([2, 5, 3])
    """
    direction = F.normalize(
        torch.randn(points.shape, generator=generator, device=points.device, dtype=points.dtype),
        dim=-1,
    )
    magnitude = minimum_m + torch.rand(
        (len(points), 1, 1),
        generator=generator,
        device=points.device,
        dtype=points.dtype,
    ) * (maximum_m - minimum_m)
    return direction * magnitude


def sample_fixed_motion(points: torch.Tensor, step_m: float, generator: torch.Generator) -> torch.Tensor:
    """Sample isotropic per-keypoint motion with fixed metric magnitude.

    >>> sample_fixed_motion(torch.zeros(2, 5, 3), 0.002, torch.Generator()).shape
    torch.Size([2, 5, 3])
    """
    return (
        F.normalize(
            torch.randn(points.shape, generator=generator, device=points.device, dtype=points.dtype),
            dim=-1,
        )
        * step_m
    )


__all__ = [
    "COLLISION_MARGIN_M",
    "CURVATURE_STEP_M",
    "INTERFINGER_MARGIN_SCALE",
    "INTERFINGER_MIN_MARGIN_M",
    "MotionWorkspace",
    "PINCH_THRESHOLD_M",
    "build_motion_workspace",
    "collision_loss",
    "conformal_loss",
    "cosine_motion_loss",
    "curvature_loss",
    "interfinger_loss",
    "motion_alignment_loss",
    "pinch_loss",
    "sample_fixed_motion",
    "sample_motion",
]
