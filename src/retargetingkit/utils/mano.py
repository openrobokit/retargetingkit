"""MANO hand model and inverse keypoint fitter, backed by robokit."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from robokit.assets.body_models import get_body_models_dir
from robokit.smplx import BodyModelSpec, Side, load_mano, mano_forward_pca_mm


def default_mano_root() -> Path:
    """Return the user-provided MANO directory, honoring ROBOKIT_BODY_MODELS.

    >>> default_mano_root().name
    'mano'
    """
    return get_body_models_dir() / "mano"


@dataclass(frozen=True)
class ManoHandConfig:
    """Configuration for the MANO hand target.

    >>> ManoHandConfig().side
    'right'
    """

    side: Side = "right"
    device: str = "auto"


@dataclass(frozen=True)
class ManoHandState:
    """Batched MANO mesh and OpenPose-order keypoints in meters.

    >>> ManoHandState(np.zeros((1, 778, 3)), np.zeros((1, 21, 3)), np.zeros((1, 3), dtype=np.int32)).keypoints.shape
    (1, 21, 3)
    """

    vertices: np.ndarray
    keypoints: np.ndarray
    faces: np.ndarray


class ManoHand:
    """Forward manopth-style MANO pose to mesh vertices and 21 hand keypoints."""

    def __init__(self, config: ManoHandConfig = ManoHandConfig()):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device
        self.spec = load_mano(default_mano_root() / f"MANO_{config.side.upper()}.npz", side=config.side)
        self.faces = self.spec.faces.astype(np.int32)

    def forward_pca(self, pose_48: np.ndarray) -> ManoHandState:
        """Run MANO from `[global_orient(3), finger_pca(45)]` pose(s) with the flat hand mean."""
        pose = torch.as_tensor(pose_48, dtype=torch.float32, device=self.device).reshape(-1, 48)
        betas = torch.zeros((pose.shape[0], self.spec.num_betas), dtype=torch.float32, device=self.device)
        transl = torch.zeros((pose.shape[0], 3), dtype=torch.float32, device=self.device)
        vertices_mm, keypoints_mm = mano_forward_pca_mm(self.spec, pose, betas, transl, flat_hand_mean=True)
        return ManoHandState((vertices_mm * 0.001).cpu().numpy(), (keypoints_mm * 0.001).cpu().numpy(), self.faces)


@dataclass(frozen=True)
class ManoFitConfig:
    """Configuration for gradient-based MANO keypoint fitting.

    >>> ManoFitConfig(iterations=1).iterations
    1
    """

    iterations: int = 220
    lr: float = 0.02
    pose_l2_weight: float = 1.0e-4
    temporal_weight: float = 1.0e-4
    flat_hand_mean: bool = False
    use_pca: bool = True
    active_pca_components: int = 15
    device: str = "auto"


@dataclass(frozen=True)
class ManoFitResult:
    """Fitted MANO sequence.

    >>> ManoFitResult(
    ...     pose_48=np.zeros((1, 48), dtype=np.float32),
    ...     transl=np.zeros((1, 3), dtype=np.float32),
    ...     scale=np.ones((1,), dtype=np.float32),
    ...     vertices=np.zeros((1, 778, 3), dtype=np.float32),
    ...     keypoints=np.zeros((1, 21, 3), dtype=np.float32),
    ...     losses=np.zeros((1,), dtype=np.float32),
    ... ).keypoints.shape
    (1, 21, 3)
    """

    pose_48: np.ndarray
    transl: np.ndarray
    scale: np.ndarray
    vertices: np.ndarray
    keypoints: np.ndarray
    losses: np.ndarray


class ManoKeypointFitter:
    """Fit MANO pose, hand scale, and wrist translation to OpenPose-order 21 hand keypoints."""

    def __init__(self, spec: BodyModelSpec, config: ManoFitConfig = ManoFitConfig()):
        self.spec = spec
        self.config = config
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device

    def fit(self, target_keypoints: np.ndarray) -> ManoFitResult:
        """Fit a batch of MANO poses to `[B, 21, 3]` target keypoints in meters."""
        cfg = self.config
        target = torch.as_tensor(target_keypoints, dtype=torch.float32, device=self.device).reshape(-1, 21, 3)
        B = target.shape[0]
        betas = torch.zeros((B, self.spec.num_betas), dtype=torch.float32, device=self.device)
        zero_transl = torch.zeros((B, 3), dtype=torch.float32, device=self.device)
        pose_48 = torch.nn.Parameter(torch.zeros((B, 48), dtype=torch.float32, device=self.device))
        pose_mask = torch.ones(48, dtype=torch.float32, device=self.device)
        if cfg.use_pca:
            pose_mask[3 + cfg.active_pca_components :] = 0.0

        with torch.no_grad():
            _, rest_mm = mano_forward_pca_mm(self.spec, pose_48, betas, zero_transl, cfg.flat_hand_mean, cfg.use_pca)
            rest = rest_mm * 0.001
            # the wrist is MANO's root joint, so its position does not depend on pose
            wrist = rest[:, :1]
            tips = [4, 8, 12, 16, 20]
            target_span = torch.linalg.norm(target[:, tips] - target[:, :1], dim=-1).mean(dim=-1)
            rest_span = torch.linalg.norm(rest[:, tips] - wrist, dim=-1).mean(dim=-1)
        log_scale = torch.nn.Parameter(torch.log(torch.clamp(target_span / rest_span, min=0.75, max=1.35)))
        optimizer = torch.optim.Adam([pose_48, log_scale], lr=cfg.lr)
        losses: list[float] = []
        for _ in range(cfg.iterations):
            optimizer.zero_grad()
            pose = pose_48 * pose_mask
            keypoints_mm = mano_forward_pca_mm(self.spec, pose, betas, zero_transl, cfg.flat_hand_mean, cfg.use_pca)[1]
            keypoints = target[:, :1] + (keypoints_mm * 0.001 - wrist) * torch.exp(log_scale)[:, None, None]
            loss = ((keypoints - target) ** 2).mean() + (pose[:, 3:] ** 2).mean() * cfg.pose_l2_weight
            if B > 1:
                loss = loss + ((pose[1:] - pose[:-1]) ** 2).mean() * cfg.temporal_weight
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        with torch.no_grad():
            pose = pose_48 * pose_mask
            scale = torch.exp(log_scale)
            vertices_mm, keypoints_mm = mano_forward_pca_mm(
                self.spec, pose, betas, zero_transl, cfg.flat_hand_mean, cfg.use_pca
            )
            vertices = target[:, :1] + (vertices_mm * 0.001 - wrist) * scale[:, None, None]
            keypoints = target[:, :1] + (keypoints_mm * 0.001 - wrist) * scale[:, None, None]
        return ManoFitResult(
            pose_48=pose.cpu().numpy(),
            transl=(target[:, 0] - wrist[:, 0]).cpu().numpy(),
            scale=scale.cpu().numpy(),
            vertices=vertices.cpu().numpy(),
            keypoints=keypoints.cpu().numpy(),
            losses=np.asarray(losses, dtype=np.float32),
        )


__all__ = [
    "ManoFitConfig",
    "ManoFitResult",
    "ManoHand",
    "ManoHandConfig",
    "ManoHandState",
    "ManoKeypointFitter",
    "default_mano_root",
]
