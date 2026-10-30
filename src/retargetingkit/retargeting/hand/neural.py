"""Per-finger neural model and checkpoint inference."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from retargetingkit.assets import neural_hand
from retargetingkit.retargeting.hand.presets import HandPreset
from retargetingkit.retargeting.retargeter import Retargeter, RetargeterConfig
from retargetingkit.sources.frames import HandFrame, Side
from retargetingkit.targets.commands import QPosCommand


def select_mano_keypoints(keypoints: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    """Select ordered points from MANO-21 frames.

    >>> from retargetingkit.sources.frames import MANO21
    >>> select_mano_keypoints(np.zeros((21, 3)), (MANO21.THUMB_TIP, MANO21.INDEX_TIP)).shape
    (2, 3)
    """
    return keypoints[..., list(indices), :].astype(np.float32)


class IKModel(nn.Module):
    """Independent per-finger MLPs producing canonical robot joint order."""

    def __init__(self, finger_joint_sizes: tuple[int, ...]) -> None:
        super().__init__()
        hidden = 128
        self.finger_joint_sizes = finger_joint_sizes
        self.networks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(3, hidden),
                nn.LeakyReLU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.LeakyReLU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, size),
                nn.Tanh(),
            )
            for size in finger_joint_sizes
        )

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [network(keypoints[:, finger]) for finger, network in enumerate(self.networks)],
            dim=-1,
        )


def normalize_joints(joints: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Map physical joint angles to `[-1, 1]`.

    >>> normalize_joints(torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([2.0])).tolist()
    [0.0]
    """
    return ((joints - lower) / (upper - lower) - 0.5) * 2.0


def denormalize_joints(joints: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Map normalized joints back to physical joint angles.

    >>> denormalize_joints(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([2.0])).tolist()
    [1.0]
    """
    return (joints + 1.0) * 0.5 * (upper - lower) + lower


@dataclass
class NeuralHandRetargeterConfig(RetargeterConfig[HandFrame, QPosCommand]):
    """One human hand mapped onto one robot hand by a trained checkpoint.

    >>> from pathlib import Path
    >>> from retargetingkit.retargeting.hand.presets import HANDS
    >>> NeuralHandRetargeterConfig(Path("run"), HANDS["xhand"]).side
    'right'
    """

    checkpoint_dir: Path
    hand: HandPreset
    side: Side = "right"

    def build(self, device: str) -> "NeuralHandRetargeter":
        """Load the checkpoint onto `device`.

        >>> NeuralHandRetargeterConfig(Path("run"), HANDS["xhand"]).build("cpu")  # doctest: +SKIP
        """
        return NeuralHandRetargeter(self, device)


@dataclass
class NeuralHandAssetRetargeterConfig(RetargeterConfig[HandFrame, QPosCommand]):
    """A named Hugging Face checkpoint resolved when the retargeter is built."""

    checkpoint: str
    hand: HandPreset
    side: Side = "right"

    def build(self, device: str) -> "NeuralHandRetargeter":
        """Download the checkpoint if needed, then load it on `device`.

        >>> config.build("cpu")  # doctest: +SKIP
        """
        return NeuralHandRetargeterConfig(neural_hand.get(self.checkpoint), self.hand, self.side).build(device)


class NeuralHandRetargeter(Retargeter[HandFrame, QPosCommand]):
    """Load one checkpoint and map MANO-21 frames to physical robot joints.

    >>> config.build("cpu")  # doctest: +SKIP
    """

    def __init__(self, config: NeuralHandRetargeterConfig, device: str) -> None:
        hand = config.hand
        self.side: Side = config.side
        self.device = torch.device(device)
        self._human_keypoint_indices = hand.human_keypoint_indices
        self._lower = torch.tensor(hand.joint_lower, dtype=torch.float32, device=self.device)
        self._upper = torch.tensor(hand.joint_upper, dtype=torch.float32, device=self.device)
        self._model = IKModel(hand.finger_joint_sizes).to(self.device)
        self._model.load_state_dict(
            torch.load(Path(config.checkpoint_dir) / "weights.pt", map_location=self.device, weights_only=True)
        )
        self._model.eval()

    @torch.no_grad()
    def predict_normalized(self, selected_keypoints: torch.Tensor) -> torch.Tensor:
        """Predict normalized joints from already-selected keypoints on `device`."""
        return self._model(selected_keypoints)

    @torch.no_grad()
    def retarget_batch(self, mano_keypoints: np.ndarray) -> np.ndarray:
        """Retarget a batch of MANO-21 frames to `[N, Q]` physical joint angles.

        >>> retargeter.retarget_batch(np.zeros((2, 21, 3), np.float32)).shape  # doctest: +SKIP
        (2, 16)
        """
        selected = select_mano_keypoints(mano_keypoints, self._human_keypoint_indices)
        inputs = torch.from_numpy(selected).to(self.device)
        joints = denormalize_joints(self.predict_normalized(inputs), self._lower, self._upper)
        return joints.cpu().numpy()

    def retarget(self, mano_keypoints: np.ndarray) -> np.ndarray:
        """Retarget one MANO-21 frame to `[Q]` physical joint angles.

        >>> retargeter.retarget(np.zeros((21, 3), np.float32)).shape  # doctest: +SKIP
        (16,)
        """
        return self.retarget_batch(mano_keypoints[None])[0]

    def solve(self, frame: HandFrame) -> QPosCommand:
        """Retarget one canonical hand frame."""
        return QPosCommand(self.retarget(frame.hands[self.side].positions).astype(np.float32))


__all__ = [
    "IKModel",
    "NeuralHandAssetRetargeterConfig",
    "NeuralHandRetargeterConfig",
    "NeuralHandRetargeter",
    "denormalize_joints",
    "normalize_joints",
    "select_mano_keypoints",
]
