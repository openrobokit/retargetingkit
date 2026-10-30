"""Teleop rig schemas, one dataclass per `scripts/teleop` runtime; named rig presets live in sibling packages."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from retargetingkit.retargeting.arm.online import ArmRetargeterConfig
from retargetingkit.retargeting.humanoid.online import HumanoidRetargeterConfig
from retargetingkit.retargeting.retargeter import RetargeterConfig
from retargetingkit.sources.frames import ArmFrame, ArmHandFrame, BodyFrame, HandFrame, Side
from retargetingkit.sources.source import SourceConfig
from retargetingkit.sources.vr.pico import PicoSourceConfig
from retargetingkit.targets.commands import QPosCommand
from retargetingkit.targets.target import TargetConfig


@dataclass
class HandTeleopConfig:
    """One hand source retargeted to one robot hand."""

    source: SourceConfig[HandFrame]
    """Hand motion source blueprint."""
    target: TargetConfig
    """Robot hand blueprint."""
    retargeter: RetargeterConfig[HandFrame, QPosCommand]
    """Optimization or neural hand retargeting blueprint."""
    side: Side = "right"
    """Hand side shared by the source and retargeter."""
    rate: float = 30.0
    """Control-loop frequency in Hz."""
    mode: Literal["viser", "real", "both"] = "viser"
    """Viser only, hardware only, or both."""
    device: str = "cuda:0"
    """Warp device used by RoboKit."""


@dataclass
class ArmTeleopConfig:
    """One arm source controlling one robot arm per side."""

    source: SourceConfig[ArmFrame]
    """Wrist motion source blueprint."""
    arms: dict[Side, TargetConfig]
    """Robot arm blueprint per side."""
    retargeters: dict[Side, ArmRetargeterConfig]
    """Wrist-to-arm retargeting blueprint per side."""
    mode: Literal["viser", "real", "both"] = "viser"
    """Viser only, hardware only, or both."""
    device: str = "cuda:0"
    """Warp device used by RoboKit."""


@dataclass
class ArmHandTeleopConfig:
    """One arm-and-hand source controlling one robot arm and one robot hand per side."""

    source: SourceConfig[ArmHandFrame]
    """Wrist and hand motion source blueprint."""
    arms: dict[Side, TargetConfig]
    """Robot arm blueprint per side."""
    hands: dict[Side, TargetConfig]
    """Robot hand blueprint per side."""
    arm_retargeters: dict[Side, ArmRetargeterConfig]
    """Wrist-to-arm retargeting blueprint per side."""
    hand_retargeters: dict[Side, RetargeterConfig[HandFrame, QPosCommand]]
    """Hand retargeting blueprint per side."""
    assembly_urdf: Path
    """Combined arm-and-hand URDF used only for visualization."""
    mode: Literal["viser", "real", "both"] = "viser"
    """Viser only, hardware only, or both."""
    device: str = "cuda:0"
    """Warp device used by RoboKit."""


@dataclass
class HumanoidTeleopConfig:
    """One body source retargeted to one or more humanoid targets."""

    source: SourceConfig[BodyFrame]
    """Body motion source blueprint."""
    retargeter: HumanoidRetargeterConfig
    """Body-to-humanoid retargeting blueprint."""
    targets: tuple[TargetConfig, ...]
    """Humanoid target blueprints fed the same command."""
    device: str = "cuda:0"
    """Warp device used by RoboKit."""


@dataclass
class HumanoidHandTeleopConfig:
    """One PICO connection controlling a humanoid and two robot hands."""

    source: PicoSourceConfig
    """PICO body and controller source blueprint."""
    retargeter: HumanoidRetargeterConfig
    """Body-to-humanoid retargeting blueprint."""
    targets: tuple[TargetConfig, ...]
    """Humanoid target blueprints fed the same command."""
    left_hand: TargetConfig
    """Left robot hand blueprint."""
    right_hand: TargetConfig
    """Right robot hand blueprint."""
    open_qpos: tuple[float, ...]
    """Hand joint positions commanded when the trigger is released."""
    closed_qpos: tuple[float, ...]
    """Hand joint positions commanded when the trigger is pressed."""
    device: str = "cuda:0"
    """Warp device used by RoboKit."""


__all__ = [
    "ArmHandTeleopConfig",
    "ArmTeleopConfig",
    "HandTeleopConfig",
    "HumanoidHandTeleopConfig",
    "HumanoidTeleopConfig",
]
