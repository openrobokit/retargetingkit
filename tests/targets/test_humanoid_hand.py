import signal

import numpy as np
import pytest
import torch

from retargetingkit.configs.teleop.config import HumanoidHandTeleopConfig
from retargetingkit.configs.teleop.humanoid_hand import _HUMANOID_HAND_TELEOP_CONFIGS
from retargetingkit.sources.frames import BodyFrame
from retargetingkit.targets.commands import BasePoseQPosCommand
from retargetingkit.targets.target import Target
from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnlineConfig
from scripts.teleop import humanoid_hand


class _FakeSource:
    def __init__(self) -> None:
        self.closed = False
        self.frames_served = 0

    def get_frame(self) -> BodyFrame:
        if self.frames_served == 1:
            signal.raise_signal(signal.SIGINT)
        self.frames_served += 1
        transforms = np.zeros((1, 7), dtype=np.float32)
        transforms[:, 3] = 1.0
        return BodyFrame(("pelvis",), transforms, 1.75)

    def get_buttons(self) -> dict[str, bool]:
        return {"B": True, "Y": False}

    def close(self) -> None:
        self.closed = True


class _FakeSourceConfig:
    def __init__(self, source: _FakeSource) -> None:
        self.source = source

    def build(self) -> _FakeSource:
        return self.source


class _FakeRetargeter:
    def solve(self, _frame: BodyFrame) -> BasePoseQPosCommand:
        return BasePoseQPosCommand(np.zeros(29), np.zeros(7))


class _FakeRetargeterConfig:
    preset = HumanoidRetargeterOnlineConfig(urdf_path="robot.urdf")

    def build(self, _device: str) -> _FakeRetargeter:
        return _FakeRetargeter()


class _FakeHumanoidTarget(Target):
    def __init__(self) -> None:
        self.states: list[BasePoseQPosCommand] = []
        self.closed = False

    def set_command(self, command: BasePoseQPosCommand) -> None:
        self.states.append(command)

    def close(self) -> None:
        self.closed = True


class _FakeHumanoidTargetConfig:
    def __init__(self, target: _FakeHumanoidTarget) -> None:
        self.target = target

    def build(self) -> _FakeHumanoidTarget:
        return self.target


class _FakeHand(Target):
    def __init__(self) -> None:
        self.commands: list[np.ndarray] = []
        self.stopped = False

    def set_qpos(self, qpos: np.ndarray) -> None:
        self.commands.append(qpos.copy())

    def close(self) -> None:
        self.stopped = True


class _FakeHandConfig:
    def __init__(self, target: _FakeHand) -> None:
        self.target = target

    def build(self) -> _FakeHand:
        return self.target


class TestHumanoidHandTeleop:
    def test_one_pico_connection_drives_body_and_right_hand(self, monkeypatch: pytest.MonkeyPatch) -> None:
        source = _FakeSource()
        humanoid_target = _FakeHumanoidTarget()
        left_hand = _FakeHand()
        right_hand = _FakeHand()
        config = HumanoidHandTeleopConfig(
            source=_FakeSourceConfig(source),  # type: ignore[arg-type]
            retargeter=_FakeRetargeterConfig(),  # type: ignore[arg-type]
            targets=(_FakeHumanoidTargetConfig(humanoid_target),),  # type: ignore[arg-type]
            left_hand=_FakeHandConfig(left_hand),  # type: ignore[arg-type]
            right_hand=_FakeHandConfig(right_hand),  # type: ignore[arg-type]
            open_qpos=(0.0,) * 6,
            closed_qpos=(1.3, 0.27, 0.85, 0.85, 0.85, 0.85),
            device="cpu",
        )
        monkeypatch.setattr(humanoid_hand.time, "sleep", lambda _seconds: None)

        humanoid_hand.run(config)

        assert source.frames_served == 2
        assert len(humanoid_target.states) == 2
        assert not left_hand.commands
        assert len(right_hand.commands) == 1
        torch.testing.assert_close(
            torch.from_numpy(right_hand.commands[0]),
            torch.tensor(config.closed_qpos, dtype=torch.float64),
        )
        assert source.closed
        assert humanoid_target.closed
        assert left_hand.stopped
        assert right_hand.stopped

    def test_named_preset_owns_pico_and_both_inspire_configs(self) -> None:
        config = _HUMANOID_HAND_TELEOP_CONFIGS["pico_g1_inspire"]
        assert type(config.source).__name__ == "PicoSourceConfig"
        assert type(config.left_hand).__name__ == "InspireHandConfig"
        assert type(config.right_hand).__name__ == "InspireHandConfig"
