from dataclasses import dataclass

import numpy as np
import pytest

from retargetingkit.configs.teleop.arm import _ARM_TELEOP_CONFIGS
from retargetingkit.configs.teleop.config import ArmTeleopConfig
from retargetingkit.retargeting.arm.online import ArmRetargeterConfig
from retargetingkit.sources.frames import ArmFrame
from retargetingkit.sources.source import Source, SourceConfig
from retargetingkit.targets.arms.xarm7 import XARM7_EE_LINK, XARM7_HOME_QPOS, XArm7Config
from robokit.assets.robots.arms import xarm7
from scripts.teleop import arm as arm_teleop


class _FakeSource(Source[ArmFrame]):
    closed = False
    frames_served = 0

    def get_frame(self) -> ArmFrame | None:
        if _FakeSource.frames_served == 1:
            raise KeyboardInterrupt
        _FakeSource.frames_served += 1
        return ArmFrame({"right": np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)})

    def close(self) -> None:
        _FakeSource.closed = True


@dataclass
class _FakeSourceConfig(SourceConfig[ArmFrame]):
    def build(self) -> Source[ArmFrame]:
        return _FakeSource()


class TestArmTeleop:
    def test_one_viser_frame(self) -> None:
        _FakeSource.closed = False
        _FakeSource.frames_served = 0
        config = ArmTeleopConfig(
            source=_FakeSourceConfig(),
            arms={"right": XArm7Config("192.168.1.242")},
            retargeters={
                "right": ArmRetargeterConfig(xarm7.URDF_GLB_PATH, XARM7_EE_LINK, XARM7_HOME_QPOS, side="right"),
            },
            mode="viser",
            device="cpu",
        )

        with pytest.raises(KeyboardInterrupt):
            arm_teleop.run(config)

        assert _FakeSource.frames_served == 1
        assert _FakeSource.closed

    def test_presets_have_matching_sides(self) -> None:
        for config in _ARM_TELEOP_CONFIGS.values():
            assert config.arms.keys() == config.retargeters.keys()
            assert all(side == retargeter.side for side, retargeter in config.retargeters.items())

    def test_dual_arm_viser_order_is_left_then_right(self) -> None:
        config = _ARM_TELEOP_CONFIGS["visionpro_dual_xarm7"]
        assert tuple(config.arms) == tuple(config.retargeters) == ("left", "right")
