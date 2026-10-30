from dataclasses import dataclass

import numpy as np
import pytest

from retargetingkit.configs.teleop.config import HandTeleopConfig
from retargetingkit.configs.teleop.hand import _HAND_TELEOP_CONFIGS
from retargetingkit.retargeting.hand.neural import NeuralHandAssetRetargeterConfig, NeuralHandRetargeterConfig
from retargetingkit.retargeting.hand.optimization import OptimizationHandRetargeterConfig
from retargetingkit.sources.frames import HandFrame, HandPose
from retargetingkit.sources.glove.manus_glove import ManusSourceConfig
from retargetingkit.sources.source import Source, SourceConfig
from retargetingkit.sources.vr.vision_pro import VisionProHandConfig
from retargetingkit.targets.hands.ability_hand import AbilityHandConfig
from retargetingkit.targets.hands.sharpa_hand import SharpaHandConfig
from retargetingkit.targets.hands.xhand import XHandConfig
from retargetingkit.targets.target import TargetConfig
from robokit.assets.robots.end_effectors import ability_hand, sharpa_hand, xhand
from robokit.helpers.hand_retargeting.presets import ability, sharpa
from robokit.helpers.hand_retargeting.presets import xhand as xhand_preset
from scripts.teleop import hand as hand_teleop


RIGS = (
    (
        XHandConfig("/dev/null"),
        OptimizationHandRetargeterConfig(
            xhand.URDF_PATH_RIGHT,
            xhand_preset.spec,
            xhand_preset.online,
            "right",
            collision_spheres=xhand.COLLISION_SPHERE_PATH_RIGHT,
            collision_ignore=xhand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT,
        ),
    ),
    (
        AbilityHandConfig("/dev/null"),
        OptimizationHandRetargeterConfig(
            ability_hand.URDF_PATH_RIGHT,
            ability.spec,
            ability.online,
            "right",
            collision_spheres=ability_hand.COLLISION_SPHERE_PATH_RIGHT,
        ),
    ),
    (
        SharpaHandConfig(),
        OptimizationHandRetargeterConfig(
            sharpa_hand.URDF_PATH_RIGHT,
            sharpa.spec,
            sharpa.online,
            "right",
            collision_spheres=sharpa_hand.COLLISION_SPHERE_PATH_RIGHT,
            collision_ignore=sharpa_hand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT,
        ),
    ),
)


def _open_hand_frame() -> HandFrame:
    points = np.zeros((21, 3), dtype=np.float32)
    for finger, x in enumerate((-0.04, -0.02, 0.0, 0.02, 0.04)):
        for joint in range(4):
            points[1 + finger * 4 + joint] = (x, 0.02 * (joint + 1), 0.0)
    return HandFrame({"right": HandPose(points)})


class _FakeSource(Source[HandFrame]):
    closed = False
    frames_served = 0

    def get_frame(self) -> HandFrame | None:
        if _FakeSource.frames_served == 1:
            raise KeyboardInterrupt
        _FakeSource.frames_served += 1
        return _open_hand_frame()

    def close(self) -> None:
        _FakeSource.closed = True


@dataclass
class _FakeSourceConfig(SourceConfig[HandFrame]):
    def build(self) -> Source[HandFrame]:
        return _FakeSource()


class TestHandTeleop:
    @pytest.mark.parametrize(("target", "retargeter"), RIGS)
    def test_one_viser_frame(self, target: TargetConfig, retargeter: OptimizationHandRetargeterConfig) -> None:
        _FakeSource.closed = False
        _FakeSource.frames_served = 0

        with pytest.raises(KeyboardInterrupt):
            hand_teleop.run(
                HandTeleopConfig(
                    source=_FakeSourceConfig(),
                    target=target,
                    retargeter=retargeter,
                    mode="viser",
                    device="cpu",
                )
            )

        assert _FakeSource.frames_served == 1
        assert _FakeSource.closed

    def test_presets_agree_on_side(self) -> None:
        for name, config in _HAND_TELEOP_CONFIGS.items():
            assert isinstance(
                config.retargeter,
                (OptimizationHandRetargeterConfig, NeuralHandRetargeterConfig, NeuralHandAssetRetargeterConfig),
            )
            assert config.retargeter.side == config.side, name

    def test_visionpro_sharpa_preset(self) -> None:
        config = _HAND_TELEOP_CONFIGS["visionpro_sharpa"]
        assert isinstance(config.source, VisionProHandConfig)
        assert isinstance(config.target, SharpaHandConfig)

    def test_manus_vector_presets_restore_benchmark_preprocessing(self) -> None:
        optimization = _HAND_TELEOP_CONFIGS["manus_sharpa"]
        xhand = _HAND_TELEOP_CONFIGS["manus_xhand"]
        neural = _HAND_TELEOP_CONFIGS["manus_sharpa_neural"]

        for config in (optimization, xhand):
            assert isinstance(config.source, ManusSourceConfig)
            assert config.source.use_calibration
            assert config.source.calibration_path is None
        assert isinstance(optimization.target, SharpaHandConfig)
        assert optimization.target.speed_coeff == 1.0
        assert optimization.target.current_coeff == 0.5
        assert optimization.rate == 60.0
        assert isinstance(neural.source, ManusSourceConfig)
        assert not neural.source.use_calibration
        assert isinstance(neural.retargeter, NeuralHandAssetRetargeterConfig)
        assert neural.retargeter.checkpoint == "sharpa_motion_alignment_contact_inhand"
        assert isinstance(neural.target, SharpaHandConfig)
        assert neural.target.speed_coeff == 1.0
        assert neural.target.current_coeff == 0.5
        assert neural.rate == 60.0
