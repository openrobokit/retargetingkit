import numpy as np

from retargetingkit.retargeting.hand.optimization import OptimizationHandRetargeterConfig
from retargetingkit.sources.frames import HandFrame, HandPose
from robokit.assets.robots.end_effectors import ability_hand, inspire_hand, sharpa_hand, xhand
from robokit.helpers.hand_retargeting.presets import ability, inspire, sharpa
from robokit.helpers.hand_retargeting.presets import xhand as xhand_preset


RIGHT_HANDS = (
    (
        OptimizationHandRetargeterConfig(
            xhand.URDF_PATH_RIGHT,
            xhand_preset.spec,
            xhand_preset.online,
            "right",
            collision_spheres=xhand.COLLISION_SPHERE_PATH_RIGHT,
            collision_ignore=xhand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT,
        ),
        12,
    ),
    (
        OptimizationHandRetargeterConfig(
            ability_hand.URDF_PATH_RIGHT,
            ability.spec,
            ability.online,
            "right",
            collision_spheres=ability_hand.COLLISION_SPHERE_PATH_RIGHT,
        ),
        6,
    ),
    (
        OptimizationHandRetargeterConfig(
            sharpa_hand.URDF_PATH_RIGHT,
            sharpa.spec,
            sharpa.online,
            "right",
            collision_spheres=sharpa_hand.COLLISION_SPHERE_PATH_RIGHT,
            collision_ignore=sharpa_hand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT,
        ),
        22,
    ),
    (
        OptimizationHandRetargeterConfig(
            inspire_hand.URDF_PATH_RIGHT,
            inspire.spec,
            inspire.online,
            "right",
            collision_spheres=inspire_hand.COLLISION_SPHERE_PATH_RIGHT,
            collision_ignore=inspire_hand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT,
        ),
        6,
    ),
)

LEFT_HANDS = (
    (
        OptimizationHandRetargeterConfig(
            xhand.URDF_PATH_LEFT, xhand_preset.spec_left, xhand_preset.online_left, "left"
        ),
        12,
    ),
    (
        OptimizationHandRetargeterConfig(inspire_hand.URDF_PATH_LEFT, inspire.spec_left, inspire.online_left, "left"),
        6,
    ),
)


def _open_hand_frame(side: str) -> HandFrame:
    points = np.zeros((21, 3), dtype=np.float32)
    for finger, x in enumerate((-0.04, -0.02, 0.0, 0.02, 0.04)):
        for joint in range(4):
            points[1 + finger * 4 + joint] = (x, 0.02 * (joint + 1), 0.0)
    return HandFrame({side: HandPose(points)})  # type: ignore[dict-item]


class TestOptimizationHandRetargeter:
    def test_right_hand_variants(self) -> None:
        frame = _open_hand_frame("right")
        for config, joint_count in RIGHT_HANDS:
            retargeter = config.build("cpu")
            assert retargeter.urdf_path.exists()
            qpos = retargeter.solve(frame).qpos
            assert qpos.shape == (joint_count,)
            assert bool(np.isfinite(qpos).all())

    def test_left_hand_variants(self) -> None:
        frame = _open_hand_frame("left")
        for config, joint_count in LEFT_HANDS:
            qpos = config.build("cpu").solve(frame).qpos
            assert qpos.shape == (joint_count,)
            assert bool(np.isfinite(qpos).all())
