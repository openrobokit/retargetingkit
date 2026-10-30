import numpy as np
import warp as wp

from retargetingkit.assets import sample_bvh
from retargetingkit.configs.teleop.humanoid import _HUMANOID_TELEOP_CONFIGS
from retargetingkit.retargeting.humanoid.presets.g1 import g1, g1_aligned
from retargetingkit.retargeting.humanoid.presets.g1_pico import g1_pico
from retargetingkit.sources.misc.human_frame_file import BodyFileSource
from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnline
from robokit.helpers.humanoid_retargeting.loaders import load_bvh_motion
from scripts.teleop.humanoid import RATE_HZ


class TestHumanoidPresets:
    def test_pipeline_shape(self) -> None:
        config = _HUMANOID_TELEOP_CONFIGS["bvh_g1_sonic"]
        assert type(config.source).__name__ == "BodyFileSourceConfig"
        assert config.retargeter.preset is g1_aligned
        assert len(config.targets) == 1

    def test_fixed_runtime_rate(self) -> None:
        assert RATE_HZ == 50.0

    def test_bvh_source_builds_directly(self) -> None:
        source = _HUMANOID_TELEOP_CONFIGS["bvh_g1_sonic"].source.build()
        assert isinstance(source, BodyFileSource)
        assert source.num_frames > 0

    def test_presets_are_explicit(self) -> None:
        assert _HUMANOID_TELEOP_CONFIGS["pico_g1_sonic"].retargeter.preset is g1_pico
        assert _HUMANOID_TELEOP_CONFIGS["picofile_g1_sonic"].retargeter.preset is g1_pico

    def test_presets_keep_ankle_foot_targets(self) -> None:
        for name, config in _HUMANOID_TELEOP_CONFIGS.items():
            assert config.source.use_ankle_as_foot_target, name

    def test_target_list_composes_sonic_and_viser(self) -> None:
        config = _HUMANOID_TELEOP_CONFIGS["bvh_g1_sonic_viser"]
        assert [type(target).__name__ for target in config.targets] == ["SonicTargetConfig", "ViserConfig"]

    def test_g1_one_frame(self) -> None:
        wp.init()
        device = "cuda:0" if wp.is_cuda_available() else "cpu"

        for config in (g1, g1_aligned, g1_pico):
            robot = config.load_robot()
            retargeter = HumanoidRetargeterOnline(config, device=device, robot=robot)
            motion, _ = load_bvh_motion(sample_bvh.EXAMPLE_BVH, retargeter.human_joint_names)
            retargeter.warmup(1)
            robot_pose = retargeter.solve_numpy(motion[None, 0], np.array([1.75], dtype=np.float32))[0]

            assert robot_pose.shape == (7 + robot.spec.num_actuated_joints,)
            assert np.isfinite(robot_pose).all()
