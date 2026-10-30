import numpy as np
import torch

from retargetingkit.configs.teleop.arm_hand import _ARM_HAND_TELEOP_CONFIGS
from retargetingkit.retargeting.arm.online import ArmRetargeter, ArmRetargeterConfig
from retargetingkit.sources.frames import ArmFrame
from retargetingkit.targets.arms.xarm6 import XARM6_EE_LINK, XARM6_HOME_QPOS
from retargetingkit.targets.arms.xarm7 import XARM7_EE_LINK, XARM7_HOME_QPOS
from retargetingkit.targets.hands.inspire import InspireHandConfig
from retargetingkit.targets.hands.sharpa_hand import SharpaHandConfig
from retargetingkit.targets.viser import Viser, assembly_qpos_indices
from robokit.assets.robots.arms import xarm6, xarm7
from robokit.assets.robots.assembly import (
    xarm6_sharpa,
    xarm7_ability,
    xarm7_inspire,
    xarm7_sharpa,
    xarm7_xhand,
)
from robokit.assets.robots.end_effectors import sharpa_hand
from robokit.robo import Robot


class TestArmHandPipeline:
    def test_initial_wrist_maps_to_arm_home(self) -> None:
        frame = ArmFrame({"right": np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)})
        arms = (
            (xarm6.URDF_PATH, XARM6_EE_LINK, XARM6_HOME_QPOS),
            (xarm7.URDF_GLB_PATH, XARM7_EE_LINK, XARM7_HOME_QPOS),
        )
        for urdf, ee_link, home_qpos in arms:
            retargeter = ArmRetargeterConfig(urdf, ee_link, home_qpos, "right").build("cpu")
            state = retargeter.solve(frame)
            torch.testing.assert_close(torch.from_numpy(state.qpos), torch.from_numpy(retargeter.home_qpos))

    def test_relative_translation_scale(self) -> None:
        initial_ee = np.array([0.4, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        initial_wrist = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        wrist = np.array([0.1, -0.05, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        expected = np.array([0.6, -0.3, 0.7, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        actual = ArmRetargeter.wrist_to_ee_pose(initial_ee, initial_wrist, wrist, 2.0)

        torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(expected))

    def test_translation_ignores_initial_wrist_rotation(self) -> None:
        initial_ee = np.array([0.4, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        initial_wrist = np.array([0.0, 0.0, 0.0, 2**-0.5, 0.0, 2**-0.5, 0.0], dtype=np.float32)
        wrist = initial_wrist.copy()
        wrist[0] = 0.1
        expected = np.array([0.6, -0.2, 0.3], dtype=np.float32)

        actual = ArmRetargeter.wrist_to_ee_pose(initial_ee, initial_wrist, wrist, 2.0)

        torch.testing.assert_close(torch.from_numpy(actual[:3]), torch.from_numpy(expected))

    def test_relative_rotation(self) -> None:
        initial_ee = np.array([0.4, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        initial_wrist = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        wrist = np.array([0.0, 0.0, 0.0, 2**-0.5, 0.0, 0.0, 2**-0.5], dtype=np.float32)
        expected = torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5], dtype=torch.float32)

        actual = torch.from_numpy(ArmRetargeter.wrist_to_ee_pose(initial_ee, initial_wrist, wrist, 1.0)[3:])

        torch.testing.assert_close(torch.abs(actual @ expected), torch.tensor(1.0), rtol=0.0, atol=1e-6)

    def test_smoothing_blends_in_one_quaternion_hemisphere(self) -> None:
        previous = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        current = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        actual = ArmRetargeter.smooth_pose7(previous, current, 0.5)

        expected = torch.tensor([0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        torch.testing.assert_close(torch.from_numpy(actual), expected)

    def test_unit_gain_returns_current(self) -> None:
        previous = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        current = np.array([0.2, -0.1, 0.4, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        actual = ArmRetargeter.smooth_pose7(previous, current, 1.0)

        torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(current))

    def test_viser_consumes_states_for_all_assemblies(self) -> None:
        assemblies = (
            xarm7_xhand.URDF_GLB_PATH,
            xarm7_ability.URDF_GLB_PATH,
            xarm7_inspire.URDF_PATH,
            xarm7_sharpa.URDF_PATH,
            xarm6_sharpa.URDF_PATH,
        )
        for urdf in assemblies:
            qpos = Robot.load(str(urdf)).spec.zero_q.astype(np.float32)
            visualizer = Viser((urdf,))
            visualizer.set_qpos(qpos)
            visualizer.close()

    def test_xarm7_hand_presets(self) -> None:
        inspire_config = _ARM_HAND_TELEOP_CONFIGS["visionpro_xarm7_inspire"]
        sharpa_config = _ARM_HAND_TELEOP_CONFIGS["visionpro_xarm7_sharpa"]
        assert isinstance(inspire_config.hands["right"], InspireHandConfig)
        assert isinstance(sharpa_config.hands["right"], SharpaHandConfig)
        assert inspire_config.assembly_urdf == xarm7_inspire.URDF_PATH
        assert sharpa_config.assembly_urdf == xarm7_sharpa.URDF_PATH

    def test_xarm7_sharpa_assembly_joint_order(self) -> None:
        order = assembly_qpos_indices(
            xarm7_sharpa.URDF_PATH,
            (xarm7.URDF_GLB_PATH, sharpa_hand.URDF_PATH_RIGHT),
        )
        component_names = (
            *Robot.load(str(xarm7.URDF_GLB_PATH)).spec.actuated_joint_names,
            *Robot.load(str(sharpa_hand.URDF_PATH_RIGHT)).spec.actuated_joint_names,
        )
        assembly_names = Robot.load(str(xarm7_sharpa.URDF_PATH)).spec.actuated_joint_names
        assert tuple(np.asarray(component_names)[order]) == tuple(assembly_names)
        assert not np.array_equal(order, np.arange(len(order)))

    def test_dual_arm_hand_viser_order_is_left_then_right(self) -> None:
        config = _ARM_HAND_TELEOP_CONFIGS["visionpro_dual_xarm7_xhand"]
        assert tuple(config.arms) == tuple(config.hands) == ("left", "right")
        assert tuple(config.arm_retargeters) == tuple(config.hand_retargeters) == ("left", "right")
