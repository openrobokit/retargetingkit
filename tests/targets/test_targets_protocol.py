from retargetingkit.targets.hands.inspire import InspireHand, InspireHandConfig
from robokit.assets.robots.end_effectors import inspire_hand
from robokit.robo.robot_spec import RobotSpec


class TestInspireTarget:
    def test_inspire_hand_declares_target_interface(self) -> None:
        # dir() inspects the class without opening the serial port.
        assert {"set_qpos", "get_qpos", "joint_names", "close"}.issubset(dir(InspireHand))
        assert len(InspireHand.JOINT_NAMES) == 6

    def test_inspire_config_builds_hand_target(self) -> None:
        config = InspireHandConfig("/dev/ttyUSB0", 1, 115200)
        assert config.serial_port == "/dev/ttyUSB0"

    def test_retargeter_and_target_agree(self) -> None:
        # Same joint names in the same order, so no joint matching is needed.
        for urdf in (inspire_hand.URDF_PATH_LEFT, inspire_hand.URDF_PATH_RIGHT):
            robot = RobotSpec.parse(urdf, load_meshes=False, mesh_dir=None)
            assert tuple(robot.actuated_joint_names) == InspireHand.JOINT_NAMES
