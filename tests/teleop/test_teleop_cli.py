import importlib
import sys

import pytest


class TestTeleopCli:
    @pytest.mark.parametrize(
        ("module_name", "rig"),
        [
            ("scripts.teleop.hand", "manus_xhand"),
            ("scripts.teleop.arm", "visionpro_xarm7"),
            ("scripts.teleop.arm_hand", "visionpro_xarm7_xhand"),
            ("scripts.teleop.humanoid", "bvh_g1_sonic"),
            ("scripts.teleop.humanoid_hand", "pico_g1_inspire"),
        ],
    )
    def test_root_help(
        self,
        module_name: str,
        rig: str,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        module = importlib.import_module(module_name)
        monkeypatch.setattr(sys, "argv", [f"{module_name}.py", "--help"])

        module.main()

        assert rig in capsys.readouterr().out

    def test_arm_group_help(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from scripts.teleop import arm

        monkeypatch.setattr(sys, "argv", ["arm.py", "visionpro_xarm7", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            arm.main()

        assert exc_info.value.code == 0
        output = capsys.readouterr().out
        assert "--source.ip" in output
        assert "--arms.right.ip" in output
        assert "--mode" in output

    def test_arm_hand_group_help(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from scripts.teleop import arm_hand

        monkeypatch.setattr(sys, "argv", ["arm_hand.py", "visionpro_xarm7_xhand", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            arm_hand.main()

        assert exc_info.value.code == 0
        output = capsys.readouterr().out
        assert "--arms.right.ip" in output
        assert "--hands.right.serial-port" in output
        assert "--mode" in output

    def test_hand_group_help(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from scripts.teleop import hand

        monkeypatch.setattr(sys, "argv", ["hand.py", "manus_xhand", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            hand.main()

        assert exc_info.value.code == 0
        output = capsys.readouterr().out
        assert "--mode" in output
        assert "--source.left-id" in output

    def test_humanoid_group_help(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from scripts.teleop import humanoid

        monkeypatch.setattr(sys, "argv", ["humanoid.py", "bvh_g1_sonic", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            humanoid.main()

        assert exc_info.value.code == 0
        output = capsys.readouterr().out
        assert "--source.path" in output
        assert "--retargeter.z-offset" in output
        assert "--targets.0.port" in output

    def test_humanoid_hand_group_help(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from scripts.teleop import humanoid_hand

        monkeypatch.setattr(sys, "argv", ["humanoid_hand.py", "pico_g1_inspire", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            humanoid_hand.main()

        assert exc_info.value.code == 0
        output = capsys.readouterr().out
        assert "--source.human-height" in output
        assert "--targets.0.port" in output
        assert "--left-hand.serial-port" in output
        assert "--right-hand.serial-port" in output
