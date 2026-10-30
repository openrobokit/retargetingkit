import sys
import types

import numpy as np
import torch

from retargetingkit.targets.humanoids.dexmate import _COMPONENTS, Dexmate, DexmateConfig


class _FakeComponent:
    def __init__(self, dof: int, base: float):
        self._q = base + np.arange(dof, dtype=np.float32)

    def get_joint_pos(self):
        return self._q


class _FakeRobot:
    def __init__(self, auto_shutdown: bool = True):
        self.auto_shutdown = auto_shutdown
        self.torso = _FakeComponent(3, 0.0)
        self.head = _FakeComponent(3, 10.0)
        self.left_arm = _FakeComponent(7, 20.0)
        self.right_arm = _FakeComponent(7, 30.0)
        self.calls = []

    def set_joint_pos(self, targets, wait_time=0.0, wait_kwargs=None):
        self.calls.append((targets, wait_time, wait_kwargs))

    def shutdown(self):
        self.shut = True


def _make() -> Dexmate:
    fake = types.ModuleType("dexcontrol.robot")
    fake.Robot = _FakeRobot
    sys.modules["dexcontrol"] = types.ModuleType("dexcontrol")
    sys.modules["dexcontrol.robot"] = fake
    return Dexmate(DexmateConfig(auto_shutdown=False))


class TestDexmate:
    def test_qpos_roundtrip_and_names(self) -> None:
        d = _make()
        assert d._robot.auto_shutdown is False
        assert len(d.joint_names) == sum(dof for _, dof in _COMPONENTS) == 20
        assert d.joint_names[13] == "R_arm_j1"
        q = d.get_qpos()
        assert q.shape == (20,) and q[[0, 3, 6, 13]].tolist() == [0.0, 10.0, 20.0, 30.0]

        target = np.arange(20, dtype=np.float32)
        d.set_qpos(target, wait_time=2.0, max_vel=0.2)
        targets, wait_time, wait_kwargs = d._robot.calls[-1]
        assert wait_time == 2.0 and wait_kwargs == {"max_vel": 0.2}
        assert list(targets) == ["torso", "head", "left_arm", "right_arm"]
        torch.testing.assert_close(
            torch.from_numpy(targets["right_arm"]), torch.from_numpy(target[13:20]), rtol=0, atol=0
        )

        d.set_qpos(target)
        assert d._robot.calls[-1][1:] == (0.0, None)
