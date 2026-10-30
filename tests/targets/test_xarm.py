"""FK/IK round-trip tests for xArm7 and xArm6."""

import numpy as np
import torch
import warp as wp

from robokit.assets.robots.arms import xarm6, xarm7
from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig
from robokit.robo import Robot
from robokit.terms import PositionLimit, PositionTask, RotationTask, SmoothnessTask
from robokit.utils.warp_utils import wp_vec7


URDF = str(xarm7.URDF_GLB_PATH)
EE_LINK = "link_eef"
XARM6_URDF = str(xarm6.URDF_PATH)
XARM6_EE_LINK = "link6"


class TestFKIKRoundTrip:
    """FK -> IK -> FK consistency for xArm7."""

    @classmethod
    def setup_class(cls):
        wp.init()
        cls.device = "cuda" if wp.is_cuda_available() else "cpu"
        cls.robot = Robot.load(URDF)
        cls.ee_idx = cls.robot.spec.link_names.index(EE_LINK)
        ik_config = (
            IKSolverConfig()
            .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
            .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
            .add(PositionLimit(weight=50.0))
            .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=0.3))
        )
        cls.ik = IKSolver(ik_config, cls.robot, EE_LINK, device=cls.device)

    def _fk(self, q: np.ndarray) -> np.ndarray:
        state = self.robot.state(q=q)
        self.robot.forward_kinematics(state)
        return state.get_T_world_link(self.ee_idx).numpy().squeeze()

    def _ik(self, pose: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        target = wp.from_numpy(pose.astype(np.float32).reshape(1, 1, 7), dtype=wp_vec7, device=self.device)
        init_state = self.robot.state(
            q=wp.from_numpy(q_init.astype(np.float32).reshape(1, -1), dtype=wp.float32, device=self.device),
        )
        result = self.ik.solve(target, prev_state=init_state, init_state=init_state)
        return result.q.numpy().squeeze()

    def test_round_trip(self):
        q = np.array([0.1, -0.3, 0.2, 0.5, -0.1, 0.4, 0.0], dtype=np.float32)
        pose = self._fk(q)
        q_solved = self._ik(pose, q)
        pose_recovered = self._fk(q_solved)
        # Position within 1mm
        torch.testing.assert_close(
            torch.from_numpy(pose_recovered[:3]), torch.from_numpy(pose[:3]), atol=1e-3, rtol=0.0
        )
        # Quaternion: |dot| ~ 1 (handles q ~ -q ambiguity)
        torch.testing.assert_close(
            torch.tensor(np.abs(pose_recovered[3:] @ pose[3:])), torch.tensor(1.0), atol=1e-3, rtol=0.0
        )


class TestXArm6FKIKRoundTrip:
    """FK -> IK -> FK consistency for xArm6."""

    @classmethod
    def setup_class(cls):
        wp.init()
        cls.device = "cuda" if wp.is_cuda_available() else "cpu"
        cls.robot = Robot.load(XARM6_URDF)
        cls.ee_idx = cls.robot.spec.link_names.index(XARM6_EE_LINK)
        ik_config = (
            IKSolverConfig()
            .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
            .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
            .add(PositionLimit(weight=50.0))
            .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=0.3))
        )
        cls.ik = IKSolver(ik_config, cls.robot, XARM6_EE_LINK, device=cls.device)

    def _fk(self, q: np.ndarray) -> np.ndarray:
        state = self.robot.state(q=q)
        self.robot.forward_kinematics(state)
        return state.get_T_world_link(self.ee_idx).numpy().squeeze()

    def _ik(self, pose: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        target = wp.from_numpy(pose.astype(np.float32).reshape(1, 1, 7), dtype=wp_vec7, device=self.device)
        init_state = self.robot.state(
            q=wp.from_numpy(q_init.astype(np.float32).reshape(1, -1), dtype=wp.float32, device=self.device),
        )
        result = self.ik.solve(target, prev_state=init_state, init_state=init_state)
        return result.q.numpy().squeeze()

    def test_round_trip(self):
        q = np.array([0.1, -0.3, 0.2, 0.5, -0.1, 0.4], dtype=np.float32)
        pose = self._fk(q)
        q_solved = self._ik(pose, q)
        pose_recovered = self._fk(q_solved)
        # Position within 1mm
        torch.testing.assert_close(
            torch.from_numpy(pose_recovered[:3]), torch.from_numpy(pose[:3]), atol=1e-3, rtol=0.0
        )
        # Quaternion: |dot| ~ 1 (handles q ~ -q ambiguity)
        torch.testing.assert_close(
            torch.tensor(np.abs(pose_recovered[3:] @ pose[3:])), torch.tensor(1.0), atol=1e-3, rtol=0.0
        )
