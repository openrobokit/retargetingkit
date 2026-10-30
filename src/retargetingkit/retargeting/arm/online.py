"""Canonical arm-frame to robot-arm-state retargeting."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from retargetingkit.retargeting.retargeter import Retargeter, RetargeterConfig
from retargetingkit.sources.frames import ArmFrame, Side
from retargetingkit.targets.commands import QPosCommand
from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig
from robokit.robo import Robot
from robokit.terms import PositionLimit, PositionTask, RotationTask, SmoothnessTask
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy import inverse_tf_mat, matrix_to_quaternion, pose7_to_tf_mat


@dataclass
class ArmRetargeterConfig(RetargeterConfig[ArmFrame, QPosCommand]):
    """One human wrist mapped onto one robot arm.

    >>> ArmRetargeterConfig(Path("arm.urdf"), "link_eef", (0.0,), "right").translation_scale
    1.5
    """

    urdf: Path
    ee_link: str
    home_qpos: tuple[float, ...]
    side: Side
    translation_scale: float = 1.5
    ee_gain: float = 1.0

    def build(self, device: str) -> "ArmRetargeter":
        """Build the online arm retargeter.

        >>> ArmRetargeterConfig(Path("arm.urdf"), "link_eef", (0.0,), "right").build("cpu")  # doctest: +SKIP
        """
        return ArmRetargeter(self, device)


class ArmRetargeter(Retargeter[ArmFrame, QPosCommand]):
    """One human wrist to one robot arm; the first wrist sample is the calibration origin and maps to the home pose.

    >>> ArmRetargeter(config, "cpu")  # doctest: +SKIP
    """

    def __init__(self, config: "ArmRetargeterConfig", device: str) -> None:
        self.side: Side = config.side
        self.urdf_path = config.urdf
        self.home_qpos = np.asarray(config.home_qpos, dtype=np.float32)
        self._translation_scale = config.translation_scale
        self._ee_gain = config.ee_gain
        self._robot = Robot.load(str(self.urdf_path))
        self._ee_link_index = self._robot.spec.link_names.index(config.ee_link)
        self._ik = IKSolver(
            IKSolverConfig()
            .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
            .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
            .add(PositionLimit(weight=50.0))
            .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=0.3)),
            self._robot,
            config.ee_link,
            device=device,
        )
        self._arm_state = self._robot.state(
            q=wp.from_numpy(self.home_qpos[None], dtype=wp.float32, device=device),
        )
        self._robot.forward_kinematics(self._arm_state)
        self._initial_ee_pose = (
            self._arm_state.get_T_world_link(self._ee_link_index).numpy().squeeze().astype(np.float32)
        )
        self._initial_wrist: np.ndarray | None = None
        self._previous_ee_pose: np.ndarray | None = None

    @staticmethod
    def wrist_to_ee_pose(
        initial_ee_pose: np.ndarray,
        initial_wrist: np.ndarray,
        wrist: np.ndarray,
        translation_scale: float,
    ) -> np.ndarray:
        """Replay the wrist delta since calibration from the home EE pose, scaling translation to robot reach.

        >>> ee = np.array([1, 2, 3, 1, 0, 0, 0], dtype=np.float32)
        >>> identity = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        >>> ArmRetargeter.wrist_to_ee_pose(ee, identity, identity, 1.5)
        array([1., 2., 3., 1., 0., 0., 0.], dtype=float32)
        """
        initial_ee = pose7_to_tf_mat(initial_ee_pose.astype(np.float64))
        initial = pose7_to_tf_mat(initial_wrist.astype(np.float64))
        target = initial_ee @ inverse_tf_mat(initial) @ pose7_to_tf_mat(wrist.astype(np.float64))
        target[:3, 3] = initial_ee_pose[:3] + translation_scale * (wrist[:3] - initial_wrist[:3])
        return np.concatenate([target[:3, 3], matrix_to_quaternion(target[:3, :3])]).astype(np.float32)

    @staticmethod
    def smooth_pose7(previous: np.ndarray, current: np.ndarray, gain: float) -> np.ndarray:
        """EMA-blend two pose7s while keeping quaternions in one hemisphere.

        >>> previous = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        >>> current = np.array([1, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        >>> ArmRetargeter.smooth_pose7(previous, current, 0.5)
        array([0.5, 0. , 0. , 1. , 0. , 0. , 0. ], dtype=float32)
        """
        quat = current[3:] if np.dot(previous[3:], current[3:]) >= 0 else -current[3:]
        blended = previous + gain * (np.concatenate([current[:3], quat]) - previous)
        blended[3:] /= np.linalg.norm(blended[3:])
        return blended

    def solve(self, frame: ArmFrame) -> QPosCommand:
        wrist = frame.wrists[self.side]
        if self._initial_wrist is None:
            self._initial_wrist = wrist.copy()
        ee_pose = self.wrist_to_ee_pose(
            self._initial_ee_pose,
            self._initial_wrist,
            wrist,
            self._translation_scale,
        )
        if self._previous_ee_pose is not None:
            ee_pose = self.smooth_pose7(self._previous_ee_pose, ee_pose, self._ee_gain)
        self._previous_ee_pose = ee_pose.copy()
        target = wp.from_numpy(ee_pose[None, None], dtype=wp_vec7, device=self._arm_state.device)
        self._arm_state = self._ik.solve(target, prev_state=self._arm_state, init_state=self._arm_state)
        return QPosCommand(self._arm_state.q.numpy().squeeze().astype(np.float32))
