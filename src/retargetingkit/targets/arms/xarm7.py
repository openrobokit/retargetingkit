"""xArm7 servo controller using multiprocessing."""

import ctypes
import enum
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray

import numpy as np
import warp as wp
from jaxtyping import Float, Float64

from retargetingkit.targets.target import Target, TargetConfig
from robokit.assets.robots.arms import xarm7
from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig
from robokit.robo import Robot
from robokit.terms import PositionLimit, PositionTask, RotationTask, SmoothnessTask
from robokit.utils.warp_utils import wp_vec7


NUM_JOINTS = 7
XARM7_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, NUM_JOINTS + 1))
XARM7_EE_LINK = "link_eef"
XARM7_HOME_QPOS: tuple[float, ...] = (-0.03, 0.13, 0.03, 0.24, 3.14, 1.46, 0.00)
"""Default reference joint pose used by teleop/visualization scripts."""


class XArmMode(enum.IntEnum):
    SERVO = 1
    """Teleoperation: sends positions directly, needs smooth input at ≥250Hz."""
    TEACHING = 2
    """Drag-to-teach: manual mode with gravity compensation, no servo commands."""
    VELOCITY = 4
    """Policy rollout: PD-smoothed velocity control, safer with noisy input."""


@dataclass
class XArm7Config(TargetConfig):
    """xArm7 controller configuration."""

    ip: str
    """IP address of the xArm controller."""
    mode: XArmMode = XArmMode.VELOCITY
    """Control mode."""
    dt: float = 1.0 / 250
    """Servo loop timestep in seconds."""
    ee_link: str = XARM7_EE_LINK
    """End effector link name for IK."""
    kp: np.ndarray = field(default_factory=lambda: np.array([10.0, 10.0, 5.0, 5.0, 5.0, 5.0, 5.0]))
    """Proportional gains for velocity mode."""
    kd: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25]))
    """Derivative gains for velocity mode."""
    max_velocity: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 3.0]))
    """Maximum joint velocities (rad/s), used by both SERVO and VELOCITY modes."""

    def build(self) -> "XArm7":
        arm = XArm7(self)
        arm.start()
        return arm


class XArm7(Target):
    """xArm7 controller with 250Hz servo loop in a child process.

    >>> # XArm7(XArm7Config("192.168.1.242"))  # doctest: +SKIP
    """

    def __init__(self, config: XArm7Config):
        self._config = config
        self._target = mp.Array(ctypes.c_double, NUM_JOINTS)
        self._current = mp.Array(ctypes.c_double, NUM_JOINTS)
        self._running = mp.Value(ctypes.c_bool, True)
        self._ready = mp.Value(ctypes.c_bool, False)
        self._process: mp.Process | None = None

        wp.init()
        self._device = "cuda" if wp.is_cuda_available() else "cpu"
        self._robot = Robot.load(str(xarm7.URDF_GLB_PATH))
        self._ee_link_idx = self._robot.spec.link_names.index(config.ee_link)
        ik_config = (
            IKSolverConfig()
            .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
            .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
            .add(PositionLimit(weight=50.0))
            .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=0.3))
        )
        self._ik_helper = IKSolver(ik_config, self._robot, config.ee_link, device=self._device)

    def start(self):
        self._process = mp.Process(
            target=XArm7._servo_loop,
            args=(self._config, self._target, self._current, self._running, self._ready),
            daemon=True,
        )
        self._process.start()
        while not self._ready.value:
            time.sleep(0.01)
        self._home_qpos = self.get_qpos()
        self._prev_state = self._robot.state(
            q=wp.from_numpy(self._home_qpos.astype(np.float32), dtype=wp.float32, device=self._device)
        )

    @property
    def home_qpos(self) -> np.ndarray:
        """Joint positions recorded at start()."""
        return self._home_qpos.copy()

    def close(self):
        self._running.value = False
        if self._process:
            self._process.join(timeout=5.0)

    def set_qpos(self, qpos: Float[np.ndarray, "7"]):
        q = np.asarray(qpos, dtype=np.float32).reshape(-1)
        assert q.shape == (NUM_JOINTS,), f"qpos shape must be ({NUM_JOINTS},), got {q.shape}"
        assert np.isfinite(q).all(), "qpos must be finite"
        self._target[:] = q
        self._prev_state = self._robot.state(q=wp.from_numpy(q[None, :], dtype=wp.float32, device=self._device))

    def get_qpos(self) -> Float64[np.ndarray, "7"]:
        qpos = np.array(self._current[:])
        assert qpos.shape == (NUM_JOINTS,), f"qpos shape must be ({NUM_JOINTS},), got {qpos.shape}"
        assert qpos.dtype == np.float64, f"qpos dtype must be float64, got {qpos.dtype}"
        assert np.isfinite(qpos).all(), "qpos must be finite"
        return qpos

    def get_ee_pose(self) -> Float64[np.ndarray, "7"]:
        """Return end-effector pose `[x, y, z, qw, qx, qy, qz]` via FK."""
        state = self._robot.state(q=self.get_qpos())
        self._robot.forward_kinematics(state)
        pose = state.get_T_world_link(self._ee_link_idx).numpy().astype(np.float64).squeeze()
        assert pose.shape == (7,), f"ee pose shape must be (7,), got {pose.shape}"
        assert pose.dtype == np.float64, f"ee pose dtype must be float64, got {pose.dtype}"
        assert np.isfinite(pose).all(), "ee pose must be finite"
        return pose

    def set_eepose(self, pose: Float[np.ndarray, "7"]):
        """Set target via ee pose `[x, y, z, qw, qx, qy, qz]`."""
        pose = np.asarray(pose, dtype=np.float32)
        assert pose.shape == (7,), f"ee pose shape must be (7,), got {pose.shape}"
        assert pose.dtype == np.float32, f"ee pose dtype must be float32, got {pose.dtype}"
        assert np.isfinite(pose).all(), "ee pose must be finite"
        target = wp.from_numpy(pose.astype(np.float32)[None, None], dtype=wp_vec7, device=self._device)
        self._prev_state = self._ik_helper.solve(target, prev_state=self._prev_state, init_state=self._prev_state)
        self.set_qpos(self._prev_state.q.numpy().squeeze())

    @staticmethod
    def _servo_loop(
        config: XArm7Config,
        target: SynchronizedArray,
        current: SynchronizedArray,
        running: Synchronized,
        ready: Synchronized,
    ) -> None:
        from xarm import XArmAPI  # type: ignore[import-unresolved]

        arm = XArmAPI(config.ip, is_radian=True)
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(state=0)
        arm.clean_warn()
        arm.clean_error()
        arm.set_mode(config.mode)
        arm.set_state(state=0)
        time.sleep(0.5)

        code, state = arm.get_joint_states(is_radian=True)
        assert code == 0, f"get_joint_states failed: code={code}"
        qpos = np.array(state[0])
        assert qpos.shape == (NUM_JOINTS,), f"qpos shape must be ({NUM_JOINTS},), got {qpos.shape}"
        assert qpos.dtype == np.float64, f"qpos dtype must be float64, got {qpos.dtype}"
        assert np.isfinite(qpos).all(), "qpos must be finite"
        target[:] = qpos
        current[:] = qpos
        ready.value = True

        prev_err = np.zeros(NUM_JOINTS)

        while running.value:
            time.sleep(config.dt)
            target_arr = np.array(target[:])

            code, arm_state = arm.get_state()
            if code != 0:
                print(f"xArm error: code={code}, state={arm_state}")
                break

            code, joint_state = arm.get_joint_states(is_radian=True)
            assert code == 0, f"get_joint_states failed: code={code}"
            qpos = np.array(joint_state[0])
            assert qpos.shape == (NUM_JOINTS,), f"qpos shape must be ({NUM_JOINTS},), got {qpos.shape}"
            assert qpos.dtype == np.float64, f"qpos dtype must be float64, got {qpos.dtype}"
            assert np.isfinite(qpos).all(), "qpos must be finite"
            current[:] = qpos

            if config.mode == XArmMode.TEACHING:
                continue
            if config.mode == XArmMode.SERVO:
                err = target_arr - qpos
                scale = np.max(np.abs(err) / (config.max_velocity * config.dt))
                if scale > 1:
                    target_arr = qpos + err / scale
                arm.set_servo_angle_j(angles=target_arr)
            else:
                err = target_arr - qpos
                qvel = config.kp * err + config.kd * (err - prev_err) / config.dt
                prev_err = err.copy()
                scale = np.max(np.abs(qvel) / config.max_velocity)
                if scale > 1:
                    qvel /= scale
                arm.vc_set_joint_velocity(qvel)

        if config.mode == XArmMode.VELOCITY:
            arm.vc_set_joint_velocity(np.zeros(NUM_JOINTS))
        arm.disconnect()
