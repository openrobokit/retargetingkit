"""RBY1-A right-arm servo controller using multiprocessing."""

import ctypes
import enum
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray

import numpy as np
import warp as wp

from retargetingkit.targets.target import Target, TargetConfig
from robokit.assets import fetch
from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig
from robokit.robo import Robot
from robokit.terms import PositionLimit, PositionTask, RotationTask, SmoothnessTask
from robokit.utils.warp_utils import wp_vec7


NUM_ARM_JOINTS = 7
RIGHT_ARM_SLICE = slice(8, 15)
NUM_HEAD_JOINTS = 2
HEAD_SLICE = slice(22, 24)


class RBY1AMode(enum.IntEnum):
    SERVO = 1
    """Teleoperation: sends positions directly, needs smooth input at ≥100Hz."""
    VELOCITY = 4
    """Policy rollout: PD-smoothed velocity control, safer with noisy input."""
    TEACHING = 5
    """Kinesthetic teaching: per-arm gravity compensation; user drags the weightless arm."""


@dataclass
class RBY1AConfig(TargetConfig):
    """RBY1-A right-arm controller configuration."""

    ip: str
    """IP address with port (e.g. `192.168.30.1:50051`)."""
    mode: RBY1AMode = RBY1AMode.VELOCITY
    """Control mode."""
    dt: float = 1.0 / 100
    """Servo loop timestep in seconds."""
    ee_link: str = "link_right_arm_6"
    """End effector link name for IK."""
    cutoff_frequency: float = 5.0
    """SDK-side position command smoothing frequency (Hz)."""
    kp: np.ndarray = field(default_factory=lambda: np.array([10.0, 10.0, 5.0, 5.0, 5.0, 5.0, 5.0]))
    """Proportional gains for velocity mode."""
    kd: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25]))
    """Derivative gains for velocity mode."""
    max_velocity: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 3.0]))
    """Maximum joint velocities (rad/s), used by both SERVO and VELOCITY modes."""
    home_qpos: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, -np.pi / 2, -np.pi / 2, 0, 0]))
    """Target joint positions for the home configuration (rad)."""
    head_max_velocity: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))
    """Maximum head joint velocities (rad/s)."""
    head_home_qpos: np.ndarray = field(default_factory=lambda: np.zeros(NUM_HEAD_JOINTS))
    """Target head joint positions for the home configuration (rad)."""

    def build(self) -> "RBY1A":
        """Connect to the robot and start its servo loop.

        >>> RBY1AConfig("192.168.30.1:50051").build()  # doctest: +SKIP
        """
        target = RBY1A(self)
        target.start()
        return target


class RBY1A(Target):
    """RBY1-A right-arm controller with servo loop in a child process.

    >>> # RBY1A(RBY1AConfig("192.168.30.1:50051"))  # doctest: +SKIP
    """

    def __init__(self, config: RBY1AConfig):
        self._config = config
        self._target = mp.Array(ctypes.c_double, NUM_ARM_JOINTS)
        self._current = mp.Array(ctypes.c_double, NUM_ARM_JOINTS)
        self._head_target = mp.Array(ctypes.c_double, NUM_HEAD_JOINTS)
        self._head_current = mp.Array(ctypes.c_double, NUM_HEAD_JOINTS)
        self._running = mp.Value(ctypes.c_bool, True)
        self._ready = mp.Value(ctypes.c_bool, False)
        self._process: mp.Process | None = None

        wp.init()
        self._device = "cuda" if wp.is_cuda_available() else "cpu"
        rby1_base = "robots/robot_description/assembly/rby1_sharpa"
        self._robot = Robot.load(str(fetch([f"{rby1_base}/**"]) / f"{rby1_base}/rby1_sharpa_right.urdf"))
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
            target=RBY1A._servo_loop,
            args=(
                self._config,
                self._target,
                self._current,
                self._head_target,
                self._head_current,
                self._running,
                self._ready,
            ),
            daemon=True,
        )
        self._process.start()
        while not self._ready.value:
            time.sleep(0.01)
        self._home_qpos = self.get_qpos()
        self._head_home_qpos = self.get_head_qpos()
        full_q = np.zeros(self._robot.spec.num_actuated_joints, dtype=np.float32)
        full_q[:NUM_ARM_JOINTS] = self._home_qpos
        self._prev_state = self._robot.state(q=wp.from_numpy(full_q, dtype=wp.float32, device=self._device))

    @property
    def home_qpos(self) -> np.ndarray:
        """Joint positions recorded at start()."""
        return self._home_qpos.copy()

    def close(self) -> None:
        self._running.value = False
        if self._process:
            self._process.join(timeout=5.0)

    def set_qpos(self, qpos: np.ndarray):
        self._target[:] = qpos

    def get_qpos(self) -> np.ndarray:
        return np.array(self._current[:])

    @property
    def head_home_qpos(self) -> np.ndarray:
        """Head joint positions recorded at start()."""
        return self._head_home_qpos.copy()

    def set_head_qpos(self, qpos: np.ndarray):
        self._head_target[:] = qpos

    def get_head_qpos(self) -> np.ndarray:
        return np.array(self._head_current[:])

    def get_ee_pose(self) -> np.ndarray:
        """Return end-effector pose `[x, y, z, qw, qx, qy, qz]` via FK."""
        full_q = np.zeros(self._robot.spec.num_actuated_joints, dtype=np.float32)
        full_q[:NUM_ARM_JOINTS] = self.get_qpos()
        state = self._robot.state(q=full_q)
        self._robot.forward_kinematics(state)
        return state.get_T_world_link(self._ee_link_idx).numpy().squeeze()

    def set_eepose(self, pose: np.ndarray):
        """Set target via ee pose `[x, y, z, qw, qx, qy, qz]`."""
        target = wp.from_numpy(pose.astype(np.float32)[None, None], dtype=wp_vec7, device=self._device)
        self._prev_state = self._ik_helper.solve(target, prev_state=self._prev_state, init_state=self._prev_state)
        self.set_qpos(self._prev_state.q.numpy().squeeze()[:NUM_ARM_JOINTS])

    @staticmethod
    def _servo_loop(
        config: RBY1AConfig,
        target: SynchronizedArray,
        current: SynchronizedArray,
        head_target: SynchronizedArray,
        head_current: SynchronizedArray,
        running: Synchronized,
        ready: Synchronized,
    ) -> None:
        import rby1_sdk  # type: ignore[import-untyped]

        robot = rby1_sdk.create_robot_a(config.ip)
        robot.connect()
        robot.power_on(".*")
        robot.servo_on(".*")
        robot.reset_fault_control_manager()
        robot.enable_control_manager()
        robot.set_parameter("joint_position_command.cutoff_frequency", str(config.cutoff_frequency))
        time.sleep(0.5)

        state = robot.get_state()
        body_qpos = np.array(state.position).flatten()
        arm_qpos = body_qpos[RIGHT_ARM_SLICE]
        head_qpos = body_qpos[HEAD_SLICE]
        target[:] = arm_qpos
        current[:] = arm_qpos
        head_target[:] = head_qpos
        head_current[:] = head_qpos
        ready.value = True

        stream = robot.create_command_stream()
        prev_err = np.zeros(NUM_ARM_JOINTS)
        head_cmd = head_qpos.copy()

        while running.value:
            time.sleep(config.dt)
            target_arr = np.array(target[:])

            state = robot.get_state()
            body_qpos = np.array(state.position)
            arm_qpos = body_qpos[RIGHT_ARM_SLICE]
            head_qpos = body_qpos[HEAD_SLICE]
            current[:] = arm_qpos
            head_current[:] = head_qpos

            head_target_arr = np.array(head_target[:])
            err_h = head_target_arr - head_cmd
            scale_h = np.max(np.abs(err_h) / (config.head_max_velocity * config.dt))
            if scale_h > 1:
                head_cmd = head_cmd + err_h / scale_h
            else:
                head_cmd = head_target_arr
            head_cmd_builder = rby1_sdk.HeadCommandBuilder(
                rby1_sdk.JointPositionCommandBuilder()
                .set_command_header(rby1_sdk.CommandHeaderBuilder().set_control_hold_time(1.0))
                .set_minimum_time(config.dt)
                .set_position(head_cmd.tolist())
            )

            if config.mode == RBY1AMode.TEACHING:
                arm_cmd = rby1_sdk.ArmCommandBuilder(
                    rby1_sdk.GravityCompensationCommandBuilder()
                    .set_command_header(rby1_sdk.CommandHeaderBuilder().set_control_hold_time(1.0))
                    .set_on(True)
                )
            else:
                if config.mode == RBY1AMode.SERVO:
                    err = target_arr - arm_qpos
                    scale = np.max(np.abs(err) / (config.max_velocity * config.dt))
                    if scale > 1:
                        target_arr = arm_qpos + err / scale
                else:
                    err = target_arr - arm_qpos
                    qvel = config.kp * err + config.kd * (err - prev_err) / config.dt
                    prev_err = err.copy()
                    scale = np.max(np.abs(qvel) / config.max_velocity)
                    if scale > 1:
                        qvel /= scale
                    target_arr = arm_qpos + qvel * config.dt
                arm_cmd = rby1_sdk.ArmCommandBuilder(
                    rby1_sdk.JointPositionCommandBuilder()
                    .set_command_header(rby1_sdk.CommandHeaderBuilder().set_control_hold_time(1.0))
                    .set_minimum_time(config.dt)
                    .set_position(target_arr.tolist())
                )

            cmd = rby1_sdk.RobotCommandBuilder().set_command(
                rby1_sdk.ComponentBasedCommandBuilder()
                .set_body_command(
                    rby1_sdk.BodyCommandBuilder(
                        rby1_sdk.BodyComponentBasedCommandBuilder().set_right_arm_command(arm_cmd)
                    )
                )
                .set_head_command(head_cmd_builder)
            )
            stream.send_command(cmd)

        robot.disable_control_manager()
        robot.disconnect()
