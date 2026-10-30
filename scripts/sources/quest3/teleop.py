"""Quest3 controller poses via NVIDIA Isaac Teleop / CloudXR -> xArm SDK or Viser URDF end-effector target (no ROS2)."""

import argparse
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp
from viser.extras import ViserUrdf

from retargetingkit.sources.vr.quest3 import LEFT, RIGHT, Quest3Config
from retargetingkit.targets.arms.xarm6 import (
    XARM6_EE_LINK,
    XARM6_HOME_QPOS,
    XArm6,
    XArm6Config,
)
from retargetingkit.targets.arms.xarm7 import (
    XARM7_EE_LINK,
    XARM7_HOME_QPOS,
    XArm7,
    XArm7Config,
    XArmMode,
)
from robokit.assets.robots.arms import xarm6 as xarm6_assets
from robokit.assets.robots.arms import xarm7 as xarm7_assets
from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig
from robokit.robo import Robot
from robokit.terms import PositionLimit, PositionTask, RotationTask, SmoothnessTask
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy import matrix_to_quaternion, pose7_to_tf_mat, tf_mat_to_pose7


ARM_SPECS = {
    "xarm7": (XArm7, XArm7Config, XARM7_EE_LINK, XARM7_HOME_QPOS, str(xarm7_assets.URDF_GLB_PATH)),
    "xarm6": (XArm6, XArm6Config, XARM6_EE_LINK, XARM6_HOME_QPOS, str(xarm6_assets.URDF_PATH)),
}

DEFAULT_XARM_IP = {RIGHT: "192.168.1.242", LEFT: "192.168.1.209"}
CONTROLLER_TO_XARM_POSITION_R = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
CONTROLLER_TO_XARM_ORIENTATION_R = {
    RIGHT: np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float64),
    LEFT: np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64),
}


@dataclass
class ArmRuntime:
    side: str
    target: Any
    current_frame: Any
    controller_frame: Any
    target_frame: Any
    controller_to_robot_position_R: np.ndarray
    controller_to_robot_orientation_R: np.ndarray
    controller_init_T: np.ndarray | None = None
    robot_init_T: np.ndarray | None = None
    stream_enabled: bool = False
    calibrate_prev: bool = False
    pause_prev: bool = False
    reset_prev: bool = False


class ViserArm:
    """xArm-compatible Viser preview used by this playground."""

    def __init__(
        self,
        server: Any,
        urdf_path: str,
        ee_link: str,
        root_node_name: str,
        root_position: tuple[float, float, float] | None,
    ) -> None:
        if root_position is not None:
            server.scene.add_frame(root_node_name, show_axes=False, position=root_position)
        self._urdf = ViserUrdf(server, Path(urdf_path), root_node_name=root_node_name)
        self._qpos = np.zeros(len(self._urdf.get_actuated_joint_limits()))

        wp.init()
        self._device = "cuda" if wp.is_cuda_available() else "cpu"
        self._robot = Robot.load(urdf_path)
        self._ee_link_idx = self._robot.spec.link_names.index(ee_link)
        ik_config = (
            IKSolverConfig()
            .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
            .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
            .add(PositionLimit(weight=50.0))
            .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=0.3))
        )
        self._ik = IKSolver(ik_config, self._robot, ee_link, device=self._device)
        self._state = self._robot.state(
            q=wp.from_numpy(self._qpos.astype(np.float32), dtype=wp.float32, device=self._device)
        )

    def set_qpos(self, qpos: np.ndarray) -> None:
        self._qpos = qpos.copy()
        self._urdf.update_cfg(qpos)
        self._state = self._robot.state(q=wp.from_numpy(qpos.astype(np.float32), dtype=wp.float32, device=self._device))

    def get_ee_pose(self) -> np.ndarray:
        self._robot.forward_kinematics(self._state)
        return self._state.get_T_world_link(self._ee_link_idx).numpy().squeeze()

    def set_eepose(self, pose: np.ndarray) -> None:
        target = wp.from_numpy(pose.astype(np.float32)[None, None], dtype=wp_vec7, device=self._device)
        self._state = self._ik.solve(target, prev_state=self._state, init_state=self._state)
        self.set_qpos(self._state.q.numpy().squeeze())


def compute_xarm_target_T(
    controller_init_T: np.ndarray,
    controller_now_T: np.ndarray,
    robot_init_T: np.ndarray,
    controller_to_robot_position_R: np.ndarray,
    controller_to_robot_orientation_R: np.ndarray,
    translation_scale: float,
) -> np.ndarray:
    """Map Isaac Teleop controller movement to a robot end-effector target.

    >>> T = np.eye(4)
    >>> target = compute_xarm_target_T(T, T, T, np.eye(3), np.eye(3), 0.5)
    >>> target.shape
    (4, 4)
    """
    target_T = robot_init_T.copy()
    controller_delta_pos = controller_now_T[:3, 3] - controller_init_T[:3, 3]
    controller_delta_R = controller_init_T[:3, :3].T @ controller_now_T[:3, :3]
    target_T[:3, 3] = robot_init_T[:3, 3] + controller_to_robot_position_R @ controller_delta_pos * translation_scale
    target_T[:3, :3] = (
        robot_init_T[:3, :3]
        @ controller_to_robot_orientation_R
        @ controller_delta_R
        @ controller_to_robot_orientation_R.T
    )
    return target_T


def create_xarm_target(
    arm_type: str,
    ip: str,
    mode_str: str,
    visualize: bool,
    root_node_name: str,
    root_position: tuple[float, float, float] | None,
    server: Any,
) -> Any:
    """Create an xArm hardware target or Viser URDF target."""
    arm_cls, arm_config_cls, ee_link, _, urdf_path = ARM_SPECS[arm_type]
    if visualize:
        return ViserArm(server, urdf_path, ee_link, root_node_name, root_position)
    mode = XArmMode.VELOCITY if mode_str == "velocity" else XArmMode.SERVO
    return arm_cls(arm_config_cls(ip=ip, mode=mode))


def update_frame(frame: Any, T: np.ndarray) -> None:
    assert frame is not None
    frame.position = T[:3, 3]
    frame.wxyz = matrix_to_quaternion(T[:3, :3])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quest3 Isaac Teleop -> xArm direct teleoperation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=("single", "bimanual"), default="single", help="Controller/arm mode")
    parser.add_argument("--single-side", choices=(LEFT, RIGHT), default=RIGHT, help="Controller side for single mode")
    parser.add_argument("--arm", type=str, default="xarm7", choices=tuple(ARM_SPECS), help="Arm type")
    parser.add_argument("--arm-ip", type=str, default=None, help="Single/right xArm IP")
    parser.add_argument("--right-xarm-ip", type=str, default=None, help="Right xArm IP")
    parser.add_argument("--left-xarm-ip", type=str, default=None, help="Left xArm IP for bimanual mode")
    parser.add_argument("--arm-mode", choices=("velocity", "servo"), default="velocity", help="xArm control mode")
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=0.5,
        help="Scale for controller position displacement",
    )
    parser.add_argument("--rate", type=float, default=100, help="Control loop rate in Hz")
    parser.add_argument("--home-settle-seconds", type=float, default=2.0, help="Wait time after homing")
    parser.add_argument("--app-name", type=str, default="RetargetingKitQuest3XArm", help="OpenXR app name")
    parser.add_argument(
        "--calibrate-button",
        choices=("A", "B"),
        default="B",
        help="Quest A/B button to calibrate (X/Y on the left controller)",
    )
    parser.add_argument(
        "--pause-button",
        choices=("A", "B"),
        default="B",
        help="Quest A/B button to pause or resume (X/Y on the left controller)",
    )
    parser.add_argument(
        "--reset-button",
        choices=("A", "B"),
        default="A",
        help="Quest A/B button to return to the calibrated pose (X/Y on the left controller)",
    )
    parser.add_argument("--visualize", action="store_true", help="Preview teleop in Viser without xArm hardware")
    parser.add_argument("--viser-port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--bimanual-separation", type=float, default=0.70, help="Visualizer root separation in meters")
    args = parser.parse_args()

    translation_scale = args.translation_scale
    rate_hz = args.rate
    home_settle_seconds = args.home_settle_seconds
    loop_dt = 1.0 / rate_hz
    home_qpos = np.asarray(ARM_SPECS[args.arm][3], dtype=np.float32)
    sides = (args.single_side,) if args.mode == "single" else (LEFT, RIGHT)

    server = None
    if args.visualize:
        import viser

        server = viser.ViserServer(port=args.viser_port)
        server.scene.add_grid("/grid", width=2.0, height=2.0)

    runtimes: dict[str, ArmRuntime] = {}
    for side in sides:
        ip = args.left_xarm_ip if side == LEFT else args.right_xarm_ip
        if args.mode == "single":
            ip = ip or args.arm_ip
        elif side == RIGHT:
            ip = ip or args.arm_ip
        ip = ip or DEFAULT_XARM_IP[side]
        if not args.visualize:
            assert ip is not None, f"Missing xArm IP for {side}"
        root_position = None
        root_node_name = "/robot" if args.mode == "single" else f"/{side}_robot"
        if args.visualize and args.mode == "bimanual":
            y = args.bimanual_separation * (0.5 if side == LEFT else -0.5)
            root_position = (0.0, y, 0.0)
        target = create_xarm_target(
            args.arm,
            ip,
            args.arm_mode,
            args.visualize,
            root_node_name,
            root_position,
            server,
        )
        if not args.visualize:
            target.start()
        target.set_qpos(home_qpos)
        current_frame = controller_frame = target_frame = None
        if args.visualize:
            assert server is not None
            prefix = f"/{side}" if args.mode == "bimanual" else "/robot"
            current_frame = server.scene.add_frame(f"{prefix}/current", axes_length=0.07, axes_radius=0.0035)
            controller_frame = server.scene.add_frame(f"{prefix}/controller", axes_length=0.07, axes_radius=0.0035)
            target_frame = server.scene.add_frame(f"{prefix}/target", axes_length=0.08, axes_radius=0.004)
        else:
            time.sleep(home_settle_seconds)
        runtimes[side] = ArmRuntime(
            side,
            target,
            current_frame,
            controller_frame,
            target_frame,
            CONTROLLER_TO_XARM_POSITION_R,
            CONTROLLER_TO_XARM_ORIENTATION_R[side],
        )

    quest = Quest3Config(app_name=args.app_name, sides=sides).build()

    def _stop(_signum: int, _frame: object) -> None:
        quest.close()
        for runtime in runtimes.values():
            if not args.visualize:
                runtime.target.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)

    if args.visualize:
        print(f"Viser ready - open http://localhost:{args.viser_port}")
    print("Waiting for valid Quest3 controller tracking.")
    print("In the Quest browser, click Connect and Play, then keep the Touch controller(s) awake.")
    while True:
        frame = quest.get_frame()
        invalid_sides = [side for side in sides if side not in frame.controllers]
        if len(invalid_sides) == 0:
            break
        print(f"Waiting for valid controller pose: {', '.join(invalid_sides)}")
        time.sleep(1.0)
    print(f"Align Quest3 controller(s) with the xArm end-effector(s), then press {args.calibrate_button} to calibrate.")

    last_tracking_warning_time = 0.0
    while True:
        frame = quest.get_frame()
        invalid_sides = [side for side in sides if side not in frame.controllers]
        if len(invalid_sides) > 0:
            now = time.time()
            if now - last_tracking_warning_time >= 1.0:
                print(f"Skipping target update until controller tracking is valid: {', '.join(invalid_sides)}")
                last_tracking_warning_time = now
            time.sleep(loop_dt)
            continue
        for side, runtime in runtimes.items():
            controller = frame.controllers[side]
            buttons = {"A": controller.primary_click, "B": controller.secondary_click}
            current_robot_T = pose7_to_tf_mat(runtime.target.get_ee_pose())
            calibrate_down = buttons[args.calibrate_button]
            if runtime.controller_init_T is None:
                if calibrate_down and not runtime.calibrate_prev:
                    runtime.controller_init_T = controller.pose
                    runtime.robot_init_T = current_robot_T
                    runtime.stream_enabled = True
                    runtime.pause_prev = buttons[args.pause_button]
                    runtime.reset_prev = buttons[args.reset_button]
                    if args.visualize:
                        update_frame(runtime.current_frame, current_robot_T)
                        update_frame(runtime.controller_frame, current_robot_T)
                        update_frame(runtime.target_frame, current_robot_T)
                    print(f"Calibrated {side} controller. Teleop running.")
                runtime.calibrate_prev = calibrate_down
                continue
            reset_down = buttons[args.reset_button]
            if reset_down and not runtime.reset_prev:
                assert runtime.robot_init_T is not None
                runtime.controller_init_T = controller.pose
                runtime.target.set_eepose(tf_mat_to_pose7(runtime.robot_init_T))
                print(f"{side} arm returned to the calibrated pose.")
            runtime.reset_prev = reset_down
            pause_down = buttons[args.pause_button]
            if pause_down and not runtime.pause_prev:
                runtime.stream_enabled = not runtime.stream_enabled
                runtime.controller_init_T = controller.pose
                runtime.robot_init_T = current_robot_T
                status = "resumed" if runtime.stream_enabled else "paused"
                print(f"{side} arm teleop {status}.")
            runtime.pause_prev = pause_down
            if not runtime.stream_enabled:
                continue
            assert runtime.controller_init_T is not None
            assert runtime.robot_init_T is not None
            target_T = compute_xarm_target_T(
                runtime.controller_init_T,
                controller.pose,
                runtime.robot_init_T,
                runtime.controller_to_robot_position_R,
                runtime.controller_to_robot_orientation_R,
                translation_scale,
            )
            if args.visualize:
                update_frame(runtime.current_frame, current_robot_T)
                update_frame(runtime.controller_frame, target_T)
                update_frame(runtime.target_frame, target_T)
            runtime.target.set_eepose(tf_mat_to_pose7(target_T))
        time.sleep(loop_dt)


if __name__ == "__main__":
    main()
