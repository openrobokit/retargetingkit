"""Named arm-hand-source to robot arm-and-hand teleoperation rigs.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run --extra vis scripts/teleop/arm_hand.py \
        visionpro_xarm7_xhand

    CUDA_VISIBLE_DEVICES=0 uv run --extra vis scripts/teleop/arm_hand.py \
        visionpro_xarm7_xhand --source.recording-path <avp_frames.pkl>
"""

import logging
import signal
import sys
import time
from contextlib import ExitStack
from types import FrameType
from typing import cast

import numpy as np
import tyro
import warp as wp

from retargetingkit.configs.teleop.arm_hand import _ARM_HAND_TELEOP_CONFIGS
from retargetingkit.configs.teleop.config import ArmHandTeleopConfig
from retargetingkit.retargeting.hand.optimization import OptimizationHandRetargeterConfig
from retargetingkit.sources.frames import Side
from retargetingkit.targets.target import Target
from retargetingkit.targets.viser import Viser, assembly_qpos_indices


logger = logging.getLogger(__name__)
RATE_HZ = 50.0
ARM_SETTLE_SECONDS = 2.5


def run(config: ArmHandTeleopConfig) -> None:
    """Run the source-agnostic arm-and-hand teleop loop."""
    wp.init()
    sides = tuple(config.arms)
    assert config.arms.keys() == config.hands.keys() == config.arm_retargeters.keys() == config.hand_retargeters.keys()
    for side in sides:
        assert config.arm_retargeters[side].side == side
        assert config.hand_retargeters[side].side == side
    arm_retargeters = {side: config.arm_retargeters[side].build(config.device) for side in sides}
    hand_retargeters = {side: config.hand_retargeters[side].build(config.device) for side in sides}
    source = config.source.build()
    use_viser = config.mode in ("viser", "both")
    use_hardware = config.mode in ("real", "both")
    visualizer: Viser | None = None
    arm_targets: dict[Side, Target] = {}
    hand_targets: dict[Side, Target] = {}
    visual_orders: dict[Side, np.ndarray] = {}
    running = True

    def request_stop(_signum: int, _frame: FrameType | None) -> None:
        nonlocal running
        running = False

    with ExitStack() as stack:
        stack.callback(source.close)
        previous_sigint = signal.signal(signal.SIGINT, request_stop)
        stack.callback(signal.signal, signal.SIGINT, previous_sigint)

        if use_viser:
            visualizer = Viser((config.assembly_urdf,) * len(sides))
            stack.callback(visualizer.close)
            logger.info("Viser: http://localhost:%d", visualizer.port)
            for side in sides:
                hand_config = cast(OptimizationHandRetargeterConfig, config.hand_retargeters[side])
                visual_orders[side] = assembly_qpos_indices(
                    config.assembly_urdf,
                    (config.arm_retargeters[side].urdf, hand_config.urdf),
                )

        if use_hardware:
            for side in sides:
                arm = config.arms[side].build()
                stack.callback(arm.close)
                hand = config.hands[side].build()
                stack.callback(hand.close)
                arm_targets[side] = arm
                hand_targets[side] = hand
            for side, target in arm_targets.items():
                target.set_qpos(arm_retargeters[side].home_qpos)
            logger.info("Moving robot arms to their home poses.")
            time.sleep(ARM_SETTLE_SECONDS)

        logger.info("%s started; the first wrist frame defines the home pose.", type(config.source).__name__)
        period = 1.0 / RATE_HZ
        logger.info("Arm-hand teleop running at %.0f Hz. Press Ctrl+C to stop.", RATE_HZ)
        while running:
            started = time.monotonic()
            frame = source.get_frame()
            if frame is not None:
                for side in sides:
                    if side in frame.arm.wrists and side in frame.hand.hands:
                        arm_command = arm_retargeters[side].solve(frame.arm)
                        hand_command = hand_retargeters[side].solve(frame.hand)
                        if side in arm_targets:
                            arm_targets[side].set_qpos(arm_command.qpos)
                            hand_targets[side].set_qpos(hand_command.qpos)
                        if visualizer is not None:
                            qpos = np.concatenate([arm_command.qpos, hand_command.qpos])
                            visualizer.set_qpos(qpos[visual_orders[side]], sides.index(side))
            time.sleep(max(0.0, period - (time.monotonic() - started)))


def main() -> None:
    """Resolve `<rig> [overrides]` from argv, then run the selected preset."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        print(
            "Usage: uv run scripts/teleop/arm_hand.py <rig> [OPTIONS]\n\nRigs:\n  "
            + "\n  ".join(_ARM_HAND_TELEOP_CONFIGS)
        )
        return
    config = tyro.cli(
        ArmHandTeleopConfig,
        default=_ARM_HAND_TELEOP_CONFIGS[sys.argv[1]],
        args=sys.argv[2:],
    )
    run(config)


if __name__ == "__main__":
    main()
