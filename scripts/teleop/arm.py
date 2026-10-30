"""Named wrist-source to robot-arm teleoperation rigs.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run --extra vis scripts/teleop/arm.py visionpro_xarm7

    CUDA_VISIBLE_DEVICES=0 uv run --extra vis scripts/teleop/arm.py \
        visionpro_xarm7 --source.recording-path <avp_frames.pkl>
"""

import logging
import signal
import sys
import time
from contextlib import ExitStack
from types import FrameType

import tyro
import warp as wp

from retargetingkit.configs.teleop.arm import _ARM_TELEOP_CONFIGS
from retargetingkit.configs.teleop.config import ArmTeleopConfig
from retargetingkit.sources.frames import Side
from retargetingkit.targets.target import Target
from retargetingkit.targets.viser import Viser


logger = logging.getLogger(__name__)
RATE_HZ = 50.0
ARM_SETTLE_SECONDS = 2.5


def run(config: ArmTeleopConfig) -> None:
    """Run the source-agnostic arm teleop loop."""
    wp.init()
    sides = tuple(config.arms)
    assert config.arms.keys() == config.retargeters.keys()
    for side, retargeter in config.retargeters.items():
        assert retargeter.side == side
    retargeters = {side: retargeter.build(config.device) for side, retargeter in config.retargeters.items()}
    source = config.source.build()
    use_viser = config.mode in ("viser", "both")
    use_hardware = config.mode in ("real", "both")
    visualizer: Viser | None = None
    targets: dict[Side, Target] = {}
    running = True

    def request_stop(_signum: int, _frame: FrameType | None) -> None:
        nonlocal running
        running = False

    with ExitStack() as stack:
        stack.callback(source.close)
        previous_sigint = signal.signal(signal.SIGINT, request_stop)
        stack.callback(signal.signal, signal.SIGINT, previous_sigint)

        if use_viser:
            visualizer = Viser(tuple(retargeters[side].urdf_path for side in sides))
            stack.callback(visualizer.close)
            logger.info("Viser: http://localhost:%d", visualizer.port)

        if use_hardware:
            for side, arm in config.arms.items():
                target = arm.build()
                stack.callback(target.close)
                targets[side] = target
                target.set_qpos(retargeters[side].home_qpos)
            logger.info("Moving robot arms to their home poses.")
            time.sleep(ARM_SETTLE_SECONDS)

        logger.info("%s started; the first wrist frame defines the home pose.", type(config.source).__name__)
        period = 1.0 / RATE_HZ
        logger.info("Arm teleop running at %.0f Hz. Press Ctrl+C to stop.", RATE_HZ)
        while running:
            started = time.monotonic()
            frame = source.get_frame()
            if frame is not None:
                for side, retargeter in retargeters.items():
                    if side in frame.wrists:
                        command = retargeter.solve(frame)
                        if side in targets:
                            targets[side].set_qpos(command.qpos)
                        if visualizer is not None:
                            visualizer.set_qpos(command.qpos, sides.index(side))
            time.sleep(max(0.0, period - (time.monotonic() - started)))


def main() -> None:
    """Resolve `<rig> [overrides]` from argv, then run the selected preset."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        print("Usage: uv run scripts/teleop/arm.py <rig> [OPTIONS]\n\nRigs:\n  " + "\n  ".join(_ARM_TELEOP_CONFIGS))
        return
    config = tyro.cli(
        ArmTeleopConfig,
        default=_ARM_TELEOP_CONFIGS[sys.argv[1]],
        args=sys.argv[2:],
    )
    run(config)


if __name__ == "__main__":
    main()
