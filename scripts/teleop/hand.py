"""Hand teleoperation: a named rig preset streamed through RoboKit online optimization.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run --extra manus --extra vis scripts/teleop/hand.py \
        manus_xhand --mode both

    uv run --extra vis scripts/teleop/hand.py visionpro_xhand --source.recording-path <avp_frames.pkl>
"""

import logging
import signal
import sys
import time
from contextlib import ExitStack
from types import FrameType

import tyro
import warp as wp

from retargetingkit.configs.teleop.config import HandTeleopConfig
from retargetingkit.configs.teleop.hand import _HAND_TELEOP_CONFIGS
from retargetingkit.retargeting.hand.neural import NeuralHandAssetRetargeterConfig, NeuralHandRetargeterConfig
from retargetingkit.retargeting.hand.optimization import OptimizationHandRetargeterConfig
from retargetingkit.targets.target import Target


logger = logging.getLogger(__name__)


def run(config: HandTeleopConfig) -> None:
    """Run the source-agnostic teleop loop: get_frame -> solve -> targets."""
    side = config.side
    assert isinstance(
        config.retargeter,
        (OptimizationHandRetargeterConfig, NeuralHandRetargeterConfig, NeuralHandAssetRetargeterConfig),
    )
    assert config.retargeter.side == side
    urdf_path = (
        config.retargeter.urdf
        if isinstance(config.retargeter, OptimizationHandRetargeterConfig)
        else config.retargeter.hand.urdf_path
    )
    wp.init()

    retargeter = config.retargeter.build(config.device)
    logger.info("Hand retargeting initialized for the %s hand.", side)

    source = config.source.build()
    logger.info("%s started; waiting for %s hand data.", type(config.source).__name__, side)

    use_viser = config.mode in ("viser", "both")
    use_hardware = config.mode in ("real", "both")
    visualizer = None
    target: Target | None = None
    running = True

    def request_stop(_signum: int, _frame: FrameType | None) -> None:
        nonlocal running
        running = False

    with ExitStack() as stack:
        stack.callback(source.close)
        previous_sigint = signal.signal(signal.SIGINT, request_stop)
        stack.callback(signal.signal, signal.SIGINT, previous_sigint)
        if use_viser:
            from retargetingkit.sources.frames import MANO21_EDGES
            from retargetingkit.targets.viser import Viser

            visualizer = Viser((urdf_path,), MANO21_EDGES)
            stack.callback(visualizer.close)
            logger.info("Viser: http://localhost:%d", visualizer.port)

        if use_hardware:
            target = config.target.build()
            stack.callback(target.close)
            logger.info("%s hardware connected.", type(config.target).__name__)

        period = 1.0 / config.rate
        logger.info("Hand teleop running at %g Hz. Press Ctrl+C to stop.", config.rate)
        while running:
            started = time.monotonic()
            frame = source.get_frame()
            if frame is not None and side in frame.hands:
                command = retargeter.solve(frame)
                if target is not None:
                    target.set_qpos(command.qpos)
                if visualizer is not None:
                    visualizer.set_qpos(command.qpos)
                    visualizer.set_points(frame.hands[side].positions)
            time.sleep(max(0.0, period - (time.monotonic() - started)))


def main() -> None:
    """Resolve `<rig> [overrides]` from argv, then run the selected preset."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        print("Usage: uv run scripts/teleop/hand.py <rig> [OPTIONS]\n\nRigs:\n  " + "\n  ".join(_HAND_TELEOP_CONFIGS))
        return
    config = tyro.cli(
        HandTeleopConfig,
        default=_HAND_TELEOP_CONFIGS[sys.argv[1]],
        args=sys.argv[2:],
    )
    run(config)


if __name__ == "__main__":
    main()
