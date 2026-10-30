"""Named body-source to humanoid-target teleoperation rigs.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run scripts/teleop/humanoid.py bvh_g1_sonic

    CUDA_VISIBLE_DEVICES=0 uv run --extra vis scripts/teleop/humanoid.py \
        bvh_g1_sonic_viser
"""

import logging
import signal
import sys
import time
from contextlib import ExitStack
from types import FrameType

import tyro

from retargetingkit.configs.teleop.config import HumanoidTeleopConfig
from retargetingkit.configs.teleop.humanoid import _HUMANOID_TELEOP_CONFIGS
from retargetingkit.targets.target import Target


logger = logging.getLogger(__name__)
RATE_HZ = 50.0


def run(config: HumanoidTeleopConfig) -> None:
    """Run the source-agnostic humanoid teleop loop."""
    logger.info("Loading robot URDF: %s", config.retargeter.preset.urdf_path)
    retargeter = config.retargeter.build(config.device)
    running = True

    def request_stop(_signum: int, _frame: FrameType | None) -> None:
        nonlocal running
        running = False

    with ExitStack() as stack:
        previous_sigint = signal.signal(signal.SIGINT, request_stop)
        stack.callback(signal.signal, signal.SIGINT, previous_sigint)
        source = config.source.build()
        stack.callback(source.close)
        targets: list[Target] = []
        for target_config in config.targets:
            target = target_config.build()
            stack.callback(target.close)
            targets.append(target)

        logger.info(
            "%s -> %s -> %s",
            type(config.source).__name__,
            type(config.retargeter).__name__,
            [type(target).__name__ for target in targets],
        )
        time.sleep(0.5)
        period = 1.0 / RATE_HZ
        logger.info("Humanoid teleop running at %.0f Hz. Press Ctrl+C to stop.", RATE_HZ)
        while running:
            started = time.monotonic()
            frame = source.get_frame()
            if frame is not None:
                command = retargeter.solve(frame)
                for target in targets:
                    target.set_command(command)
            time.sleep(max(0.0, period - (time.monotonic() - started)))


def main() -> None:
    """Resolve `<rig> [overrides]` from argv, then run the selected preset."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        print(
            "Usage: uv run scripts/teleop/humanoid.py <rig> [OPTIONS]\n\nRigs:\n  "
            + "\n  ".join(_HUMANOID_TELEOP_CONFIGS)
        )
        return
    config = tyro.cli(
        HumanoidTeleopConfig,
        default=_HUMANOID_TELEOP_CONFIGS[sys.argv[1]],
        args=sys.argv[2:],
    )
    run(config)


if __name__ == "__main__":
    main()
