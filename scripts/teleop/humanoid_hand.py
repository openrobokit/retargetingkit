"""PICO body and buttons controlling G1 and dual Inspire hands in one process.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run --extra pico --extra inspire \
        scripts/teleop/humanoid_hand.py pico_g1_inspire
"""

import logging
import signal
import sys
import time
from contextlib import ExitStack
from types import FrameType

import numpy as np
import tyro

from retargetingkit.configs.teleop.config import HumanoidHandTeleopConfig
from retargetingkit.configs.teleop.humanoid_hand import _HUMANOID_HAND_TELEOP_CONFIGS
from retargetingkit.targets.target import Target


logger = logging.getLogger(__name__)
RATE_HZ = 50.0


def run(config: HumanoidHandTeleopConfig) -> None:
    """Run one direct PICO connection for body and binary hand control."""
    logger.info("Loading robot URDF: %s", config.retargeter.preset.urdf_path)
    retargeter = config.retargeter.build(config.device)
    open_qpos = np.asarray(config.open_qpos, dtype=np.float64)
    closed_qpos = np.asarray(config.closed_qpos, dtype=np.float64)
    assert open_qpos.shape == closed_qpos.shape == (6,)
    running = True
    previous_buttons = {"B": False, "Y": False}
    closed = {"left": False, "right": False}

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
        left_hand = config.left_hand.build()
        stack.callback(left_hand.close)
        right_hand = config.right_hand.build()
        stack.callback(right_hand.close)

        logger.info(
            "%s -> %s -> %s + [%s, %s]",
            type(config.source).__name__,
            type(config.retargeter).__name__,
            [type(target).__name__ for target in targets],
            type(left_hand).__name__,
            type(right_hand).__name__,
        )
        time.sleep(0.5)
        period = 1.0 / RATE_HZ
        logger.info("Humanoid-hand teleop running at %.0f Hz. B=right, Y=left. Press Ctrl+C to stop.", RATE_HZ)
        while running:
            started = time.monotonic()
            frame = source.get_frame()
            if frame is not None:
                command = retargeter.solve(frame)
                for target in targets:
                    target.set_command(command)

            buttons = source.get_buttons()
            if buttons:
                if buttons["B"] and not previous_buttons["B"]:
                    closed["right"] = not closed["right"]
                    right_hand.set_qpos(closed_qpos if closed["right"] else open_qpos)
                    logger.info("right hand -> %s", "CLOSED" if closed["right"] else "OPEN")
                if buttons["Y"] and not previous_buttons["Y"]:
                    closed["left"] = not closed["left"]
                    left_hand.set_qpos(closed_qpos if closed["left"] else open_qpos)
                    logger.info("left hand -> %s", "CLOSED" if closed["left"] else "OPEN")
                previous_buttons["B"], previous_buttons["Y"] = buttons["B"], buttons["Y"]
            time.sleep(max(0.0, period - (time.monotonic() - started)))


def main() -> None:
    """Resolve `<rig> [overrides]` from argv, then run the selected preset."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        print(
            "Usage: uv run scripts/teleop/humanoid_hand.py <rig> [OPTIONS]\n\nRigs:\n  "
            + "\n  ".join(_HUMANOID_HAND_TELEOP_CONFIGS)
        )
        return
    config = tyro.cli(
        HumanoidHandTeleopConfig,
        default=_HUMANOID_HAND_TELEOP_CONFIGS[sys.argv[1]],
        args=sys.argv[2:],
    )
    run(config)


if __name__ == "__main__":
    main()
