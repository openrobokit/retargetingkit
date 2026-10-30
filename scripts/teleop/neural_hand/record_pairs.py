"""Neural Manus-to-Sharpa teleop with paired pose capture.

Keys: `t` toggles teach mode, `c` captures, and `q` saves and quits.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run --extra manus scripts/teleop/neural_hand/record_pairs.py \
        --checkpoint ik_model/sharpa_motion_alignment --output capture.npz
"""

import argparse
import time
from pathlib import Path

import numpy as np

from retargetingkit.retargeting.hand.neural import NeuralHandRetargeterConfig
from retargetingkit.retargeting.hand.presets import HANDS
from retargetingkit.sources.glove._manus_sdk import ManusClient, resolve_manus_sdk_library_path
from retargetingkit.sources.glove.manus_conversion import manus_to_mano_keypoints
from retargetingkit.sources.glove.manus_glove import (
    DualManusGloveAggregator,
    parse_glove_id,
)
from retargetingkit.sources.misc.keyboard import Keyboard
from retargetingkit.targets.hands.sharpa_hand import JOINT_NAMES, SharpaHand


def main() -> None:
    parser = argparse.ArgumentParser(description="Record paired Manus and Sharpa poses")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--manus-right-id")
    parser.add_argument("--device-sn")
    parser.add_argument("--speed-coeff", type=float, default=0.4)
    parser.add_argument("--current-coeff", type=float, default=0.5)
    args = parser.parse_args()

    spec = HANDS["sharpa"]
    assert spec.joint_names == JOINT_NAMES
    retargeter = NeuralHandRetargeterConfig(args.checkpoint, spec).build(args.device)
    client = ManusClient(resolve_manus_sdk_library_path())
    aggregator = DualManusGloveAggregator(right_override=parse_glove_id(args.manus_right_id))
    client.start(aggregator.update)
    print("Waiting for right Manus glove...")
    while aggregator.latest()[1] is None:
        time.sleep(0.1)

    hand = SharpaHand(
        speed_coeff=args.speed_coeff,
        current_coeff=args.current_coeff,
        device_sn=args.device_sn,
    )
    keyboard = Keyboard()
    keyboard.start()
    timestamps: list[float] = []
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    qpos: list[np.ndarray] = []
    modes: list[str] = []
    neural_mode = True
    previous_keys: set[str] = set()
    start_time = time.monotonic()
    period = 1.0 / args.rate
    print("NEURAL mode. Keys: t=toggle teach, c=capture, q=save and quit")

    while True:
        loop_start = time.monotonic()
        frame = aggregator.latest()[1]
        assert frame is not None
        mano = manus_to_mano_keypoints(frame.positions, frame.rotations)
        keys = set(keyboard.get_keys())
        pressed = keys - previous_keys
        previous_keys = keys
        if "t" in pressed:
            neural_mode = not neural_mode
            hand.set_control_mode("position" if neural_mode else "floating")
            print("NEURAL mode" if neural_mode else "TEACH mode")
        if neural_mode:
            hand.set_qpos(retargeter.retarget(mano))
        if "c" in pressed:
            timestamps.append(time.monotonic() - start_time)
            positions.append(frame.positions.copy())
            rotations.append(frame.rotations.copy())
            qpos.append(hand.get_qpos())
            modes.append("neural" if neural_mode else "teach")
            print(f"captured={len(qpos)} mode={modes[-1]}")
        if "q" in pressed:
            break
        time.sleep(max(0.0, period - (time.monotonic() - loop_start)))

    keyboard.stop()
    hand.close()
    client.stop()
    assert qpos
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        right_timestamps=np.asarray(timestamps, dtype=np.float64),
        right_positions=np.stack(positions).astype(np.float32),
        right_rotations=np.stack(rotations).astype(np.float32),
        qpos=np.stack(qpos).astype(np.float32),
        modes=np.asarray(modes),
    )
    print(f"saved={args.output} poses={len(qpos)}")


if __name__ == "__main__":
    main()
