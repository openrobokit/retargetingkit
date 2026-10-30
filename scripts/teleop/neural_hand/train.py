"""Train one canonical neural hand-retargeting config.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run scripts/teleop/neural_hand/train.py \
        scripts/teleop/neural_hand/configs/training/sharpa_motion_alignment.yaml
"""

import argparse
from pathlib import Path

from retargetingkit.training.hand.train import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train neural hand retargeting.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train_from_config(args.config, args.device)


if __name__ == "__main__":
    main()
