"""Generate YCB Sharpa grasp specs for later human capture.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run --extra contact scripts/teleop/neural_hand/contact_generate.py \
        scripts/teleop/neural_hand/configs/contact/sharpa_ycb_contact.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from retargetingkit.retargeting.hand.presets import HANDS
from retargetingkit.training.hand.contact import FINGER_NAMES, HandContactModel, generate_grasps
from retargetingkit.training.hand.kinematics import load_hand_robot


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate YCB Sharpa grasp specs.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    print(f"config={args.config} device={device} objects={len(config['objects'])}", flush=True)
    spec = HANDS["sharpa"]
    robot = load_hand_robot(spec)
    joint_limits = torch.from_numpy(robot.spec.actuated_joint_limits).float().to(device)
    pad_link_names = tuple(f"right_{finger}_elastomer" for finger in FINGER_NAMES)
    contact_model = HandContactModel.build(
        robot,
        joint_limits,
        pad_link_names,
        spec.collision_max_active_pairs,
    )
    data = generate_grasps(config, contact_model, spec.human_keypoint_indices, device)
    output = Path(config["output"])
    data.save(output)
    print(f"output={output} rows={len(data)}", flush=True)


if __name__ == "__main__":
    main()
