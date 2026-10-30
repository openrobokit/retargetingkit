"""Batch neural hand retargeting with Allegro.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python -m retargetingkit.examples.hand_neural.02_batch
"""

import argparse
import time

import numpy as np
import viser

from retargetingkit.assets import calib, neural_hand
from retargetingkit.retargeting.hand.neural import NeuralHandAssetRetargeterConfig
from retargetingkit.retargeting.hand.presets import HANDS
from retargetingkit.utils.viz import add_hand_urdf
from robokit.utils.visualize_utils import ViserHandSkeleton


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        choices=tuple(name for name, hand in neural_hand.CHECKPOINTS.items() if hand == "allegro"),
        default="allegro_motion_alignment",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    spec = HANDS["allegro"]
    retargeter = NeuralHandAssetRetargeterConfig(args.checkpoint, spec).build(args.device)
    keypoints = np.load(calib.get("mano_alex")).astype(np.float32)
    offsets = np.linspace(0, len(keypoints), args.batch_size, endpoint=False, dtype=int)

    if args.smoke_test:
        qpos = retargeter.retarget_batch(keypoints[offsets])
        print(f"batch smoke test: q={qpos.shape}")
        return

    server = viser.ViserServer()
    server.scene.add_grid("/grid", 1.5, 1.0)
    humans = []
    robots = []
    for index in range(args.batch_size):
        x = 0.3 * index
        humans.append(ViserHandSkeleton(server, f"/human_{index}", position=(x, -0.1, 0.0)))
        server.scene.add_frame(f"/robot_{index}", show_axes=False, position=(x, 0.1, 0.0))
        robots.append(add_hand_urdf(server, spec, f"/robot_{index}"))

    frame = 0
    while True:
        indices = (offsets + frame) % len(keypoints)
        batch = keypoints[indices]
        qpos = retargeter.retarget_batch(batch)
        with server.atomic():
            for index, (robot, joint_indices) in enumerate(robots):
                humans[index].update(batch[index])
                robot.update_cfg(qpos[index, list(joint_indices)])
        frame += 1
        time.sleep(0.03)


if __name__ == "__main__":
    main()
