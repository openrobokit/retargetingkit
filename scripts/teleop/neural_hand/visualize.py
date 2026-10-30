"""Visualize neural hand retargeting on a Manus recording.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run --extra vis scripts/teleop/neural_hand/visualize.py sharpa \
        --checkpoints ik_model/sharpa_motion_alignment ik_model/sharpa_motion_alignment_contact

    CUDA_VISIBLE_DEVICES=0 uv run --extra vis scripts/teleop/neural_hand/visualize.py xhand \
        --checkpoints ik_model/xhand_motion_alignment --data manus_motion

    CUDA_VISIBLE_DEVICES=0 uv run --extra vis scripts/teleop/neural_hand/visualize.py allegro \
            --checkpoints ik_model/allegro_motion_alignment --data manus_motion
"""

import argparse
import time
from pathlib import Path

import numpy as np
import viser

from retargetingkit.assets import calib
from retargetingkit.retargeting.hand.neural import NeuralHandRetargeterConfig
from retargetingkit.retargeting.hand.presets import HANDS
from retargetingkit.sources.glove.manus_conversion import (
    manus_positions_to_fixed_wrist_local,
    manus_to_mano_keypoints,
)
from retargetingkit.utils.viz import add_hand_urdf, draw_manus_skeleton


HUMAN_OFFSET = (0.0, 0.3, 0.0)
COLUMN_SPACING = 0.3


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize neural hand retargeting on a Manus recording.")
    parser.add_argument("hand", choices=tuple(HANDS))
    parser.add_argument("--checkpoints", nargs="+", type=Path, help="one robot column per checkpoint")
    parser.add_argument("--paired", action="store_true", help="replay measured qpos from a paired capture")
    parser.add_argument("--data", default="manus_motion", help="calib asset name or .npz path")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    assert (args.checkpoints is None) == args.paired

    spec = HANDS[args.hand]
    data = np.load(Path(args.data) if args.data.endswith(".npz") else calib.get(args.data))
    positions = data["right_positions"].astype(np.float32)
    rotations = data["right_rotations"].astype(np.float32)
    human = np.stack([manus_positions_to_fixed_wrist_local(p, r) for p, r in zip(positions, rotations, strict=True)])
    if args.paired:
        qpos = data["qpos"].astype(np.float32)
        assert qpos.shape == (len(positions), spec.joint_count)
        columns = [("measured", qpos)]
    else:
        mano = np.stack([manus_to_mano_keypoints(p, r) for p, r in zip(positions, rotations, strict=True)])
        columns = []
        for checkpoint in args.checkpoints:
            retargeter = NeuralHandRetargeterConfig(checkpoint, spec).build(args.device)
            columns.append((checkpoint.name, retargeter.retarget_batch(mano)))
    print(
        f"hand={args.hand} data={args.data} frames={len(positions)} columns={[name for name, _ in columns]}",
        flush=True,
    )

    server = viser.ViserServer()
    server.scene.add_grid("/grid", 1.0, 1.0)
    server.scene.add_frame("/human", position=HUMAN_OFFSET, show_axes=False)
    hands = []
    for index, (name, qpos) in enumerate(columns):
        server.scene.add_frame(f"/hand_{index}", position=(0.0, -COLUMN_SPACING * index, 0.0), show_axes=False)
        server.scene.add_label(f"/hand_{index}/label", name, position=(0.0, 0.0, 0.25))
        urdf, joint_indices = add_hand_urdf(server, spec, f"/hand_{index}/robot")
        hands.append((urdf, joint_indices, qpos))

    play = server.gui.add_checkbox("Play", initial_value=True)
    fps = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=args.fps)
    slider = server.gui.add_slider("Frame", min=0, max=len(positions) - 1, step=1, initial_value=0)

    def update(frame: int) -> None:
        with server.atomic():
            draw_manus_skeleton(server, human[frame], "/human")
            for urdf, joint_indices, qpos in hands:
                urdf.update_cfg(np.asarray([qpos[frame, index] for index in joint_indices], dtype=np.float32))

    slider.on_update(lambda _: update(int(slider.value)))
    update(0)
    print(f"viser={server.get_port()}", flush=True)
    while True:
        time.sleep(1.0 / fps.value)
        if play.value:
            slider.value = (int(slider.value) + 1) % len(positions)


if __name__ == "__main__":
    main()
