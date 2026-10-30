"""Neural hand gallery: switch robot checkpoints while replaying MANO-21 keypoints.

Run with ``uv run python -m retargetingkit.gallery.hand_neural``.
"""

import argparse
import time

import viser

from retargetingkit.assets import calib, neural_hand
from retargetingkit.retargeting.hand.neural import NeuralHandAssetRetargeterConfig
from retargetingkit.retargeting.hand.presets import HANDS
from retargetingkit.sources.glove.manus_conversion import load_mano_keypoints_from_manus_npz
from retargetingkit.utils.viz import add_hand_urdf
from robokit.utils.visualize_utils import ViserHandSkeleton


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=tuple(HANDS), default="sharpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    checkpoints_by_robot = {
        robot: tuple(name for name, hand in neural_hand.CHECKPOINTS.items() if hand == robot) for robot in HANDS
    }
    keypoints = load_mano_keypoints_from_manus_npz(calib.get("manus_motion"))

    if args.smoke_test:
        checkpoint = checkpoints_by_robot[args.robot][0]
        spec = HANDS[args.robot]
        retargeter = NeuralHandAssetRetargeterConfig(checkpoint, spec).build(args.device)
        qpos = retargeter.retarget(keypoints[0])
        print(f"neural gallery smoke test: robot={args.robot} checkpoint={checkpoint} q={qpos.shape}")
        return

    server = viser.ViserServer()
    server.scene.add_grid("/grid", 1.0, 1.0)
    human = ViserHandSkeleton(server, "/human", position=(0.0, -0.1, 0.0))
    server.scene.add_frame("/robot", show_axes=False, position=(0.0, 0.1, 0.0))
    with server.gui.add_folder("Model"):
        robot_dd = server.gui.add_dropdown("Robot", tuple(HANDS), initial_value=args.robot)
        checkpoint_dd = server.gui.add_dropdown(
            "Checkpoint", checkpoints_by_robot[args.robot], initial_value=checkpoints_by_robot[args.robot][0]
        )
    frame_slider = server.gui.add_slider("Frame", 0, len(keypoints) - 1, 1, 0)
    playing = server.gui.add_checkbox("Play", True)
    status = server.gui.add_text("Status", initial_value="Loading...", disabled=True)

    robot = None
    joint_indices: tuple[int, ...] = ()
    retargeter = None
    built = ("", "")
    while True:
        options = checkpoints_by_robot[robot_dd.value]
        if checkpoint_dd.options != options:
            checkpoint_dd.options = options
        if checkpoint_dd.value not in options:
            checkpoint_dd.value = options[0]
        selection = (robot_dd.value, checkpoint_dd.value)
        if selection != built:
            spec = HANDS[selection[0]]
            retargeter = NeuralHandAssetRetargeterConfig(selection[1], spec).build(args.device)
            if robot is not None:
                robot.remove()
            robot, joint_indices = add_hand_urdf(server, spec, "/robot")
            status.value = f"{selection[0]} / {selection[1]}"
            built = selection

        if playing.value:
            frame_slider.value = (frame_slider.value + 1) % len(keypoints)
        index = frame_slider.value
        assert retargeter is not None and robot is not None
        qpos = retargeter.retarget(keypoints[index])
        with server.atomic():
            human.update(keypoints[index])
            robot.update_cfg(qpos[list(joint_indices)])
        time.sleep(0.03)


if __name__ == "__main__":
    main()
