# Full-Body Retargeting

| Body, hand, and object interaction. | Body and hand motion in a furnished scene. |
| --- | --- |

Retarget body motion, articulated fingers, and object interaction together to a G1 with Inspire hands. RetargetingKit loads SAGA or ParaHome clips; RoboKit solves the robot motion and object poses.

## How Retargeting Works

Each clip supplies body motion, finger motion, and interaction geometry. The body solver tracks human poses on G1; hand solvers track the fingers; grasp and scene terms keep the combined motion aligned with objects.

## Retarget one object interaction

Run the SAGA example from the RetargetingKit checkout:

```shell
uv run --extra vis python -m retargetingkit.examples.full_body.01_hoi
```

The default clip contains a body, captured right-hand motion, and one object. After solving, open the printed Viser address to play or scrub the result. Press `Ctrl+C` to stop. The example chooses CUDA when available and otherwise uses CPU.

Pass `--npz /path/to/clip.npz` for a local SAGA FullGraspPose recording. The example still fetches its SAGA asset bundle, including object meshes.

## Retarget an indoor interaction

ParaHome adds bimanual interaction and a furnished scene. List the available intervals in a sequence:

```shell
uv run --extra vis python -m retargetingkit.examples.full_body.02_hsi \
    --sequence s1 --list
```

Select an interval from that list:

```shell
uv run --extra vis python -m retargetingkit.examples.full_body.02_hsi \
    --sequence s1 --index 7 --scene_clearance
```

`--scene_clearance` adjusts body placement to reduce penetration into furniture. Use `--no_scene` to omit the room, or `--data_root /path/to/ParaHome/data` to provide local ParaHome data.

## Browse more clips

```shell
uv run --extra vis python -m retargetingkit.gallery.full_body_hoi
uv run --extra vis python -m retargetingkit.gallery.full_body_hsi
```

Choose a subject or sequence, then a clip in the viewer. By default, both galleries use captured finger poses and contact refinement. Add `--grasp` to optimize and hold grasps over contact intervals. The ParaHome gallery also exposes scene clearance in its viewer.

Both galleries accept `--max_frames 60 --headless` to solve a shorter selection and exit without opening a viewer. Dataset and model assets download on first use; gallery startup can fetch more than the selected clip.

These workflows produce motion for inspection. For live G1 control, follow [humanoid teleoperation](humanoid_retargeting.md).
