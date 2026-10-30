# Hand Retargeting

| Hand motion on a robot assembly. | Hand-object contact refinement. |
| --- | --- |

Map human hand motion to robot joint positions with RoboKit's online or offline optimization. Start with a recorded example, then connect a [Manus glove](../reference/sources/manus.md) or [Apple Vision Pro](../reference/sources/vision_pro.md).

## How Retargeting Works

A source supplies hand keypoints. The optimization retargeter matches them to a robot hand while respecting its joints and, when configured, object contact. Online solving handles one frame at a time; offline solving optimizes a trajectory. Both produce joint positions in the robot's URDF order.

## Try a recording

From the RetargetingKit checkout, run:

```shell
uv run --extra vis python -m retargetingkit.examples.hand_optimization.01_online
```

Open the Viser address printed in the terminal. The example replays hand motion and shows the retargeted robot without connecting hardware. Required assets download on first use. Press `Ctrl+C` to stop.

## Use the Python API

Build a rig's retargeter and pass it one `HandFrame` per update:

```python
# docs: hide
import pickle

import numpy as np
import warp as wp

from retargetingkit.assets.example_data import MANUS_DATA_RIGHT_PKL
from retargetingkit.sources.frames import HandFrame, HandPose

device = "cuda:0" if wp.is_cuda_available() else "cpu"
# Three recorded wrist-local MANO-21 frames from a right Manus glove.
keypoints = pickle.load(open(MANUS_DATA_RIGHT_PKL, "rb"))["keypoints"][:3]
stream = [HandFrame({"right": HandPose(points)}) for points in keypoints]
# docs: endhide
from retargetingkit.configs.teleop.hand.manus_sharpa import RIGHT_MANUS_SHARPA_CONFIG

retargeter = RIGHT_MANUS_SHARPA_CONFIG.retargeter.build(device)

for frame in stream:  # HandFrame: right-hand MANO-21 keypoints
    command = retargeter.solve(frame)
    qpos = command.qpos  # (22,), Sharpa joints in URDF order
# docs: hide
assert qpos.shape == (22,)
assert np.isfinite(qpos).all()
# docs: endhide
```

## Preview live motion

Complete the source setup first. For a right Manus glove and XHand:

```shell
uv run --extra manus --extra vis scripts/teleop/hand.py \
    manus_xhand --mode viser
```

The live runtime defaults to `cuda:0`. Select an available GPU with `CUDA_VISIBLE_DEVICES` before launching, or pass `--device cpu` for optimization on CPU.

| Optimization rig | Source | Robot hand |
| --- | --- | --- |
| `manus_xhand` | Manus | XHand |
| `manus_sharpa` | Manus | Sharpa Wave |
| `visionpro_xhand` | Vision Pro | XHand |
| `visionpro_sharpa` | Vision Pro | Sharpa Wave |

Inspect a preset's device settings with `uv run scripts/teleop/hand.py manus_xhand --help`. Vision Pro requires the `vision-pro` extra.

## Connect the robot

After checking motion in the viewer, configure the matching [XHand](../reference/targets/xhand.md) or [Sharpa](../reference/targets/sharpa_hand.md) target.

| Mode | Behavior |
| --- | --- |
| `viser` | Preview only; the robot target is not opened. This is the default. |
| `real` | Connect and command the physical hand. |
| `both` | Command the physical hand and show Viser. |

For example, `manus_xhand --mode both --target.serial-port /dev/ttyUSB0` enables XHand hardware alongside the preview.

## Work with trajectories

Use the packaged examples to inspect solver construction and array shapes:

| Module under `retargetingkit.examples.hand_optimization` | Task |
| --- | --- |
| `01_online` | Retarget one stream frame by frame |
| `02_offline` | Optimize one recorded trajectory |
| `03_batch` | Optimize several trajectories together |
| `04_arm` | Retarget an arm and hand together |
| `05_online_batch` | Retarget several arm-hand streams online |

Run any module with `uv run --extra vis python -m <MODULE>`. Robot outputs use URDF joint order. Online inputs have shape `(K, 3)` or `(B, K, 3)`, with points ordered by `spec.target_names`; offline inputs add a time axis.

## Inspect object contact

Browse DexYCB recordings with an xArm7–XHand or xArm7–Ability assembly and compare contact refinement in the viewer:

```shell
uv run --extra vis python -m retargetingkit.gallery.hand_contact
```

For learned inference, including the supported live `manus_sharpa_neural` rig, continue with [neural hand retargeting](neural_hand_retargeting.md).
