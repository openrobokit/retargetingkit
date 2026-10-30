# Arm Retargeting

Map a tracked wrist to an xArm6 or xArm7 end effector. The arm solver uses RoboKit inverse kinematics with position, orientation, joint-limit, and smoothness terms.

## How Retargeting Works

The source reports wrist poses in an `ArmFrame`. Calibration establishes the robot's home pose; inverse kinematics then maps relative wrist motion to a joint-position command for each arm.

## Preview one arm

Complete [Vision Pro setup](../reference/sources/vision_pro.md), then run:

```shell
uv run --extra vision-pro scripts/teleop/arm.py \
    visionpro_xarm7 --mode viser --device cpu \
    --source.ip <VISION_PRO_IP>
```

Open the Viser URL printed in the terminal. `visionpro_xarm6` selects xArm6; `visionpro_dual_xarm7` controls two independent arms.

## Use the Python API

Build the rig's right-arm retargeter and pass it one `ArmFrame` per update:

```python
# docs: hide
import numpy as np
import warp as wp

from retargetingkit.sources.frames import ArmFrame

device = "cuda:0" if wp.is_cuda_available() else "cpu"
# A right wrist moving 10 cm forward.
stream = [ArmFrame({"right": np.array([x, -0.2, 1.1, 1, 0, 0, 0], dtype=np.float32)}) for x in (0.0, 0.05, 0.1)]
# docs: endhide
from retargetingkit.configs.teleop.arm.visionpro_xarm7 import RIGHT_VISIONPRO_XARM7_CONFIG

retargeter = RIGHT_VISIONPRO_XARM7_CONFIG.retargeters["right"].build(device)

for frame in stream:  # ArmFrame: world-frame wrist [x, y, z, qw, qx, qy, qz]
    command = retargeter.solve(frame)
    qpos = command.qpos  # (7,), xArm7 joint positions
# docs: hide
assert qpos.shape == (7,)
assert np.isfinite(qpos).all()
# docs: endhide
```

The first frame is the calibration origin, so it returns the home pose.

## Retarget an arm and hand

Optimize the arm and fingers from each synchronized `ArmHandFrame`. The wrist pose is in world coordinates; hand keypoints are wrist-local. This example builds only the solvers and needs no tracking device or robot connection.

```python
# docs: hide
import pickle

import numpy as np
import warp as wp

from retargetingkit.assets.example_data import MANUS_DATA_RIGHT_PKL
from retargetingkit.sources.frames import ArmFrame, ArmHandFrame, HandFrame, HandPose

device = "cuda:0" if wp.is_cuda_available() else "cpu"
keypoints = pickle.load(open(MANUS_DATA_RIGHT_PKL, "rb"))["keypoints"][:3]
# Recorded finger poses paired with a wrist moving 10 cm forward.
stream = [
    ArmHandFrame(
        ArmFrame({"right": np.array([0.05 * i, -0.2, 1.1, 1, 0, 0, 0], dtype=np.float32)}),
        HandFrame({"right": HandPose(points)}),
    )
    for i, points in enumerate(keypoints)
]
# docs: endhide
from retargetingkit.configs.teleop.arm_hand.visionpro_xarm7_xhand import RIGHT_VISIONPRO_XARM7_XHAND_CONFIG

config = RIGHT_VISIONPRO_XARM7_XHAND_CONFIG
arm = config.arm_retargeters["right"].build(device)
hand = config.hand_retargeters["right"].build(device)

for frame in stream:  # ArmHandFrame: synchronized wrist and finger poses
    arm_qpos = arm.solve(frame.arm).qpos  # (7,), xArm7 joints
    hand_qpos = hand.solve(frame.hand).qpos  # (12,), XHand joints
# docs: hide
assert arm_qpos.shape == (7,)
assert hand_qpos.shape == (12,)
assert np.isfinite(arm_qpos).all()
assert np.isfinite(hand_qpos).all()
# docs: endhide
```

## Calibration and motion

The first tracked wrist pose becomes the calibration origin and maps to the robot's configured home pose. Wrist movement is then applied relative to that origin.

| Override for the right arm | Effect | Default |
| --- | --- | --- |
| `--retargeters.right.translation-scale` | Scale wrist translation | `1.5` |
| `--retargeters.right.ee-gain` | Blend each new end-effector pose with the previous pose | `1.0` |

A smaller gain smooths movement more. Use `visionpro_xarm7 --help` for all settings. Arm-hand rigs use the prefix `--arm-retargeters.right` instead.

## Connect a robot

Use the [xArm guide](../reference/targets/xarm.md) for SDK installation, network setup, and hardware mode. At startup, hardware mode moves each arm to its model-specific home pose.

To move fingers along with the wrist, choose a supported assembly on the [Vision Pro page](../reference/sources/vision_pro.md). Quest controller teleoperation has its own [standalone workflow](../reference/sources/quest3.md) and button calibration.
