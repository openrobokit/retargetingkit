# Canonical frames

Source adapters convert native device data into a shared format. Solvers receive positions in meters, scalar-first quaternions, and explicit joint names.

## Units and poses

| Value | Convention |
| --- | --- |
| Position | Meters |
| Robot joint position | Radians, in actuated URDF order |
| Quaternion | `(qw, qx, qy, qz)` |
| Pose | `(x, y, z, qw, qx, qy, qz)` |
| Body coordinates | Right-handed Z-up world frame |

Canonical poses have shape `(7,)`. Native device matrices and quaternion orders are converted inside each source adapter.

## Human motion

| Frame | Content |
| --- | --- |
| `HandFrame` | `hands`: side → wrist-local `HandPose` |
| `ArmFrame` | `wrists`: side → world-frame `(7,)` pose |
| `ArmHandFrame` | Synchronized `arm` and `hand` frames |
| `BodyFrame` | `joint_names`, `(J, 7)` world poses, `human_height` |

Sides are `"left"` and `"right"`. A frame can contain one available hand or both; check the requested side before consuming it. Body joint names travel with the data so the solver can select the joints it needs.

## Hand keypoints

`HandPose.positions` contains 21 wrist-local MANO-21 points. It also carries joint orientations, a valid-point mask, and joint names.

```python
import numpy as np
from retargetingkit.sources.frames import HandFrame, HandPose, MANO21

positions = np.zeros((21, 3), dtype=np.float32)  # shape example
frame = HandFrame({"right": HandPose(positions)})
frame.hands["right"].positions.shape  # (21, 3)
MANO21.WRIST, MANO21.INDEX_TIP        # 0, 8
MANO21.THUMB_TIP, MANO21.PINKY_TIP    # 4, 20
```

Replace the zero array with tracked keypoints before solving. See the [Frame API](../reference/api/frames.md) for all fields and defaults.

## Robot output

`QPosCommand` carries `(Q,)` joint positions. `BasePoseQPosCommand` adds a `(7,)` floating-base world pose. Neither type includes a time or batch dimension; those belong to recordings and lower-level solver APIs.
