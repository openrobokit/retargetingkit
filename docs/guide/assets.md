# Assets

Recordings, checkpoints, and robot models download on first use. Importing `retargetingkit` alone downloads nothing.

## Cache

| Asset | Source | Cache |
| --- | --- | --- |
| Recordings, calibration, training data | `openrobokit/retargetingkit` | `~/.retargetingkit/assets/<revision>/` |
| Neural checkpoints | Same dataset, `models/ik_model/` | Inside the same revision cache |
| Robot URDFs, meshes, collision and contact data | RoboKit assets | `~/.robokit/assets/<revision>/` |
| MANO / SMPL-X body models | Local model files | `ROBOKIT_BODY_MODELS` or `~/.robokit/body_models/` |
| YCB meshes | YCB benchmark S3 | `~/.retargetingkit/ycb/` |

## Resolve an asset

Use named accessors so examples and applications share the same data:

```python
from retargetingkit.assets import calib, neural_hand, sample_bvh

calibration = calib.get("manus_calib")
checkpoint = neural_hand.get("sharpa_motion_alignment_contact_inhand")
motion = sample_bvh.EXAMPLE_BVH
```

Robot models come from RoboKit:

```python
from robokit.assets.robots.end_effectors import sharpa_hand

urdf = sharpa_hand.URDF_PATH_RIGHT
```

## Pinned revision

`retargetingkit.assets.REVISION` pins the dataset commit; accessors call `robokit.assets.fetch` with it, the dataset id, and `~/.retargetingkit`. Bump the pin after publishing new data. To download again, delete the matching `~/.retargetingkit/assets/<revision>/.done/` marker.
