# Humanoid Retargeting

| Human motion mapped to a humanoid. | Humanoid motion on terrain. |
| --- | --- |

Map recorded or live body motion to G1 at 50 Hz. The runtime sends robot root pose and joint positions to its configured targets; the SONIC target derives joint velocities for whole-body control.

## How Retargeting Works

A body source returns named joint poses in a `BodyFrame`. The humanoid retargeter maps them to G1 links and solves a base pose plus joint positions. A Viser target displays the command; a SONIC target publishes it for whole-body control.

## Preview recorded motion

Start with the packaged example, which opens a Viser preview without a SONIC publisher:

```shell
uv run --extra vis python -m retargetingkit.examples.humanoid.01_online
```

Open the printed viewer address. Required assets download on first use; press `Ctrl+C` to stop.

## Use the Python API

Build the G1 retargeter and pass it one `BodyFrame` per update:

```python
# docs: hide
import numpy as np
import warp as wp

from retargetingkit.assets.sample_bvh import EXAMPLE_BVH
from retargetingkit.sources.misc.human_frame_file import BodyFileSourceConfig

device = "cuda:0" if wp.is_cuda_available() else "cpu"
source = BodyFileSourceConfig(EXAMPLE_BVH, use_ankle_as_foot_target=True).build()
frame = source.get_frame()
# docs: endhide
from retargetingkit.retargeting.humanoid.online import HumanoidRetargeterConfig
from retargetingkit.retargeting.humanoid.presets.g1 import g1_aligned

retargeter = HumanoidRetargeterConfig(g1_aligned).build(device)

command = retargeter.solve(frame)  # BodyFrame: named world-frame joint poses
base_pose = command.base_pose  # (7,), [x, y, z, qw, qx, qy, qz]
qpos = command.qpos  # (29,), G1 joints in URDF order
# docs: hide
assert base_pose.shape == (7,)
assert qpos.shape == (29,)
assert np.isfinite(qpos).all()
source.close()
# docs: endhide
```

## Publish to SONIC

The bundled BVH preset needs no tracking device:

```shell
uv run --extra vis scripts/teleop/humanoid.py bvh_g1_sonic_viser
```

Every named humanoid preset publishes to SONIC on port `5556`. The `_viser` suffix adds visualization of the same motion; it does not disable SONIC. Run a single publisher on that port. The live runtime defaults to `cuda:0`; select an available GPU with `CUDA_VISIBLE_DEVICES`, or set `--device cpu`.

Override a recording and performer height:

```shell
uv run --extra vis scripts/teleop/humanoid.py bvh_g1_sonic_viser \
    --source.path /path/to/motion.bvh --source.human-height 1.80
```

## Choose a live source

Complete [Axis Studio](../reference/sources/axis_studio.md) or [PICO](../reference/sources/pico.md) setup before starting its rig.

```shell
uv run --extra axis --extra vis scripts/teleop/humanoid.py \
    axisstudio_g1_sonic_viser --source.port 7012 --source.human-height 1.75

uv run --extra pico --extra vis scripts/teleop/humanoid.py \
    pico_g1_sonic_viser --source.no-use-ankle-as-foot-target
```

The PICO override uses tracked foot poses instead of substituting ankle poses. For a PICO recording, use `picofile_g1_sonic_viser --source.path /path/to/recording.npz`. Remove `_viser` from these preset names to publish without a viewer.

Use `uv run scripts/teleop/humanoid.py <RIG> --help` to inspect all overrides.

## Control both Inspire hands

This combined rig opens one PICO connection, publishes G1 motion to SONIC, and connects both physical Inspire hands. It has no preview mode.

```shell
uv run --extra pico --extra inspire scripts/teleop/humanoid_hand.py \
    pico_g1_inspire --source.no-use-ankle-as-foot-target \
    --left-hand.serial-port /dev/ttyUSB0 \
    --right-hand.serial-port /dev/ttyUSB1
```

`B` toggles the right hand between open and closed; `Y` toggles the left. These are preset poses, not continuous finger tracking. See [Inspire Hand](../reference/targets/inspire.md) for the serial connection.

## Run G1 in simulation

Configure the deployment checkout described in [G1 and SONIC](../reference/targets/g1.md), then start a SONIC rig and launch:

```shell
bash scripts/tools/launch_sonic_sim.sh
```

The launcher opens MuJoCo and deployment in a `tmux` session. In the deployment pane, press `]`, then `Enter`, to start the policy. The [G1 target guide](../reference/targets/g1.md) also covers connection to a real robot.
