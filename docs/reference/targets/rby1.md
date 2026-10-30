# RBY1-A

Control the RBY1-A's seven-joint right arm through the Rainbow Robotics SDK, with separate commands for the two-joint head. This is a standalone driver; no named teleoperation rig is currently registered.

## Connect

Find the robot's gRPC address in its control panel, then install:

```shell
uv sync --extra rby1
```

Use the complete address, including the port, such as `<RBY1_IP>:50051`.

## Use the driver

Building the driver powers on the robot, enables servos and the control manager, and starts a command stream. This snippet holds the measured right-arm pose:

```python
from retargetingkit.targets.humanoids.rby1 import RBY1AConfig, RBY1AMode

arm = RBY1AConfig("<RBY1_IP>:50051", mode=RBY1AMode.VELOCITY).build()
target_qpos = arm.get_qpos()  # (7,), radians
arm.set_qpos(target_qpos)
arm.close()
```

Replace `target_qpos` with the desired right-arm joint positions. `get_head_qpos()` and `set_head_qpos()` expose the head separately. Close the driver with `close()`.

| Mode | Behavior |
| --- | --- |
| `VELOCITY` | Default; converts position error to a smoothed command |
| `SERVO` | Streams position targets with a velocity limit |
| `TEACHING` | Enables right-arm gravity compensation |

The background loop runs at 100 Hz by default. Head position commands remain active in all three modes.

## Inspect feedback

```shell
uv run --extra rby1 scripts/targets/rby1/read_qpos.py \
    --ip <RBY1_IP>:50051
```

This script starts the same active controller, including power and servo enable. It prints feedback while holding the initial pose; it is not a passive state reader.
