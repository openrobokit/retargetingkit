# Dexmate Vega-1

Control Vega-1's torso, head, and both arms through the Dexmate SDK. The driver exposes 20 joints and is available as a standalone target; no named teleoperation rig is currently registered.

## Connect

Configure robot discovery and Zenoh communication for your Dexmate setup, then install:

```shell
uv sync --extra dexmate
```

RetargetingKit pins `dexcontrol==0.4.4` and `dexcomm==0.4.2` for the supported discovery stack.

Read current joint positions without sending a motion target:

```shell
uv run --extra dexmate scripts/targets/dexmate/read_qpos.py
```

Press `Ctrl+C` to close the connection.

## Use the driver

```python
from retargetingkit.targets.humanoids.dexmate import DexmateConfig

robot = DexmateConfig().build()
qpos = robot.get_qpos()  # (20,), radians
print(dict(zip(robot.joint_names, qpos)))
robot.close()
```

Use `robot.joint_names` to align commands with the model:

| Group | Joints |
| --- | --- |
| Torso | `torso_j1` through `torso_j3` |
| Head | `head_j1` through `head_j3` |
| Left arm | `L_arm_j1` through `L_arm_j7` |
| Right arm | `R_arm_j1` through `R_arm_j7` |

This is also the order of `get_qpos()` and `set_qpos()`.

## Send motion

With an open connection and a desired 20-joint `target_qpos`, choose one command style:

```python
robot.set_qpos(target_qpos)  # publish repeatedly in your control loop
```

The default `wait_time=0` publishes a setpoint and returns immediately. For one interpolated move that blocks until completion:

```python
robot.set_qpos(target_qpos, wait_time=3.0, max_vel=0.2)
```

`max_vel` is in radians per second. Both forms command the physical robot; call `robot.close()` when finished.
