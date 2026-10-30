# xArm6 and xArm7

Control a six- or seven-joint xArm through the UFACTORY SDK. Both drivers accept joint positions in radians and provide end-effector IK and teaching modes.

## Connect

Find the controller IP on the power box and open `http://<XARM_IP>:18333` for UFACTORY Studio. From the RetargetingKit checkout, install the driver:

```shell
uv sync --extra xarm
```

For xArm7, read five samples without enabling motion or sending position commands:

```shell
uv run --extra xarm scripts/targets/xarm7/read_qpos.py \
    --ip <XARM_IP> --samples 5
```

The xArm6 `read_qpos.py` script starts the active velocity controller, including motion enable and error clearing. Treat it as a controller session.

## Preview teleoperation

With [Vision Pro](../sources/vision_pro.md) configured:

```shell
uv run --extra vision-pro --extra vis scripts/teleop/arm.py \
    visionpro_xarm7 --mode viser --source.ip <VISION_PRO_IP>
```

Use `visionpro_xarm6` for xArm6. The preview does not connect the arm. The Vision Pro guide covers IP overrides and arm-hand rigs; [Quest 3](../sources/quest3.md) provides a separate controller workflow.

## Use the driver

This snippet starts a hardware controller and holds the measured pose. Replace `target_qpos` with your desired joint vector to move the arm.

```python
from retargetingkit.targets.arms.xarm7 import XArm7Config, XArmMode

arm = XArm7Config("<XARM_IP>", mode=XArmMode.VELOCITY).build()
target_qpos = arm.get_qpos()  # (7,), URDF radians
arm.set_qpos(target_qpos)
arm.close()
```

For six joints, import `XArm6Config` from `retargetingkit.targets.arms.xarm6`.

| Mode | Behavior |
| --- | --- |
| `VELOCITY` | Default; smooths position tracking through velocity control |
| `SERVO` | Streams joint position targets |
| `TEACHING` | Enables gravity-compensated teaching |

Both drivers run a background control loop at 250 Hz by default. Starting a driver enables motion, clears warnings and errors, and selects the configured control mode.
