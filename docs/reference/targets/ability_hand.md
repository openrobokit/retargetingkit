# Ability Hand

Control the PSYONIC Ability Hand's six joints through a USB serial adapter. The driver accepts radians and runs a background servo process, with position commands ramped at 60 degrees per second by default.

## Connect

Power and connect the hand according to its hardware manual. Use a stable Linux device path when available:

```shell
ls /dev/serial/by-id/
uv sync --extra ability
```

Your user needs access to the serial device, commonly through the `dialout` group. The driver uses 460800 baud.

## Preview teleoperation

The named [Vision Pro](../sources/vision_pro.md) rig combines Ability Hand with xArm7:

```shell
uv run --extra vision-pro --extra vis scripts/teleop/arm_hand.py \
    visionpro_xarm7_ability --mode viser --source.ip <VISION_PRO_IP>
```

This previews motion without opening either robot target. Follow the Vision Pro guide to configure the arm IP and hand port before selecting `real` or `both`.

## Use the driver

Starting the driver sends serial position commands, including zero-position commands during initial synchronization. The following is a hardware controller session:

```python
from retargetingkit.targets.hands.ability_hand import AbilityHandConfig

hand = AbilityHandConfig("/dev/serial/by-id/<ABILITY_HAND>").build()
target_qpos = hand.get_qpos()  # (6,), radians
hand.set_qpos(target_qpos)
hand.close()
```

Replace `target_qpos` with the desired hand pose for your control loop.

## Check motion

The diagnostic below opens and closes the physical hand once:

```shell
uv run --extra ability scripts/targets/ability_hand/open_close.py \
    --port /dev/serial/by-id/<ABILITY_HAND> --reps 1
```

`scripts/targets/ability_hand/read_qpos.py` also starts the active servo process and sends startup commands. It is not a passive serial reader.
