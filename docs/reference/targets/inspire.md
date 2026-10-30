# Inspire Hand

Control six actuated finger joints over USB serial. The driver converts URDF radians to Inspire's serial joint order and normalized position range.

## Connect

Connect the serial adapter and give your user access to its device port. On Linux this commonly requires membership in `dialout`; log out and back in after changing groups.

```shell
uv sync --extra inspire
```

The combined G1 preset uses `/dev/ttyUSB0` for the left hand and `/dev/ttyUSB1` for the right, both at 115200 baud with hand ID `1`. Confirm the actual ports before connecting.

## Use the driver

```python
from retargetingkit.targets.hands.inspire import InspireHandConfig

hand = InspireHandConfig("/dev/ttyUSB0", 1, 115200).build()
qpos = hand.get_qpos()  # (6,), URDF radians
hand.close()
```

Construction opens the serial connection and starts a background reader. Commands begin after the first `hand.set_qpos(target_qpos)` call; feedback updates asynchronously.

Joint order is thumb rotation, thumb bend, index, middle, ring, and little finger. Use `hand.joint_names` when aligning a robot model.

## Teleoperate with PICO

After completing [PICO setup](../sources/pico.md), run:

```shell
uv run --extra pico --extra inspire scripts/teleop/humanoid_hand.py \
    pico_g1_inspire --source.no-use-ankle-as-foot-target \
    --left-hand.serial-port /dev/ttyUSB0 \
    --right-hand.serial-port /dev/ttyUSB1
```

This connects both physical hands and publishes body motion to SONIC. `B` toggles the right hand open/closed; `Y` toggles the left. The rig has no preview mode and uses preset hand poses rather than tracked fingers.

For continuous Vision Pro arm-hand control, use the `visionpro_xarm7_inspire` rig described in [Vision Pro](../sources/vision_pro.md). For G1 deployment, follow [humanoid teleoperation](../../guide/humanoid_retargeting.md).
