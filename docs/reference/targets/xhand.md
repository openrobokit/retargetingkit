# XHand

Control 12 XHand joints through RS485 or EtherCAT. The driver accepts URDF radians and sends commands through a background loop at approximately 83 Hz.

## Connect

Install the `xhand_controller` wheel from the XHand vendor. There is no `xhand` package extra.

For RS485, connect the adapter and confirm its serial port. The default baud rate is 3,000,000.

Read five state samples without starting the servo controller:

```shell
uv run scripts/targets/xhand/read_qpos.py \
    --serial-port /dev/ttyUSB0 --samples 5
```

The output includes joint names, positions in radians, and shape checks.

## Preview retargeting

With a configured [Manus glove](../sources/manus.md):

```shell
uv run --extra manus --extra vis scripts/teleop/hand.py \
    manus_xhand --mode viser
```

After checking the preview, `--mode both --target.serial-port /dev/ttyUSB0` connects and commands the hand. The [Vision Pro](../sources/vision_pro.md) guide also covers `visionpro_xhand` and xArm7–XHand rigs.

## Use the driver

This starts the physical hand's servo process and holds its measured pose:

```python
from retargetingkit.targets.hands.xhand import XHandConfig

hand = XHandConfig(serial_port="/dev/ttyUSB0").build()
target_qpos = hand.get_qpos()  # (12,), URDF radians
hand.set_qpos(target_qpos)
hand.close()
```

Use `protocol="EtherCAT"` in the config for EtherCAT. Commands must contain 12 finite values in the driver's documented URDF order.

## Check a small motion

The following moves joint 4 by 0.05 radians and returns it to its starting position:

```shell
uv run scripts/targets/xhand/open_close.py \
    --serial-port /dev/ttyUSB0 --joint 4 --delta 0.05 --reps 1
```

Despite its filename, this diagnostic nudges one joint; it does not open and close all fingers.
