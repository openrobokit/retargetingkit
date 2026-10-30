# Sharpa Wave

Control 22 finger joints and read tactile measurements from five fingers per hand. The driver accepts joint positions in URDF radians and exposes position, velocity, and torque feedback.

## Connect

Get the Sharpa SDK from Sharpa and extract it so `~/.retargetingkit/sharpa-sdk/` holds its `lib/` and `python/` folders. The SDK's native extension must match your Python environment.

If the native loader reports a missing `libpython`, add your Python environment's library directory and `~/.retargetingkit/sharpa-sdk/lib` to `LD_LIBRARY_PATH` before launching.

`device_sn=None` selects the first discovered hand. Set a device serial number when several hands are connected.

Read five samples in floating mode:

```shell
uv run scripts/targets/sharpa_hand/read_qpos.py --samples 5
```

This changes the hand to floating mode for backdriving and prints angles in degrees. The Python API returns radians.

## Preview retargeting

With [Manus](../sources/manus.md) configured, preview optimization:

```shell
uv run --extra manus --extra vis scripts/teleop/hand.py \
    manus_sharpa --mode viser
```

Use `manus_sharpa_neural` for the curated neural model. `--mode real` or `--mode both` commands the physical hand; add `--target.device-sn <SERIAL>` to select it. [Vision Pro](../sources/vision_pro.md) also supports hand-only and xArm–Sharpa rigs.

## Use the driver

Construction connects the hand, selects position mode, and starts SDK control:

```python
from retargetingkit.targets.hands.sharpa_hand import SharpaHand

hand = SharpaHand(speed_coeff=0.3, current_coeff=0.3)
qpos, qvel, torque = hand.get_state()  # each has 22 values
hand.set_qpos(qpos)
hand.close()
```

Use `set_control_mode("floating")` for backdriving and `set_control_mode("position")` to resume position control.

## Read touch data

Fetch five samples from the right thumb:

```shell
uv run scripts/targets/sharpa_hand/tactile_fetch.py \
    --channel 0 --samples 5 --calibrate
```

This connects the SDK in position mode and calibrates before reading. Frames may contain `RAW`, `DEFORM`, and `F6` data. The driver API uses channels `0–4` for right thumb through pinky and `5–9` for the left; the current diagnostic labels right-hand channels only.

`scripts/targets/sharpa_hand/wave_example.py --cycles 3` moves joint 7 by 3 degrees and returns it, three times.
