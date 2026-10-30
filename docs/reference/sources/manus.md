# Manus

Use Manus gloves for hand tracking and optimization-based retargeting to XHand or Sharpa Wave. `ManusSource` converts each 25-node glove skeleton to a wrist-local MANO-21 `HandFrame`.

## Connect the gloves

Connect the gloves through the Manus host software, then install the SDK runtime and viewer from the RetargetingKit repository:

```shell
uv sync --extra manus --extra vis
```

Preview both raw skeletons:

```shell
uv run --extra manus --extra vis \
    scripts/sources/manus/visualize_keypoints.py --side both
```

Open <http://localhost:8080> and flex each hand to check its 25-node skeleton. The utility detects sides from glove metadata. If detection is ambiguous, pass `--manus-left-id` or `--manus-right-id` with a decimal or `0x` glove ID.

## Preview retargeting

```shell
CUDA_VISIBLE_DEVICES=<FREE_GPU> uv run --extra manus --extra vis \
    scripts/teleop/hand.py manus_xhand --mode viser
```

Use `manus_sharpa` for Sharpa Wave. These named rigs use online optimization and default to the right hand. Select `--mode both` for hardware plus Viser or `--mode real` for hardware only. Set `--target.serial-port` for XHand or `--target.device-sn` for Sharpa.

Manus currently supplies hand motion only and is not a source for `humanoid_hand.py`.

## Record poses

Capture one frame per Enter press; type `q` and press Enter to save and finish:

```shell
uv run --extra manus --extra vis scripts/sources/manus/record_glove.py \
    --hand right --output recording.npz
```

For a finite continuous recording, set the frame count and rate:

```shell
uv run --extra manus --extra vis scripts/sources/manus/record_glove.py \
    --hand right --no-viz --frames 300 --rate 30 \
    --output manus_recording.npz
```

Use `--hand left` or `--hand both` to change the recorded sides. The preview applies glove calibration; the NPZ preserves raw SDK positions and rotations.

## Recording format

Every field begins with `left_` or `right_` for its saved side. Positions are meters in the SDK's requested right-handed, Z-up coordinate system viewed along X (`unitScale=1.0`).

| Field suffix | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `positions` | `(N, 25, 3)` | `float32` | Raw node positions |
| `rotations` | `(N, 25, 4)` | `float32` | Raw quaternions in `xyzw` order |
| `publish_time` | `(N,)` | `uint64` | Compressed SDK timestamp |
| `recv_ns` | `(N,)` | `int64` | Host monotonic time at callback receipt |
| `timestamps` | `(N,)` | `float64` | Seconds since recording start |
| `glove_id` | `(N,)` | `uint32` | Hardware glove ID |
| `side_code` | `(N,)` | `int32` | Side enum: `1` left, `2` right |

Node ranges are wrist `0`, thumb `1–4`, index `5–9`, middle `10–14`, ring `15–19`, and little finger `20–24`. Each non-thumb finger runs from metacarpal through knuckle and two intermediate joints to the tip.

Treat `publish_time` as the opaque `ManusTimestamp`, not Unix time. Use `recv_ns` for timing intervals within one process run.
