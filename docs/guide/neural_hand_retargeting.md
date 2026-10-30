# Neural Hand Retargeting

Train a model for one robot hand, then map wrist-local MANO-21 keypoints to joint positions in URDF order. Curated checkpoints support Allegro, Sharpa Wave, and XHand.

## How Inference Works

The source returns a `HandFrame`; the selected checkpoint maps its wrist-local keypoints to a `QPosCommand`. A checkpoint belongs to one robot hand, so choose the matching hand and model before connecting a target.

## Try a trained model

Run the gallery from the RetargetingKit checkout:

```shell
uv run --extra vis python -m retargetingkit.gallery.hand_neural
```

Open the printed Viser address to choose a hand and checkpoint. The selected model downloads on first use and is cached. These commands default to CUDA; select an available GPU with `CUDA_VISIBLE_DEVICES`, or use `--device cpu`.

## Use the Python API

Load a named checkpoint for its hand and pass it one `HandFrame` per update:

```python
# docs: hide
import pickle

import numpy as np
import warp as wp

from retargetingkit.assets.example_data import MANUS_DATA_RIGHT_PKL
from retargetingkit.sources.frames import HandFrame, HandPose

device = "cuda:0" if wp.is_cuda_available() else "cpu"
# Three recorded wrist-local MANO-21 frames from a right Manus glove.
keypoints = pickle.load(open(MANUS_DATA_RIGHT_PKL, "rb"))["keypoints"][:3]
stream = [HandFrame({"right": HandPose(points)}) for points in keypoints]
# docs: endhide
from retargetingkit.retargeting.hand.neural import NeuralHandAssetRetargeterConfig
from retargetingkit.retargeting.hand.presets import HANDS

config = NeuralHandAssetRetargeterConfig("sharpa_motion_alignment", HANDS["sharpa"])
retargeter = config.build(device)

for frame in stream:  # HandFrame: right-hand MANO-21 keypoints
    command = retargeter.solve(frame)
    qpos = command.qpos  # (22,), Sharpa joint angles
# docs: hide
assert qpos.shape == (22,)
assert np.isfinite(qpos).all()
# docs: endhide
```

With a configured [Manus glove](../reference/sources/manus.md), preview live Sharpa inference:

```shell
uv run --extra manus --extra vis scripts/teleop/hand.py \
    manus_sharpa_neural --mode viser
```

This preset loads `sharpa_motion_alignment_contact_inhand`. Set `--mode real` or `--mode both` to connect and command a physical [Sharpa hand](../reference/targets/sharpa_hand.md).

## Train a recipe

Five training recipes live in `scripts/teleop/neural_hand/configs/training/`:

| Hand | Recipe filename |
| --- | --- |
| Allegro | `allegro_motion_alignment.yaml` |
| XHand | `xhand_motion_alignment.yaml` |
| Sharpa | `sharpa_motion_alignment.yaml` |
| Sharpa with contact data | `sharpa_motion_alignment_contact.yaml` |
| Sharpa with in-hand contact data | `sharpa_motion_alignment_contact_inhand.yaml` |

For example:

```shell
uv run scripts/teleop/neural_hand/train.py \
    scripts/teleop/neural_hand/configs/training/xhand_motion_alignment.yaml
```

The YAML selects named motion and calibration assets, loss weights, and an output directory. Training writes `train.yaml` with data provenance and `weights.pt` to `ik_model/xhand_motion_alignment/` for this recipe. The output directory must not already exist; choose a new `output` path for another run.

## Record your own motion

Record the right Manus glove at 30 Hz:

```shell
uv run --extra manus --extra vis scripts/teleop/neural_hand/record_motion.py \
    --output motion.npz
```

Press `Enter` to start, then `Enter` again to stop and save. Add `--no-viz` to record without a viewer. To use this recording for training, register it as a named asset in `retargetingkit.assets.calib` and reference that name under `data.unsupervised` in your YAML.

Replay the recording through your trained model:

```shell
uv run --extra vis scripts/teleop/neural_hand/visualize.py xhand \
    --checkpoints ik_model/xhand_motion_alignment --data motion.npz
```

## Add contact supervision

The Sharpa contact workflow generates candidate YCB grasps, then captures matching human poses:

```shell
uv run --extra contact scripts/teleop/neural_hand/contact_generate.py \
    scripts/teleop/neural_hand/configs/contact/sharpa_ycb_contact.yaml

uv run --extra contact --extra manus --extra vis \
    scripts/teleop/neural_hand/contact_capture.py \
    --specs sharpa_grasp.npz --output training_data.npz
```

Use the capture viewer to record, skip, and save samples. Register the result as a calibration asset and reference it under `data.contact` in a training YAML.

For paired human/robot measurements, `record_pairs.py` supports Manus and Sharpa. It starts in neural control mode and commands the physical hand; `t` toggles teaching mode, `c` captures a pair, and `q` saves and exits.
