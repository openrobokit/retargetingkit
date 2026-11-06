# Quickstart

## Try it without installing

Choose a free GPU and set `CUDA_VISIBLE_DEVICES`. Then run the interactive hand gallery without installing RetargetingKit:

```shell
uvx --from 'retargetingkit[vis]' \
    python -m retargetingkit.gallery.hand_retargeting
```

Open the Viser URL printed in the terminal. Select a robot and motion, then play or scrub the result. Replace `hand_retargeting` with any gallery below. Each preview records that gallery's Viser scene.

| ![Hand retargeting in Viser](https://openrobokit.com/assets/docs/retargetingkit/assets/gallery/hand_retargeting.webp) | ![Neural hand retargeting in Viser](https://openrobokit.com/assets/docs/retargetingkit/assets/gallery/hand_neural.webp) | ![Hand contact retargeting in Viser](https://openrobokit.com/assets/docs/retargetingkit/assets/gallery/hand_contact.webp) |
| --- | --- | --- |
| **`hand_retargeting`** Retarget tracked hands. | **`hand_neural`** Compare neural hand models. | **`hand_contact`** Refine hand-object contact. |

| ![DexMachina hand-object retargeting in Viser](https://openrobokit.com/assets/docs/retargetingkit/assets/gallery/hand_object_dexmachina.webp) | ![GigaHands hand-object retargeting in Viser](https://openrobokit.com/assets/docs/retargetingkit/assets/gallery/hand_object_gigahands.webp) | ![OakInk2 hand-object retargeting in Viser](https://openrobokit.com/assets/docs/retargetingkit/assets/gallery/hand_object_oakink2.webp) |
| --- | --- | --- |
| **`hand_object_dexmachina`** Explore DexMachina interactions. | **`hand_object_gigahands`** Explore GigaHands interactions. | **`hand_object_oakink2`** Explore OakInk2 interactions. |

| ![Humanoid retargeting in Viser](https://openrobokit.com/assets/docs/retargetingkit/assets/gallery/humanoid.webp) | ![Humanoid contact retargeting in Viser](https://openrobokit.com/assets/docs/retargetingkit/assets/gallery/humanoid_contact.webp) | ![Humanoid terrain retargeting in Viser](https://openrobokit.com/assets/docs/retargetingkit/assets/gallery/humanoid_terrain.webp) |
| --- | --- | --- |
| **`humanoid`** Browse motions and robot presets. | **`humanoid_contact`** Retarget object interactions. | **`humanoid_terrain`** Adapt motion to terrain. |

| ![Full-body object interaction in Viser](https://openrobokit.com/assets/docs/retargetingkit/assets/gallery/full_body_hoi.webp) | ![Full-body scene interaction in Viser](https://openrobokit.com/assets/docs/retargetingkit/assets/gallery/full_body_hsi.webp) |
| --- | --- |
| **`full_body_hoi`** Retarget body and hand interactions. | **`full_body_hsi`** Retarget body motion in furnished scenes. |

To list gallery module names from an installed checkout, run `uv run python -m retargetingkit.gallery`. Gallery data, robot models, and checkpoints download on first use; complete [asset setup](assets.md) first.

## Solve one humanoid retargeting problem

After [installing](installation.md), this program maps one frame of the included BVH recording to a Unitree G1 pose on CPU:

```python
from retargetingkit.assets.sample_bvh import EXAMPLE_BVH
from retargetingkit.retargeting.humanoid.online import HumanoidRetargeterConfig
from retargetingkit.retargeting.humanoid.presets.g1 import g1_aligned
from retargetingkit.sources.misc.human_frame_file import BodyFileSourceConfig

source = BodyFileSourceConfig(EXAMPLE_BVH, use_ankle_as_foot_target=True).build()
retargeter = HumanoidRetargeterConfig(g1_aligned).build("cpu")
command = retargeter.solve(source.get_frame())
print(command.qpos.shape, command.base_pose.shape)  # (29,) (7,)
source.close()
```

The source produces a `BodyFrame`; the retargeter returns G1 joint positions and a floating-base pose.

## More examples

Galleries are interactive demos. For code to copy, open a runnable script below.
Paths are relative to [`src/retargetingkit/examples`](../../src/retargetingkit/examples).

| Directory | Scripts |
| --- | --- |
| [`hand_optimization`](../../src/retargetingkit/examples/hand_optimization) | [`01_online.py`](../../src/retargetingkit/examples/hand_optimization/01_online.py), [`02_offline.py`](../../src/retargetingkit/examples/hand_optimization/02_offline.py), [`03_batch.py`](../../src/retargetingkit/examples/hand_optimization/03_batch.py), [`04_arm.py`](../../src/retargetingkit/examples/hand_optimization/04_arm.py), [`05_online_batch.py`](../../src/retargetingkit/examples/hand_optimization/05_online_batch.py), [`06_dexycb_online.py`](../../src/retargetingkit/examples/hand_optimization/06_dexycb_online.py) |
| [`hand_neural`](../../src/retargetingkit/examples/hand_neural) | [`02_batch.py`](../../src/retargetingkit/examples/hand_neural/02_batch.py) |
| [`humanoid`](../../src/retargetingkit/examples/humanoid) | [`01_online.py`](../../src/retargetingkit/examples/humanoid/01_online.py), [`02_collision.py`](../../src/retargetingkit/examples/humanoid/02_collision.py), [`03_self_collision.py`](../../src/retargetingkit/examples/humanoid/03_self_collision.py), [`04_offline.py`](../../src/retargetingkit/examples/humanoid/04_offline.py), [`05_batch.py`](../../src/retargetingkit/examples/humanoid/05_batch.py), [`06_online_batch.py`](../../src/retargetingkit/examples/humanoid/06_online_batch.py), [`07_contact.py`](../../src/retargetingkit/examples/humanoid/07_contact.py) |
| [`full_body`](../../src/retargetingkit/examples/full_body) | [`01_hoi.py`](../../src/retargetingkit/examples/full_body/01_hoi.py), [`02_hsi.py`](../../src/retargetingkit/examples/full_body/02_hsi.py) |

Example usage:

```shell
uv run --extra vis src/retargetingkit/examples/humanoid/01_online.py
```
