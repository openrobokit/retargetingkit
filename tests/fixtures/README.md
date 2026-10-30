# Offline test fixtures

Robot joint descriptions retain the original kinematics. Meshes are replaced with
5 mm spheres; these fixtures do not validate visual geometry. Available upstream
license and notice files are retained alongside the descriptions.

Recordings contain only the frames needed by the tests: four BVH frames, eight
hand-retargeting frames, and small source/training samples. No MANO or SMPL-X model
weights are included.

The one model checkpoint, `models/ik_model/sharpa_motion_alignment/weights.pt`, is the
released Sharpa neural-hand model used by the documentation example.

The root `conftest.py` routes asset access here and blocks Hugging Face before
collection. Run tests and doctests together with `uv run pytest --xdoctest`.

Sphere collision ignore tables follow RoboKit asset revision
`693882b31562d664e24b8db4a60dcde2d09625fd`.

Gallery regressions use full local robot and motion assets, including user-provided
body models. After preparing those assets, run them outside the fixture routing:

```shell
CUDA_VISIBLE_DEVICES=0 RETARGETINGKIT_ASSET_TESTS=1 uv run pytest --confcutdir=tests \
  tests/gallery/test_hand_retargeting_gallery.py tests/gallery/test_humanoid_gallery.py \
  tests/gallery/test_hand_contact.py tests/gallery/test_hand_object_offline.py
```
