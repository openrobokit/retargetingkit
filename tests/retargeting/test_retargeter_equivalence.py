import pickle
from dataclasses import replace

import numpy as np
import torch
import warp as wp

from retargetingkit.assets import example_data
from retargetingkit.sources.glove.manus_conversion import manus_to_mano_keypoints
from robokit.assets.robots.end_effectors import inspire_hand, sharpa_hand
from robokit.helpers.hand_retargeting import HandRetargeterOnline
from robokit.helpers.hand_retargeting.presets import inspire, sharpa
from robokit.robo import Robot


def _retargeter(hand: str) -> HandRetargeterOnline:
    wp.init()
    if hand == "inspire":
        urdf_path = inspire_hand.URDF_PATH_RIGHT
        collision_path = inspire_hand.COLLISION_SPHERE_PATH_RIGHT
        ignore_path = inspire_hand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT
        spec = inspire.spec
        config = inspire.online
    else:
        urdf_path = sharpa_hand.URDF_PATH_RIGHT
        collision_path = sharpa_hand.COLLISION_SPHERE_PATH_RIGHT
        ignore_path = sharpa_hand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT
        spec = sharpa.spec
        config = sharpa.online
    robot = Robot.load(
        urdf_path,
        load_collision_spheres=True,
        collision_spheres_path=collision_path,
        self_collision_ignore_path=ignore_path,
    )
    retargeter = HandRetargeterOnline(
        config=config,
        robot=robot,
        spec=replace(spec, floating_base=False),
        # The Warp LM solver converges to device-specific solutions; the goldens are CPU results.
        device=wp.get_device("cpu"),
    )
    retargeter.warmup(1)
    return retargeter


GOLDEN_QPOS = np.array(
    [
        (-0.100000, 0.140000, 1.645106, 1.615671, 1.666878, 1.700000),
        (-0.100000, 0.140000, 1.645208, 1.615856, 1.667044, 1.699586),
        (-0.100000, 0.140000, 1.645555, 1.616102, 1.669236, 1.699601),
        (-0.100000, 0.140000, 1.645950, 1.615909, 1.667529, 1.699761),
        (-0.100000, 0.140000, 1.645444, 1.615694, 1.668404, 1.698258),
        (-0.100000, 0.140000, 1.644610, 1.615512, 1.668021, 1.697904),
        (-0.100000, 0.140000, 1.644813, 1.615845, 1.667444, 1.698283),
        (-0.100000, 0.140000, 1.644411, 1.615647, 1.667776, 1.698408),
    ],
    dtype=np.float32,
)


def _synthetic_manus_frames(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(n):
        positions = np.zeros((25, 3), dtype=np.float32)
        for base, x in [(1, -0.04), (6, -0.02), (11, 0.0), (16, 0.02), (21, 0.04)]:
            for i in range(4):
                positions[base + i] = np.array([x, 0.02 * (i + 1), 0.01 * i], dtype=np.float32)
        positions += rng.normal(scale=0.006, size=(25, 3)).astype(np.float32)
        rotations = np.zeros((25, 4), dtype=np.float32)
        rotations[:, 3] = 1.0  # identity xyzw
        frames.append((positions, rotations))
    return frames


# Recorded MANO-21 keypoints (wrist-local); 8 frames evenly spanning the take, replayed through one retargeter.
MANUS_DATA = example_data.MANUS_DATA_RIGHT_PKL


def _manus_keypoints(n: int) -> np.ndarray:
    keypoints = pickle.load(open(MANUS_DATA, "rb"))["keypoints"]
    return keypoints[np.linspace(0, keypoints.shape[0] - 1, n).astype(int)]


INSPIRE_MANUS_GOLDEN = np.array(
    [
        (0.636519, 0.140000, 0.212730, 0.053703, 0.056476, 0.226774),
        (0.901912, 0.140909, 0.217076, 0.193186, 0.411823, 0.377459),
        (0.959563, 0.140545, 0.503760, 0.246360, 0.366089, 0.405010),
        (1.095738, 0.220855, 0.328509, 0.147816, 0.244860, 0.390894),
        (0.768532, 0.188513, 0.284261, 0.136864, 0.565283, 0.337772),
        (0.421119, 0.169108, 0.741466, 0.644375, 0.946194, 0.766421),
        (0.463986, 0.289487, 0.808336, 0.735101, 0.968279, 0.892538),
        (0.360661, 0.229692, 0.572281, 0.499748, 0.649311, 0.659911),
    ],
    dtype=np.float32,
)

# fmt: off
SHARPA_MANUS_GOLDEN = np.array(
    [
        (0.614928, 0.094894, 0.147432, -0.349100, 0.529268, 0.265322, -0.096865, 0.017260, 0.011466, 0.112812, 0.078411, 0.000215, 0.000000, 0.069415, 0.085503, 0.000088, 0.000038, 0.000000, -0.174533, 0.336129, 0.247237, 0.898600),
        (1.168586, 0.144943, 0.349533, -0.349100, 0.537773, 0.276830, -0.182407, 0.019611, 0.011428, 0.470894, -0.011579, 0.122880, 0.042959, 0.760000, -0.237004, 0.622209, 0.325321, 0.143671, 0.072562, 0.073546, 0.461340, 1.019847),
        (0.984902, 0.104158, 0.255959, -0.218277, 0.512763, 0.859853, 0.001684, 0.599589, 0.342269, 0.452944, 0.018512, 0.000000, 0.000000, 0.154156, -0.013491, 0.210133, 0.126701, 0.072819, -0.058983, 0.211239, 0.360829, 0.967265),
        (1.300517, 0.078927, 0.974107, -0.349100, 0.837838, 0.001349, -0.120925, 0.099502, 0.107533, 0.029664, 0.043074, 0.000000, 0.000000, 0.018926, 0.044333, 0.066687, 0.043492, 0.081788, -0.102432, 0.137935, 0.330379, 0.948034),
        (0.442086, -0.063172, 0.373032, 0.203588, 0.508073, 0.174651, -0.167504, 0.151610, 0.116889, 0.172290, 0.050262, 0.038273, 0.011434, 0.807998, -0.097874, 0.833466, 0.436371, 0.029812, -0.174533, 0.273358, 0.269429, 0.929207),
        (0.152743, -0.226779, 0.222840, 0.349100, 0.443629, 1.403430, 0.067241, 1.321789, 0.671382, 1.531418, 0.069718, 1.191624, 0.497739, 1.437398, 0.086619, 1.354951, 0.638704, 0.129375, 0.995042, 0.215482, 1.301146, 1.207829),
        (0.420657, 0.349100, 1.176093, -0.260630, 0.731833, 0.539052, -0.010635, 0.917687, 0.550186, 0.630206, 0.040771, 0.770166, 0.364079, 0.662745, -0.011046, 0.876236, 0.421095, 0.167677, 0.383826, 0.113410, 0.954434, 1.128145),
        (0.083455, -0.133208, 0.589369, -0.235049, 0.423511, -0.022120, -0.058895, 0.390813, 0.257389, 0.066230, 0.132384, 0.211293, 0.072353, -0.005141, 0.163107, 0.258518, 0.099691, 0.000000, -0.174533, 0.349100, 0.319181, 0.865474),
    ],
    dtype=np.float32,
)
# fmt: on


class TestRobokitOnlineGolden:
    def test_synthetic_inspire(self) -> None:
        retargeter = _retargeter("inspire")
        out = np.stack(
            [
                retargeter.solve_numpy(manus_to_mano_keypoints(positions, rotations))[0, 7:]
                for positions, rotations in _synthetic_manus_frames(8)
            ]
        )
        torch.testing.assert_close(torch.from_numpy(out), torch.from_numpy(GOLDEN_QPOS), rtol=3e-3, atol=3e-3)

    def test_recorded_inspire(self) -> None:
        retargeter = _retargeter("inspire")
        out = np.stack([retargeter.solve_numpy(points)[0, 7:] for points in _manus_keypoints(8)])
        torch.testing.assert_close(torch.from_numpy(out), torch.from_numpy(INSPIRE_MANUS_GOLDEN), rtol=3e-3, atol=3e-3)

    def test_recorded_sharpa(self) -> None:
        retargeter = _retargeter("sharpa")
        out = np.stack([retargeter.solve_numpy(points)[0, 7:] for points in _manus_keypoints(8)])
        torch.testing.assert_close(torch.from_numpy(out), torch.from_numpy(SHARPA_MANUS_GOLDEN), rtol=3e-3, atol=3e-3)
