"""Visualize the local MANO target with mesh, skeleton, and PCA pose controls.

Example:
    uv run --extra vis scripts/targets/mano/visualize.py
"""

import argparse
import time

import numpy as np
import viser

from retargetingkit.sources.frames import MANO21_EDGES
from retargetingkit.utils.mano import ManoHand, ManoHandConfig


MANO_ROOT_WXYZ = (np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0)


def _draw_skeleton(server: viser.ViserServer, keypoints: np.ndarray) -> None:
    for edge_idx, (start, end) in enumerate(MANO21_EDGES):
        server.scene.add_spline_catmull_rom(
            f"/mano/bone_{edge_idx}",
            points=np.asarray([keypoints[start], keypoints[end]], dtype=np.float32),
            color=(40, 170, 230),
            thickness=3.0,
            thickness_units="screen",
        )


def main() -> None:
    """Start the MANO viser scene."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    hand = ManoHand(ManoHandConfig(side=args.side, device=args.device))
    pose = np.zeros(48, dtype=np.float32)
    state = hand.forward_pca(pose)
    wrist = state.keypoints[0, 0].copy()
    vertices = state.vertices[0] - wrist
    keypoints = state.keypoints[0] - wrist

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/grid", width=0.4, height=0.4)
    server.scene.add_frame("/mano", show_axes=True, axes_length=0.03, axes_radius=0.001, wxyz=MANO_ROOT_WXYZ)
    mesh = server.scene.add_mesh_simple(
        "/mano/mesh",
        vertices=vertices,
        faces=state.faces,
        color=(230, 210, 190),
        opacity=0.72,
    )
    points = server.scene.add_point_cloud(
        "/mano/keypoints",
        keypoints,
        colors=np.full((21, 3), (40, 170, 230), dtype=np.uint8),
        point_size=0.006,
        point_shape="circle",
    )
    _draw_skeleton(server, keypoints)

    with server.gui.add_folder("Pose", expand_by_default=True):
        pca_sliders = [
            server.gui.add_slider(f"PCA {idx}", min=-2.5, max=2.5, step=0.01, initial_value=0.0) for idx in range(6)
        ]
        wrist_x = server.gui.add_slider("Wrist X", min=-1.5, max=1.5, step=0.01, initial_value=0.0)
        wrist_y = server.gui.add_slider("Wrist Y", min=-1.5, max=1.5, step=0.01, initial_value=0.0)
        wrist_z = server.gui.add_slider("Wrist Z", min=-1.5, max=1.5, step=0.01, initial_value=0.0)

    def update() -> None:
        pose[:] = 0.0
        pose[0:3] = np.array([wrist_x.value, wrist_y.value, wrist_z.value], dtype=np.float32)
        for idx, slider in enumerate(pca_sliders):
            pose[3 + idx] = float(slider.value)
        current = hand.forward_pca(pose)
        current_wrist = current.keypoints[0, 0].copy()
        current_keypoints = current.keypoints[0] - current_wrist
        mesh.vertices = current.vertices[0] - current_wrist
        points.points = current_keypoints
        _draw_skeleton(server, current_keypoints)

    for slider in [*pca_sliders, wrist_x, wrist_y, wrist_z]:
        slider.on_update(lambda _event: update())

    update()
    print(f"[mano-visualize] Open: http://localhost:{server.get_port()}")
    print(f"[mano-visualize] vertices={state.vertices.shape} keypoints={state.keypoints.shape}")
    if args.smoke_test:
        print("[mano-visualize] smoke test complete")
        return

    while True:
        time.sleep(1.0 / args.fps)


if __name__ == "__main__":
    main()
