"""Visualize fixed Manus right-hand keypoints with an inverse-fitted MANO mesh.

Example:
    uv run --extra vis --extra mano scripts/targets/mano/visualize_manus_data.py
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import viser

from retargetingkit.assets import example_data
from retargetingkit.sources.frames import MANO21_EDGES
from retargetingkit.utils.mano import ManoFitConfig, ManoKeypointFitter, default_mano_root
from robokit.smplx import load_mano


DEFAULT_DATA_PATH = example_data.MANUS_DATA_RIGHT_PKL


def _colors_for_frame(frame_idx: int, pinch_frames: set[int], open_frames: set[int]) -> np.ndarray:
    if frame_idx in pinch_frames:
        color = (255, 80, 70)
    elif frame_idx in open_frames:
        color = (80, 180, 255)
    else:
        color = (40, 170, 230)
    return np.full((21, 3), color, dtype=np.uint8)


def _draw_skeleton(server: viser.ViserServer, root: str, keypoints: np.ndarray, colors: np.ndarray) -> None:
    color = (int(colors[0, 0]), int(colors[0, 1]), int(colors[0, 2]))
    for edge_idx, (start, end) in enumerate(MANO21_EDGES):
        server.scene.add_spline_catmull_rom(
            f"{root}/bone_{edge_idx}",
            points=np.asarray([keypoints[start], keypoints[end]], dtype=np.float32),
            color=color,
            thickness=3.0,
            thickness_units="screen",
        )


def _load_bundle(path: Path) -> tuple[np.ndarray, set[int], set[int], str]:
    with open(path, "rb") as file_handle:
        bundle = pickle.load(file_handle)
    keypoints = np.asarray(bundle["keypoints"], dtype=np.float32)
    assert keypoints.ndim == 3 and keypoints.shape[1:] == (21, 3), f"bad keypoints shape: {keypoints.shape}"
    pinch_frames = set(int(i) for i in bundle["pinch_frames"])
    open_frames = set(int(i) for i in bundle["open_frames"])
    source = str(bundle["source"])
    return keypoints, pinch_frames, open_frames, source


def main() -> None:
    """Start the Manus example data viewer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--rotate-deg", type=float, default=90.0)
    parser.add_argument("--no-fit-mano", dest="fit_mano", action="store_false")
    parser.add_argument("--max-fit-frames", type=int, default=120, help="0 fits the full sequence.")
    parser.add_argument("--fit-iterations", type=int, default=220)
    parser.add_argument("--fit-lr", type=float, default=0.02)
    parser.add_argument("--fit-device", default="auto")
    parser.add_argument("--side-by-side-offset", type=float, default=0.24)
    parser.add_argument("--smoke-test", action="store_true")
    parser.set_defaults(fit_mano=True)
    args = parser.parse_args()

    keypoints, pinch_frames, open_frames, source = _load_bundle(args.data_path)
    fit_frames = keypoints.shape[0]
    if args.smoke_test:
        fit_frames = 1
    elif args.max_fit_frames > 0:
        fit_frames = min(fit_frames, args.max_fit_frames)
    keypoints = keypoints[:fit_frames]
    num_frames = keypoints.shape[0]

    fitted_vertices = np.zeros((0, 778, 3), dtype=np.float32)
    fitted_keypoints = np.zeros((0, 21, 3), dtype=np.float32)
    faces = np.zeros((0, 3), dtype=np.int32)
    fit_losses = np.zeros((0,), dtype=np.float32)
    keypoint_errors = np.zeros((0, 21), dtype=np.float32)
    if args.fit_mano:
        spec = load_mano(default_mano_root() / "MANO_RIGHT.npz", side="right")
        fitter = ManoKeypointFitter(
            spec,
            ManoFitConfig(iterations=args.fit_iterations, lr=args.fit_lr, device=args.fit_device),
        )
        fit_result = fitter.fit(keypoints)
        fitted_vertices = fit_result.vertices
        fitted_keypoints = fit_result.keypoints
        fit_losses = fit_result.losses
        faces = spec.faces.astype(np.int32)
        keypoint_errors = np.linalg.norm(fitted_keypoints - keypoints, axis=-1).astype(np.float32)

    rotate_rad = np.deg2rad(-args.rotate_deg)
    root_wxyz = (
        float(np.cos(0.5 * rotate_rad)),
        0.0,
        0.0,
        float(np.sin(0.5 * rotate_rad)),
    )

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/grid", width=0.4, height=0.4)
    server.scene.add_frame(
        "/script",
        show_axes=True,
        axes_length=0.03,
        axes_radius=0.001,
        wxyz=root_wxyz,
        position=(-0.5 * args.side_by_side_offset, 0.0, 0.0),
    )
    target_points = server.scene.add_point_cloud(
        "/script/fixed_keypoints",
        keypoints[0],
        colors=_colors_for_frame(0, pinch_frames, open_frames),
        point_size=0.006,
        point_shape="circle",
    )
    _draw_skeleton(server, "/script/fixed_skeleton", keypoints[0], _colors_for_frame(0, pinch_frames, open_frames))

    mesh = None
    if args.fit_mano:
        server.scene.add_frame(
            "/mano",
            show_axes=True,
            axes_length=0.03,
            axes_radius=0.001,
            wxyz=root_wxyz,
            position=(0.5 * args.side_by_side_offset, 0.0, 0.0),
        )
        mesh = server.scene.add_mesh_simple(
            "/mano/mesh",
            vertices=fitted_vertices[0],
            faces=faces,
            color=(230, 210, 190),
            opacity=0.68,
        )

    with server.gui.add_folder("Playback", expand_by_default=True):
        play = server.gui.add_checkbox("Play", True)
        frame_slider = server.gui.add_slider("Frame", min=0, max=num_frames - 1, step=1, initial_value=0)
        server.gui.add_text("Source", source, disabled=True)
        server.gui.add_text("Frames", str(num_frames), disabled=True)
        server.gui.add_text("Pinch frames", str(len(pinch_frames)), disabled=True)
        server.gui.add_text("Open frames", str(len(open_frames)), disabled=True)
        if args.fit_mano:
            server.gui.add_text("MANO fit loss", f"{float(fit_losses[-1]):.6f}", disabled=True)
            server.gui.add_text("Mean fit error m", f"{float(keypoint_errors.mean()):.6f}", disabled=True)
            server.gui.add_text("Max fit error m", f"{float(keypoint_errors.max()):.6f}", disabled=True)

    def update(frame_idx: int) -> None:
        colors = _colors_for_frame(frame_idx, pinch_frames, open_frames)
        target_points.points = keypoints[frame_idx]
        target_points.colors = colors
        _draw_skeleton(server, "/script/fixed_skeleton", keypoints[frame_idx], colors)
        if args.fit_mano:
            assert mesh is not None
            mesh.vertices = fitted_vertices[frame_idx]

    update(0)
    print(f"[mano-manus-data] Open: http://localhost:{server.get_port()}")
    print(f"[mano-manus-data] data={args.data_path}")
    print(f"[mano-manus-data] keypoints={keypoints.shape} source={source}")
    if args.fit_mano:
        print(
            f"[mano-manus-data] fitted_mano vertices={fitted_vertices.shape} "
            f"predicted_keypoints={fitted_keypoints.shape} final_loss={float(fit_losses[-1]):.6f} "
            f"mean_error_m={float(keypoint_errors.mean()):.6f} max_error_m={float(keypoint_errors.max()):.6f}"
        )
        print("[mano-manus-data] left=/script fixed_input_keypoints right=/mano mesh_only")
    if args.smoke_test:
        print("[mano-manus-data] smoke test complete")
        return

    frame_idx = 0
    dt = 1.0 / args.fps
    while True:
        if play.value:
            frame_idx = (frame_idx + 1) % num_frames
            frame_slider.value = frame_idx
            update(frame_idx)
        else:
            next_idx = int(frame_slider.value)
            if next_idx != frame_idx:
                frame_idx = next_idx
                update(frame_idx)
        time.sleep(dt)


if __name__ == "__main__":
    main()
