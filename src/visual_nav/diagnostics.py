from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .ai2thor_env import AI2ThorEnvironment, ThorConfig
from .classical_features import (
    camera_matrix,
    estimate_relative_pose,
    harris_corners,
    match_sift,
    sparse_stereo_depth,
)
from .learned_depth import LearnedMetricDepthEstimator
from .metrics import depth_errors
from .pointcloud import pointcloud_depth_to_observation
from .stereo import StereoDepthEstimator


def _select_informative_heading(env: AI2ThorEnvironment) -> None:
    """Rotate to the cardinal view with the most RGB variation."""
    import numpy as np

    candidates: list[tuple[float, int]] = []
    for _ in range(4):
        candidates.append((float(np.asarray(env.event.frame).std()), round(env.yaw) % 360))
        env.controller.step(action="RotateRight")
    best_yaw = max(candidates)[1]
    for _ in range(4):
        if round(env.yaw) % 360 == best_yaw:
            return
        env.controller.step(action="RotateRight")


def _save_rgb(array, path: Path) -> None:
    from PIL import Image
    import numpy as np

    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(path)


def _depth_color(depth, max_depth: float = 5.0):
    import cv2
    import numpy as np

    values = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(values) & (values > 0)
    normalized = np.zeros(values.shape, dtype=np.uint8)
    normalized[valid] = np.clip(values[valid] / max_depth * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _error_color(predicted, truth, max_error: float = 2.0):
    import cv2
    import numpy as np

    prediction = np.asarray(predicted, dtype=np.float32)
    reference = np.asarray(truth, dtype=np.float32)
    valid = np.isfinite(prediction) & (prediction > 0) & np.isfinite(reference) & (reference > 0)
    error = np.zeros(reference.shape, dtype=np.uint8)
    error[valid] = np.clip(np.abs(prediction[valid] - reference[valid]) / max_error * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(error, cv2.COLORMAP_INFERNO)
    colored[~valid] = 0
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _occupancy_image(width, height, truth, predicted, start, goal, scale: int = 20):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width * scale, height * scale), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    for x, y in truth:
        draw.rectangle((x * scale, y * scale, (x + 1) * scale - 1, (y + 1) * scale - 1), fill=(195, 195, 195))
    for x, y in predicted:
        color = (40, 100, 210) if (x, y) in truth else (220, 70, 55)
        draw.rectangle((x * scale, y * scale, (x + 1) * scale - 1, (y + 1) * scale - 1), fill=color)
    for cell, color in ((start, (40, 180, 80)), (goal, (150, 55, 190))):
        x, y = cell
        draw.ellipse((x * scale + 2, y * scale + 2, (x + 1) * scale - 3, (y + 1) * scale - 3), fill=color)
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def _draw_points(rgb, points):
    import cv2
    import numpy as np

    image = np.asarray(rgb).copy()
    for x, y in points:
        cv2.circle(image, (round(float(x)), round(float(y))), 3, (30, 230, 80), 1)
    return image


def _draw_correspondences(rgb_a, rgb_b, points_a, points_b, mask=None, max_lines: int = 100):
    import cv2
    import numpy as np

    first, second = np.asarray(rgb_a), np.asarray(rgb_b)
    height = max(first.shape[0], second.shape[0])
    canvas = np.zeros((height, first.shape[1] + second.shape[1], 3), dtype=np.uint8)
    canvas[:first.shape[0], :first.shape[1]] = first
    canvas[:second.shape[0], first.shape[1]:] = second
    valid = np.ones(len(points_a), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    indices = np.flatnonzero(valid)[:max_lines]
    for index in indices:
        a = tuple(np.rint(points_a[index]).astype(int))
        b = tuple(np.rint(points_b[index]).astype(int) + np.array([first.shape[1], 0]))
        cv2.line(canvas, a, b, (30, 220, 80), 1, cv2.LINE_AA)
        cv2.circle(canvas, a, 2, (255, 210, 30), -1)
        cv2.circle(canvas, b, 2, (255, 210, 30), -1)
    return canvas


def _labeled_panel(images: list[tuple[str, Path]], columns: int, output: Path) -> None:
    """Compose already-saved diagnostics into one presentation-ready figure."""
    from PIL import Image, ImageDraw, ImageFont

    loaded = [(title, Image.open(path).convert("RGB")) for title, path in images]
    tile_width = max(image.width for _, image in loaded)
    tile_height = max(image.height for _, image in loaded)
    label_height = 34
    rows = math.ceil(len(loaded) / columns)
    panel = Image.new(
        "RGB", (columns * tile_width, rows * (tile_height + label_height)), (255, 255, 255)
    )
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default(size=18)
    for index, (title, image) in enumerate(loaded):
        column, row = index % columns, index // columns
        x = column * tile_width + (tile_width - image.width) // 2
        y = row * (tile_height + label_height) + label_height
        text_box = draw.textbbox((0, 0), title, font=font)
        text_width = text_box[2] - text_box[0]
        draw.text(
            (column * tile_width + (tile_width - text_width) // 2, y - 27),
            title,
            fill=(25, 25, 25),
            font=font,
        )
        panel.paste(image, (x, y))
    panel.save(output)


def compose_diagnostic_panels(output: Path) -> None:
    """Build compact depth and occupancy comparison figures from saved images."""
    from PIL import Image

    reference_error = output / "error_simulator.png"
    if not reference_error.exists():
        size = Image.open(output / "depth_simulator.png").size
        Image.new("RGB", size, (0, 0, 0)).save(reference_error)
    _labeled_panel(
        [
            ("Left RGB", output / "rgb.png"),
            ("Simulator depth", output / "depth_simulator.png"),
            ("Classical stereo", output / "depth_classical_stereo.png"),
            ("Learned metric depth", output / "depth_learned.png"),
            ("Right RGB (8 cm baseline)", output / "stereo_right.png"),
            ("Reference (zero error)", output / "error_simulator.png"),
            ("Classical absolute error", output / "error_classical_stereo.png"),
            ("Learned absolute error", output / "error_learned.png"),
        ],
        columns=4,
        output=output / "depth_comparison_panel.png",
    )
    _labeled_panel(
        [
            ("Simulator occupancy", output / "occupancy_simulator_depth.png"),
            ("Classical occupancy", output / "occupancy_classical_stereo.png"),
            ("Sparse feature occupancy", output / "occupancy_classical_sparse_features.png"),
            ("Learned occupancy", output / "occupancy_learned_metric_depth.png"),
        ],
        columns=4,
        output=output / "occupancy_comparison_panel.png",
    )
    _labeled_panel(
        [
            ("Harris corners", output / "harris_corners.png"),
            ("SIFT stereo matches", output / "sift_stereo_matches.png"),
            ("Sparse feature depth", output / "depth_classical_sparse_features.png"),
            ("Temporal pose inliers", output / "temporal_pose_inliers.png"),
        ],
        columns=2,
        output=output / "classical_feature_panel.png",
    )


def capture_diagnostics(env: AI2ThorEnvironment, output: Path, seed: int = 1) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    env.reset(seed)
    _select_informative_heading(env)
    rgb = env.event.frame.copy()
    truth_depth = env.depth_frame().copy()
    focal = env.config.width / (2 * math.tan(math.radians(env.config.field_of_view) / 2))
    stereo = StereoDepthEstimator(focal, env.config.baseline_m)
    left, right = env.capture_stereo_pair()
    stereo_depth = stereo.estimate(left, right)
    sparse = sparse_stereo_depth(left, right, focal, env.config.baseline_m)
    learned_depth = LearnedMetricDepthEstimator().estimate(rgb)

    _save_rgb(rgb, output / "rgb.png")
    _save_rgb(right, output / "stereo_right.png")
    _save_rgb(_depth_color(truth_depth), output / "depth_simulator.png")
    _save_rgb(_depth_color(stereo_depth), output / "depth_classical_stereo.png")
    _save_rgb(_depth_color(sparse.depth), output / "depth_classical_sparse_features.png")
    _save_rgb(_depth_color(learned_depth), output / "depth_learned.png")
    _save_rgb(_error_color(truth_depth, truth_depth), output / "error_simulator.png")
    _save_rgb(_error_color(stereo_depth, truth_depth), output / "error_classical_stereo.png")
    _save_rgb(_error_color(sparse.depth, truth_depth), output / "error_classical_sparse_features.png")
    _save_rgb(_error_color(learned_depth, truth_depth), output / "error_learned.png")

    metrics: dict[str, object] = {"scene": env.config.scene, "seed": seed, "methods": {}}
    origin_x, origin_z = env.grid_origin
    for name, depth in (
        ("simulator_depth", truth_depth),
        ("classical_stereo", stereo_depth),
        ("classical_sparse_features", sparse.depth),
        ("learned_metric_depth", learned_depth),
    ):
        observation = pointcloud_depth_to_observation(
            depth, env.position, env.camera_position, env.floor_y, env.yaw, env.horizon,
            env.config.field_of_view, origin_x, origin_z, env.config.grid_size,
            env.ground_truth.width, env.ground_truth.height,
            stride=1 if name == "classical_sparse_features" else 6,
        )
        abs_rel, rmse, coverage = depth_errors(depth, truth_depth)
        metrics["methods"][name] = {
            "depth_abs_rel": abs_rel,
            "depth_rmse": rmse,
            "depth_coverage": coverage,
            "observed_occupied_cells": len(observation.visible_blocked),
        }
        occupancy = _occupancy_image(
            env.ground_truth.width, env.ground_truth.height,
            env.ground_truth.blocked, observation.visible_blocked, env.start, env.goal,
        )
        occupancy.save(output / f"occupancy_{name}.png")

    harris = harris_corners(rgb)
    stereo_matches = match_sift(left, right)
    _save_rgb(_draw_points(rgb, harris), output / "harris_corners.png")
    _save_rgb(
        _draw_correspondences(
            left, right, stereo_matches.points_a, stereo_matches.points_b,
        ),
        output / "sift_stereo_matches.png",
    )
    before_motion = env.event.frame.copy()
    moved = env.controller.step(action="MoveAhead")
    after_motion = moved.frame.copy()
    temporal_matches, pose = estimate_relative_pose(
        before_motion, after_motion,
        camera_matrix(env.config.width, env.config.height, env.config.field_of_view),
    )
    pose_mask = pose.inlier_mask if pose is not None else None
    _save_rgb(
        _draw_correspondences(
            before_motion, after_motion, temporal_matches.points_a,
            temporal_matches.points_b, pose_mask,
        ),
        output / "temporal_pose_inliers.png",
    )
    metrics["feature_geometry"] = {
        "harris_corners": len(harris),
        "stereo_sift_ratio_matches": sparse.ratio_matches,
        "stereo_epipolar_ransac_inliers": sparse.epipolar_inliers,
        "temporal_sift_ratio_matches": len(temporal_matches.points_a),
        "temporal_essential_ransac_inliers": pose.inliers if pose is not None else 0,
        "temporal_move_succeeded": bool(moved.metadata.get("lastActionSuccess", False)),
        "translation_direction": pose.translation_direction.tolist() if pose is not None else None,
        "rotation": pose.rotation.tolist() if pose is not None else None,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    compose_diagnostic_panels(output)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Save representative AI2-THOR CV diagnostics")
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/diagnostics"))
    args = parser.parse_args()
    env = AI2ThorEnvironment(ThorConfig(scene=args.scene))
    try:
        print(json.dumps(capture_diagnostics(env, args.output, args.seed), indent=2))
    finally:
        env.stop()


if __name__ == "__main__":
    main()
