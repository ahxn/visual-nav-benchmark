from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .ai2thor_env import AI2ThorEnvironment, ThorConfig
from .classical_features import camera_matrix, estimate_relative_pose, harris_corners
from .planning import astar


def rotation_error_degrees(rotation) -> float:
    import numpy as np

    value = (float(np.trace(rotation)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, value))))


def direction_error_degrees(direction, expected=(0.0, 0.0, -1.0)) -> float:
    import numpy as np

    first = np.asarray(direction, dtype=float)
    second = np.asarray(expected, dtype=float)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0:
        return math.nan
    cosine = float(np.dot(first, second) / denominator)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _orient_toward(env: AI2ThorEnvironment, target) -> None:
    x, z = env.position
    desired = {(0, 1): 0, (1, 0): 90, (0, -1): 180, (-1, 0): 270}[
        (target[0] - x, target[1] - z)
    ]
    while round(env.yaw) % 360 != desired:
        delta = (desired - env.yaw) % 360
        action = "RotateRight" if delta <= 180 else "RotateLeft"
        env.controller.step(action=action)


def run_visual_odometry_sequence(
    env: AI2ThorEnvironment,
    seed: int,
    max_pairs: int = 20,
) -> dict[str, object]:
    """Evaluate feature-based pose recovery on controlled forward image pairs."""
    env.reset(seed)
    path = astar(env.ground_truth, env.start, env.goal)
    if not path:
        raise RuntimeError("selected start and goal have no ground-truth path")
    intrinsics = camera_matrix(
        env.config.width, env.config.height, env.config.field_of_view
    )
    rows: list[dict[str, object]] = []
    for step, target in enumerate(path[1:max_pairs + 1], start=1):
        _orient_toward(env, target)
        before = env.event.frame.copy()
        harris_before = len(harris_corners(before))
        moved = env.controller.step(action="MoveAhead")
        if not moved.metadata.get("lastActionSuccess", False):
            rows.append({
                "step": step, "matches": 0, "inliers": 0, "inlier_ratio": 0.0,
                "translation_direction_error_deg": None,
                "rotation_error_deg": None, "harris_corners": harris_before,
                "move_succeeded": False,
            })
            break
        after = moved.frame.copy()
        matches, pose = estimate_relative_pose(before, after, intrinsics)
        rows.append({
            "step": step,
            "matches": len(matches.points_a),
            "inliers": pose.inliers if pose is not None else 0,
            "inlier_ratio": (
                pose.inliers / len(matches.points_a)
                if pose is not None and len(matches.points_a) else 0.0
            ),
            "translation_direction_error_deg": (
                direction_error_degrees(pose.translation_direction) if pose is not None else None
            ),
            "rotation_error_deg": rotation_error_degrees(pose.rotation) if pose is not None else None,
            "harris_corners": harris_before,
            "move_succeeded": True,
        })
    valid = [row for row in rows if row["translation_direction_error_deg"] is not None]
    summary = {
        "scene": env.config.scene,
        "seed": seed,
        "pairs_attempted": len(rows),
        "poses_recovered": len(valid),
        "pose_recovery_rate": len(valid) / len(rows) if rows else 0.0,
        "mean_matches": sum(int(row["matches"]) for row in rows) / len(rows) if rows else 0.0,
        "mean_inlier_ratio": sum(float(row["inlier_ratio"]) for row in rows) / len(rows) if rows else 0.0,
        "mean_translation_direction_error_deg": (
            sum(float(row["translation_direction_error_deg"]) for row in valid) / len(valid)
            if valid else None
        ),
        "mean_rotation_error_deg": (
            sum(float(row["rotation_error_deg"]) for row in valid) / len(valid)
            if valid else None
        ),
        "steps": rows,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SIFT/RANSAC visual odometry")
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results/visual_odometry.json"))
    args = parser.parse_args()
    env = AI2ThorEnvironment(ThorConfig(scene=args.scene))
    try:
        result = run_visual_odometry_sequence(env, args.seed, args.max_pairs)
    finally:
        env.stop()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    steps_path = args.output.with_suffix(".steps.csv")
    if result["steps"]:
        with steps_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["steps"][0]))
            writer.writeheader()
            writer.writerows(result["steps"])
    printable = {key: value for key, value in result.items() if key != "steps"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
