from __future__ import annotations

import math
from typing import Any

from .ai2thor_env import AI2ThorEnvironment
from .benchmark import EpisodeResult
from .classical_features import sparse_stereo_depth
from .depth_projection import scanline_depth_to_observation
from .metrics import chamfer_distance, depth_errors, occupancy_iou, spl
from .perception import EvidenceOccupancyMapper
from .planning import astar
from .pointcloud import pointcloud_depth_to_observation
from .stereo import StereoDepthEstimator


def run_thor_episode(
    env: AI2ThorEnvironment,
    seed: int,
    max_steps: int = 500,
    depth_source: str = "simulator_depth",
    learned_estimator: Any | None = None,
    mapper_mode: str = "pointcloud",
    collision_recovery: bool = True,
) -> EpisodeResult:
    """Closed-loop navigation from selected visual depth, without truth for planning."""
    if depth_source not in {
        "simulator_depth", "classical_stereo", "classical_sparse_features",
        "learned_metric_depth",
    }:
        raise ValueError(f"unsupported depth source: {depth_source}")
    if mapper_mode not in {"pointcloud", "scanline"}:
        raise ValueError(f"unsupported mapper mode: {mapper_mode}")
    env.reset(seed)
    threshold = 1 if depth_source in {"learned_metric_depth", "classical_sparse_features"} else 2
    mapper = EvidenceOccupancyMapper(
        env.ground_truth.width, env.ground_truth.height, occupied_threshold=threshold
    )
    focal_length_px = env.config.width / (
        2.0 * math.tan(math.radians(env.config.field_of_view) / 2.0)
    )
    stereo = StereoDepthEstimator(focal_length_px, env.config.baseline_m)
    learned = learned_estimator
    if depth_source == "learned_metric_depth":
        if learned is None:
            from .learned_depth import LearnedMetricDepthEstimator

            learned = LearnedMetricDepthEstimator()
    movement_steps = 0
    depth_metrics: list[tuple[float, float, float]] = []
    while env.position != env.goal and env.action_count < max_steps:
        reference_depth = env.depth_frame().copy()
        if depth_source == "simulator_depth":
            depth = reference_depth
        elif depth_source == "classical_stereo":
            left, right = env.capture_stereo_pair()
            depth = stereo.estimate(left, right)
        elif depth_source == "classical_sparse_features":
            left, right = env.capture_stereo_pair()
            depth = sparse_stereo_depth(
                left, right, focal_length_px, env.config.baseline_m
            ).depth
        else:
            depth = learned.estimate(env.event.frame)
        depth_metrics.append(depth_errors(depth, reference_depth))
        if mapper_mode == "pointcloud":
            origin_x, origin_z = env.grid_origin
            observation = pointcloud_depth_to_observation(
                depth,
                origin=env.position,
                camera_position=env.camera_position,
                floor_y=env.floor_y,
                yaw_degrees=env.yaw,
                horizon_degrees=env.horizon,
                vertical_fov_degrees=env.config.field_of_view,
                grid_origin_x=origin_x,
                grid_origin_z=origin_z,
                grid_size_m=env.config.grid_size,
                grid_width=env.ground_truth.width,
                grid_height=env.ground_truth.height,
                stride=1 if depth_source == "classical_sparse_features" else 6,
            )
        else:
            observation = scanline_depth_to_observation(
                depth.tolist() if hasattr(depth, "tolist") else depth,
                origin=env.position,
                yaw_degrees=env.yaw,
                field_of_view_degrees=env.config.field_of_view,
                grid_size_m=env.config.grid_size,
            )
        mapper.update(observation)
        # Quantized ray endpoints can land on traversable start/goal cells.
        mapper.scores[env.position] = -mapper.occupied_threshold
        mapper.scores[env.goal] = -mapper.occupied_threshold
        path = astar(mapper.as_grid(), env.position, env.goal)
        if not path and collision_recovery:
            # Treat low-resolution visual obstacles as uncertain when they disconnect
            # the map. Collision observations remain hard constraints.
            path = astar(mapper.as_grid(include_depth=False), env.position, env.goal)
        if not path or len(path) < 2:
            break
        target = path[1]
        if not env.move_to_neighbor(target):
            mapper.record_collision(target)
        movement_steps += 1
    success = env.position == env.goal
    predicted, truth = set(mapper.blocked), set(env.ground_truth.blocked)
    shortest = env.shortest_path_length()
    finite_abs_rel = [item[0] for item in depth_metrics if math.isfinite(item[0])]
    finite_rmse = [item[1] for item in depth_metrics if math.isfinite(item[1])]
    mean_abs_rel = sum(finite_abs_rel) / len(finite_abs_rel) if finite_abs_rel else None
    mean_rmse = sum(finite_rmse) / len(finite_rmse) if finite_rmse else None
    mean_coverage = (
        sum(item[2] for item in depth_metrics) / len(depth_metrics)
        if depth_metrics else None
    )
    return EpisodeResult(
        method=depth_source,
        seed=seed,
        success=success,
        collisions=env.collisions,
        path_length=movement_steps,
        shortest_path=shortest,
        spl=spl(success, shortest, movement_steps),
        occupancy_iou=occupancy_iou(predicted, truth),
        chamfer_distance=chamfer_distance(predicted, truth),
        depth_abs_rel=mean_abs_rel,
        depth_rmse=mean_rmse,
        depth_coverage=mean_coverage,
        scene=env.config.scene,
        recovery_policy="collision_only" if collision_recovery else "disabled",
    )

