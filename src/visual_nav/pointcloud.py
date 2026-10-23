from __future__ import annotations

import math
from collections.abc import Sequence

from .depth_projection import _bresenham
from .grid import Cell
from .perception import RangeObservation


def project_depth_to_world(
    depth,
    camera_position: Sequence[float],
    yaw_degrees: float,
    horizon_degrees: float,
    vertical_fov_degrees: float,
    max_range_m: float = 5.0,
    stride: int = 6,
):
    """Project radial depth pixels to Nx3 world points.

    Camera coordinates use +x right, +y up, and +z forward. AI2-THOR yaw 0
    faces world +z and positive camera horizon looks downward.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install the 'vision' optional dependencies") from exc
    image = np.asarray(depth, dtype=np.float32)
    if image.ndim != 2 or not image.size:
        raise ValueError("depth must be a non-empty 2-D image")
    if stride <= 0 or max_range_m <= 0:
        raise ValueError("stride and maximum range must be positive")
    height, width = image.shape
    rows = np.arange(0, height, stride)
    columns = np.arange(0, width, stride)
    uu, vv = np.meshgrid(columns, rows)
    ranges = image[vv, uu]
    valid = np.isfinite(ranges) & (ranges > 0) & (ranges <= max_range_m)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)
    focal = 0.5 * height / math.tan(math.radians(vertical_fov_degrees) / 2.0)
    x = (uu[valid] + 0.5 - width / 2.0) / focal
    y = -(vv[valid] + 0.5 - height / 2.0) / focal
    z = np.ones_like(x)
    rays = np.stack((x, y, z), axis=1)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    points = rays * ranges[valid, None]

    pitch = math.radians(horizon_degrees)
    cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
    pitched_y = points[:, 1] * cos_pitch - points[:, 2] * sin_pitch
    pitched_z = points[:, 1] * sin_pitch + points[:, 2] * cos_pitch
    points[:, 1], points[:, 2] = pitched_y, pitched_z

    yaw = math.radians(yaw_degrees)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    world_x = cos_yaw * points[:, 0] + sin_yaw * points[:, 2]
    world_z = -sin_yaw * points[:, 0] + cos_yaw * points[:, 2]
    points[:, 0], points[:, 2] = world_x, world_z
    points += np.asarray(camera_position, dtype=np.float32)
    return points


def pointcloud_depth_to_observation(
    depth,
    origin: Cell,
    camera_position: Sequence[float],
    floor_y: float,
    yaw_degrees: float,
    horizon_degrees: float,
    vertical_fov_degrees: float,
    grid_origin_x: float,
    grid_origin_z: float,
    grid_size_m: float,
    grid_width: int,
    grid_height: int,
    obstacle_min_height_m: float = 0.08,
    obstacle_max_height_m: float = 1.5,
    max_range_m: float = 5.0,
    stride: int = 6,
) -> RangeObservation:
    """Create a 2-D occupancy observation from a height-filtered point cloud."""
    points = project_depth_to_world(
        depth, camera_position, yaw_degrees, horizon_degrees,
        vertical_fov_degrees, max_range_m=max_range_m, stride=stride,
    )
    blocked: set[Cell] = set()
    endpoints: set[Cell] = set()
    for x, y, z in points:
        cell = (
            round((float(x) - grid_origin_x) / grid_size_m),
            round((float(z) - grid_origin_z) / grid_size_m),
        )
        if not (0 <= cell[0] < grid_width and 0 <= cell[1] < grid_height):
            continue
        endpoints.add(cell)
        height = float(y) - floor_y
        if obstacle_min_height_m <= height <= obstacle_max_height_m:
            blocked.add(cell)
    free: set[Cell] = {origin}
    for endpoint in endpoints:
        ray = _bresenham(origin, endpoint)
        free.update(ray[:-1] if endpoint in blocked else ray)
    free.difference_update(blocked)
    return RangeObservation(origin, frozenset(blocked), frozenset(free))

