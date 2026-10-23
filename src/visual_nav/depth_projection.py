from __future__ import annotations

import math
from collections.abc import Sequence

from .grid import Cell
from .perception import RangeObservation


def _bresenham(start: Cell, end: Cell) -> list[Cell]:
    """Integer cells on a line, including both endpoints."""
    x0, y0 = start
    x1, y1 = end
    dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
    dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
    error = dx + dy
    cells: list[Cell] = []
    while True:
        cells.append((x0, y0))
        if (x0, y0) == (x1, y1):
            return cells
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy


def scanline_depth_to_observation(
    depth: Sequence[Sequence[float]],
    origin: Cell,
    yaw_degrees: float,
    field_of_view_degrees: float,
    grid_size_m: float,
    max_range_m: float = 5.0,
    row_fraction: float = 0.5,
    column_stride: int = 4,
) -> RangeObservation:
    """Project a horizontal depth scanline into a 2-D occupancy observation.

    Depth values are radial distances in meters. Each valid ray clears traversed
    cells; its endpoint is occupied only when it terminates before max range.
    This conservative projection is usable with simulator depth or stereo depth.
    """
    if not depth or not depth[0]:
        raise ValueError("depth image cannot be empty")
    if grid_size_m <= 0 or max_range_m <= 0:
        raise ValueError("grid size and maximum range must be positive")
    height, width = len(depth), len(depth[0])
    row = min(height - 1, max(0, round((height - 1) * row_fraction)))
    free: set[Cell] = {origin}
    blocked: set[Cell] = set()
    for column in range(0, width, column_stride):
        distance = float(depth[row][column])
        if not math.isfinite(distance) or distance <= 0:
            continue
        clipped = min(distance, max_range_m)
        offset = ((column + 0.5) / width - 0.5) * field_of_view_degrees
        angle = math.radians(yaw_degrees + offset)
        # AI2-THOR yaw 0 faces world +z; grid coordinates are (+x, +z).
        dx = round(math.sin(angle) * clipped / grid_size_m)
        dz = round(math.cos(angle) * clipped / grid_size_m)
        endpoint = (origin[0] + dx, origin[1] + dz)
        ray = _bresenham(origin, endpoint)
        hit = distance < max_range_m
        free.update(ray[:-1] if hit else ray)
        if hit and endpoint != origin:
            blocked.add(endpoint)
    free.difference_update(blocked)
    return RangeObservation(origin, frozenset(blocked), frozenset(free))

