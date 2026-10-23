from __future__ import annotations

import heapq
from math import inf

from .grid import Cell, GridMap


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid: GridMap, start: Cell, goal: Cell) -> list[Cell] | None:
    """Return an inclusive shortest path, or None when no path exists."""
    if not grid.free(start) or not grid.free(goal):
        return None
    frontier: list[tuple[int, int, Cell]] = [(manhattan(start, goal), 0, start)]
    came_from: dict[Cell, Cell | None] = {start: None}
    cost: dict[Cell, int] = {start: 0}
    sequence = 0
    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current == goal:
            path: list[Cell] = []
            cursor: Cell | None = current
            while cursor is not None:
                path.append(cursor)
                cursor = came_from[cursor]
            return list(reversed(path))
        for nxt in grid.neighbors(current):
            candidate = cost[current] + 1
            if candidate < cost.get(nxt, inf):
                cost[nxt] = candidate
                came_from[nxt] = current
                sequence += 1
                heapq.heappush(frontier, (candidate + manhattan(nxt, goal), sequence, nxt))
    return None

