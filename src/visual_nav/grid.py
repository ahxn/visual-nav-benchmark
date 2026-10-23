from __future__ import annotations

import math
from dataclasses import dataclass

Cell = tuple[int, int]


@dataclass(frozen=True)
class GridMap:
    width: int
    height: int
    blocked: frozenset[Cell]

    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and cell not in self.blocked

    def neighbors(self, cell: Cell) -> list[Cell]:
        x, y = cell
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [candidate for candidate in candidates if self.free(candidate)]

    def inflate(self, radius: int) -> "GridMap":
        if radius <= 0:
            return self
        expanded: set[Cell] = set()
        for ox, oy in self.blocked:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if math.hypot(dx, dy) <= radius:
                        cell = (ox + dx, oy + dy)
                        if self.in_bounds(cell):
                            expanded.add(cell)
        return GridMap(self.width, self.height, frozenset(expanded))

