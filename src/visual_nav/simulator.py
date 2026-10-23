from __future__ import annotations

import random
from dataclasses import dataclass

from .grid import Cell, GridMap
from .perception import RangeObservation
from .planning import astar


@dataclass
class SyntheticRoom:
    ground_truth: GridMap
    start: Cell
    goal: Cell
    position: Cell
    sensor_radius: int = 4
    collisions: int = 0

    @classmethod
    def generate(cls, seed: int, width: int = 20, height: int = 14, density: float = 0.18) -> "SyntheticRoom":
        rng = random.Random(seed)
        start, goal = (1, height // 2), (width - 2, height // 2)
        for _ in range(100):
            blocked = {
                (x, y)
                for x in range(width)
                for y in range(height)
                if (x, y) not in (start, goal) and rng.random() < density
            }
            grid = GridMap(width, height, frozenset(blocked))
            if astar(grid, start, goal) is not None:
                return cls(grid, start, goal, start)
        raise RuntimeError("failed to generate a connected room")

    def observe(self) -> RangeObservation:
        x0, y0 = self.position
        visible: set[Cell] = set()
        for x in range(max(0, x0 - self.sensor_radius), min(self.ground_truth.width, x0 + self.sensor_radius + 1)):
            for y in range(max(0, y0 - self.sensor_radius), min(self.ground_truth.height, y0 + self.sensor_radius + 1)):
                if abs(x - x0) + abs(y - y0) <= self.sensor_radius:
                    visible.add((x, y))
        blocked = visible & set(self.ground_truth.blocked)
        return RangeObservation(self.position, frozenset(blocked), frozenset(visible - blocked))

    def move(self, target: Cell) -> bool:
        if target not in self.ground_truth.neighbors(self.position):
            self.collisions += 1
            return False
        self.position = target
        return True

