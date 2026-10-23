from __future__ import annotations

import random
from dataclasses import dataclass

from .grid import Cell, GridMap


@dataclass(frozen=True)
class RangeObservation:
    origin: Cell
    visible_blocked: frozenset[Cell]
    visible_free: frozenset[Cell]


class OccupancyMapper:
    """Accumulates observed free/occupied cells; unknown cells remain traversable."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.blocked: set[Cell] = set()
        self.free: set[Cell] = set()

    def update(self, observation: RangeObservation) -> None:
        self.free.update(observation.visible_free)
        self.blocked.update(observation.visible_blocked)
        self.free.difference_update(self.blocked)

    def as_grid(self) -> GridMap:
        return GridMap(self.width, self.height, frozenset(self.blocked))


class EvidenceOccupancyMapper:
    """Occupancy map that requires repeated depth evidence before blocking a cell."""

    def __init__(self, width: int, height: int, occupied_threshold: int = 2):
        self.width = width
        self.height = height
        self.occupied_threshold = occupied_threshold
        self.scores: dict[Cell, int] = {}
        self.collision_blocked: set[Cell] = set()

    def update(self, observation: RangeObservation) -> None:
        for cell in observation.visible_free:
            self.scores[cell] = max(-self.occupied_threshold, self.scores.get(cell, 0) - 1)
        for cell in observation.visible_blocked:
            self.scores[cell] = min(self.occupied_threshold + 2, self.scores.get(cell, 0) + 1)

    def record_collision(self, cell: Cell) -> None:
        self.collision_blocked.add(cell)

    @property
    def blocked(self) -> set[Cell]:
        inferred = {cell for cell, score in self.scores.items() if score >= self.occupied_threshold}
        return inferred | self.collision_blocked

    def as_grid(self, include_depth: bool = True) -> GridMap:
        blocked = self.blocked if include_depth else self.collision_blocked
        return GridMap(self.width, self.height, frozenset(blocked))


class DepthMethod:
    name = "base"

    def transform(self, observation: RangeObservation, rng: random.Random) -> RangeObservation:
        raise NotImplementedError


class OracleDepth(DepthMethod):
    name = "oracle_depth"

    def transform(self, observation: RangeObservation, rng: random.Random) -> RangeObservation:
        return observation


class NoisyDepth(DepthMethod):
    """Controlled depth-error proxy, not a learned model."""

    name = "noisy_depth"

    def __init__(self, miss_probability: float = 0.12, false_positive_probability: float = 0.01):
        self.miss_probability = miss_probability
        self.false_positive_probability = false_positive_probability

    def transform(self, observation: RangeObservation, rng: random.Random) -> RangeObservation:
        occupied = {cell for cell in observation.visible_blocked if rng.random() >= self.miss_probability}
        free: set[Cell] = set()
        for cell in observation.visible_free:
            if rng.random() < self.false_positive_probability:
                occupied.add(cell)
            else:
                free.add(cell)
        return RangeObservation(observation.origin, frozenset(occupied), frozenset(free))


METHODS: dict[str, type[DepthMethod]] = {
    OracleDepth.name: OracleDepth,
    NoisyDepth.name: NoisyDepth,
}

