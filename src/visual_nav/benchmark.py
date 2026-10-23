from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from .metrics import chamfer_distance, occupancy_iou, spl
from .perception import DepthMethod, OccupancyMapper
from .planning import astar
from .simulator import SyntheticRoom


@dataclass(frozen=True)
class EpisodeResult:
    method: str
    seed: int
    success: bool
    collisions: int
    path_length: int
    shortest_path: int
    spl: float
    occupancy_iou: float
    chamfer_distance: float
    depth_abs_rel: float | None = None
    depth_rmse: float | None = None
    depth_coverage: float | None = None
    scene: str | None = None
    recovery_policy: str | None = None

    def __post_init__(self) -> None:
        # Result files written before the ablation field was introduced used the
        # collision-only fallback whenever they came from AI2-THOR.
        if self.scene is not None and self.recovery_policy is None:
            object.__setattr__(self, "recovery_policy", "collision_only")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_episode(method: DepthMethod, seed: int, max_steps: int = 250) -> EpisodeResult:
    room = SyntheticRoom.generate(seed)
    mapper = OccupancyMapper(room.ground_truth.width, room.ground_truth.height)
    rng = random.Random(seed * 1009 + sum(map(ord, method.name)))
    shortest = len(astar(room.ground_truth, room.start, room.goal) or []) - 1
    steps = 0
    attempted_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    while room.position != room.goal and steps < max_steps:
        mapper.update(method.transform(room.observe(), rng))
        plan = astar(mapper.as_grid(), room.position, room.goal)
        if not plan or len(plan) < 2:
            break
        origin, target = room.position, plan[1]
        moved = room.move(target)
        steps += 1
        if not moved:
            # Collision is itself a range observation: mark the attempted cell occupied.
            mapper.blocked.add(target)
            attempted_edges.add((origin, target))
            if len(attempted_edges) > room.ground_truth.width * room.ground_truth.height:
                break
    predicted = set(mapper.blocked)
    truth = set(room.ground_truth.blocked)
    success = room.position == room.goal
    return EpisodeResult(
        method=method.name,
        seed=seed,
        success=success,
        collisions=room.collisions,
        path_length=steps,
        shortest_path=shortest,
        spl=spl(success, shortest, steps),
        occupancy_iou=occupancy_iou(predicted, truth),
        chamfer_distance=chamfer_distance(predicted, truth),
    )


def summarize(results: list[EpisodeResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    conditions = sorted({(result.method, result.recovery_policy or "") for result in results})
    for method, recovery_policy in conditions:
        subset = [
            result for result in results
            if result.method == method and (result.recovery_policy or "") == recovery_policy
        ]
        rows.append({
            "method": method,
            "recovery_policy": recovery_policy or None,
            "episodes": len(subset),
            "success_rate": sum(r.success for r in subset) / len(subset),
            "mean_collisions": sum(r.collisions for r in subset) / len(subset),
            "mean_path_length": sum(r.path_length for r in subset) / len(subset),
            "mean_spl": sum(r.spl for r in subset) / len(subset),
            "mean_occupancy_iou": sum(r.occupancy_iou for r in subset) / len(subset),
            "mean_chamfer": sum(r.chamfer_distance for r in subset) / len(subset),
            "mean_depth_abs_rel": _optional_mean(r.depth_abs_rel for r in subset),
            "mean_depth_rmse": _optional_mean(r.depth_rmse for r in subset),
            "mean_depth_coverage": _optional_mean(r.depth_coverage for r in subset),
        })
    return rows


def _optional_mean(values) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None

