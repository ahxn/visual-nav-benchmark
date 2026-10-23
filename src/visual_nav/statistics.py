from __future__ import annotations

import math
import random
from itertools import combinations

from .benchmark import EpisodeResult


DEFAULT_METRICS = (
    "success",
    "collisions",
    "path_length",
    "spl",
    "occupancy_iou",
    "chamfer_distance",
    "depth_abs_rel",
    "depth_rmse",
    "depth_coverage",
)


def _percentile(sorted_values: list[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def paired_bootstrap_comparisons(
    results: list[EpisodeResult],
    iterations: int = 5000,
    random_seed: int = 0,
) -> list[dict[str, object]]:
    """Return paired mean differences with percentile bootstrap 95% CIs.

    Episodes are paired by scene, seed, and recovery policy. Differences are
    always method B minus method A, keeping the sign convention explicit.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    methods = sorted({result.method for result in results})
    indexed = {
        (result.scene, result.seed, result.recovery_policy, result.method): result
        for result in results
    }
    pair_keys = sorted(
        {(result.scene or "", result.seed, result.recovery_policy or "") for result in results}
    )
    rng = random.Random(random_seed)
    rows: list[dict[str, object]] = []
    for method_a, method_b in combinations(methods, 2):
        policies = sorted({policy for _, _, policy in pair_keys})
        for policy in policies:
            pairs = [
                (indexed[(scene or None, seed, policy or None, method_a)],
                 indexed[(scene or None, seed, policy or None, method_b)])
                for scene, seed, pair_policy in pair_keys
                if pair_policy == policy
                and (scene or None, seed, policy or None, method_a) in indexed
                and (scene or None, seed, policy or None, method_b) in indexed
            ]
            for metric in DEFAULT_METRICS:
                differences = []
                for result_a, result_b in pairs:
                    raw_a, raw_b = getattr(result_a, metric), getattr(result_b, metric)
                    value_a = float(raw_a) if raw_a is not None else math.nan
                    value_b = float(raw_b) if raw_b is not None else math.nan
                    if math.isfinite(value_a) and math.isfinite(value_b):
                        differences.append(value_b - value_a)
                if not differences:
                    continue
                bootstrapped = sorted(
                    sum(rng.choice(differences) for _ in differences) / len(differences)
                    for _ in range(iterations)
                )
                rows.append({
                    "recovery_policy": policy or None,
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": metric,
                    "paired_episodes": len(differences),
                    "mean_difference_b_minus_a": sum(differences) / len(differences),
                    "ci95_low": _percentile(bootstrapped, 0.025),
                    "ci95_high": _percentile(bootstrapped, 0.975),
                })
    return rows


def paired_recovery_comparisons(
    results: list[EpisodeResult],
    iterations: int = 5000,
    random_seed: int = 1,
) -> list[dict[str, object]]:
    """Compare collision-only recovery against disabled recovery per method."""
    indexed = {
        (result.scene, result.seed, result.method, result.recovery_policy): result
        for result in results
    }
    rng = random.Random(random_seed)
    rows: list[dict[str, object]] = []
    for method in sorted({result.method for result in results}):
        pairs = []
        for scene, seed in sorted({(result.scene or "", result.seed) for result in results}):
            disabled_key = (scene or None, seed, method, "disabled")
            recovered_key = (scene or None, seed, method, "collision_only")
            if disabled_key in indexed and recovered_key in indexed:
                pairs.append((indexed[disabled_key], indexed[recovered_key]))
        for metric in DEFAULT_METRICS:
            differences = []
            for disabled, recovered in pairs:
                raw_disabled, raw_recovered = getattr(disabled, metric), getattr(recovered, metric)
                if raw_disabled is None or raw_recovered is None:
                    continue
                difference = float(raw_recovered) - float(raw_disabled)
                if math.isfinite(difference):
                    differences.append(difference)
            if not differences:
                continue
            bootstrapped = sorted(
                sum(rng.choice(differences) for _ in differences) / len(differences)
                for _ in range(iterations)
            )
            rows.append({
                "method": method,
                "metric": metric,
                "paired_episodes": len(differences),
                "mean_difference_collision_only_minus_disabled": sum(differences) / len(differences),
                "ci95_low": _percentile(bootstrapped, 0.025),
                "ci95_high": _percentile(bootstrapped, 0.975),
            })
    return rows
