from __future__ import annotations

import math

from .grid import Cell


def occupancy_iou(predicted: set[Cell], truth: set[Cell]) -> float:
    union = predicted | truth
    return len(predicted & truth) / len(union) if union else 1.0


def chamfer_distance(a: set[Cell], b: set[Cell]) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return math.inf

    def directed(source: set[Cell], target: set[Cell]) -> float:
        return sum(min(math.dist(p, q) for q in target) for p in source) / len(source)

    return 0.5 * (directed(a, b) + directed(b, a))


def spl(success: bool, shortest: int, actual: int) -> float:
    if not success:
        return 0.0
    return shortest / max(shortest, actual, 1)


def depth_errors(predicted, truth) -> tuple[float, float, float]:
    """Return absolute-relative error, RMSE in meters, and valid coverage."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install the 'vision' optional dependencies") from exc
    prediction = np.asarray(predicted, dtype=np.float32)
    reference = np.asarray(truth, dtype=np.float32)
    if prediction.shape != reference.shape:
        raise ValueError("predicted and reference depth must have identical shapes")
    truth_valid = np.isfinite(reference) & (reference > 0)
    valid = truth_valid & np.isfinite(prediction) & (prediction > 0)
    coverage = float(valid.sum() / max(int(truth_valid.sum()), 1))
    if not np.any(valid):
        return math.inf, math.inf, coverage
    residual = prediction[valid] - reference[valid]
    abs_rel = float(np.mean(np.abs(residual) / reference[valid]))
    rmse = float(np.sqrt(np.mean(residual**2)))
    return abs_rel, rmse, coverage

