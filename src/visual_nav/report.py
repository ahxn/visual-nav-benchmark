from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .benchmark import EpisodeResult, summarize
from .statistics import paired_bootstrap_comparisons, paired_recovery_comparisons


def load_results(paths: list[Path]) -> list[EpisodeResult]:
    unique: dict[tuple[str | None, str | None, str, int], EpisodeResult] = {}
    for path in paths:
        for row in json.loads(path.read_text(encoding="utf-8")):
            result = EpisodeResult(**row)
            unique[(result.scene, result.recovery_policy, result.method, result.seed)] = result
    return list(unique.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine visual-navigation episode result files")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/combined_summary.csv"))
    args = parser.parse_args()
    rows = summarize(load_results(args.inputs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparisons = paired_bootstrap_comparisons(load_results(args.inputs))
    if comparisons:
        comparison_path = args.output.with_name(f"{args.output.stem}.paired_ci.csv")
        with comparison_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
    recovery_comparisons = paired_recovery_comparisons(load_results(args.inputs))
    if recovery_comparisons:
        recovery_path = args.output.with_name(f"{args.output.stem}.recovery_ci.csv")
        with recovery_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(recovery_comparisons[0]))
            writer.writeheader()
            writer.writerows(recovery_comparisons)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
