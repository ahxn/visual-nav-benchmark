from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .benchmark import run_episode, summarize
from .perception import METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired visual-navigation benchmark episodes")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=sorted(METHODS))
    parser.add_argument("--output", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    results = [
        run_episode(METHODS[name](), seed)
        for seed in range(args.seed, args.seed + args.episodes)
        for name in args.methods
    ]
    summary = summarize(results)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "episodes.json").write_text(
        json.dumps([result.to_dict() for result in results], indent=2), encoding="utf-8"
    )
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(
            f"{row['method']:12} success={row['success_rate']:.1%} "
            f"SPL={row['mean_spl']:.3f} collisions={row['mean_collisions']:.2f} "
            f"IoU={row['mean_occupancy_iou']:.3f} chamfer={row['mean_chamfer']:.3f}"
        )

