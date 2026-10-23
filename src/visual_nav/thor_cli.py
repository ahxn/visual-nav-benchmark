from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .ai2thor_env import AI2ThorEnvironment, ThorConfig
from .benchmark import summarize
from .statistics import paired_bootstrap_comparisons
from .thor_benchmark import run_thor_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI2-THOR depth navigation episodes")
    scene_group = parser.add_mutually_exclusive_group()
    scene_group.add_argument("--scene", help="single AI2-THOR scene (default: FloorPlan1)")
    scene_group.add_argument("--scenes", nargs="+", help="multiple AI2-THOR scenes")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--methods", nargs="+",
        choices=[
            "simulator_depth", "classical_stereo", "classical_sparse_features",
            "learned_metric_depth",
        ],
        default=["simulator_depth"],
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="inference device for learned_metric_depth (default: auto-detect)",
    )
    parser.add_argument("--mapper", choices=["pointcloud", "scanline"], default="pointcloud")
    parser.add_argument(
        "--disable-recovery", action="store_true",
        help="do not fall back to collision-only obstacles when visual mapping disconnects the grid",
    )
    parser.add_argument("--output", type=Path, default=Path("results/thor_episodes.json"))
    args = parser.parse_args()
    scenes = args.scenes or [args.scene or "FloorPlan1"]
    learned = None
    if "learned_metric_depth" in args.methods:
        from .learned_depth import LearnedMetricDepthEstimator

        device = None if args.device == "auto" else args.device
        learned = LearnedMetricDepthEstimator(device=device)
    results = []
    total = len(scenes) * args.episodes * len(args.methods)
    for scene in scenes:
        env = AI2ThorEnvironment(ThorConfig(scene=scene, headless=args.headless))
        try:
            for seed in range(args.seed, args.seed + args.episodes):
                for method in args.methods:
                    result = run_thor_episode(
                    env, seed, depth_source=method, learned_estimator=learned,
                    mapper_mode=args.mapper, collision_recovery=not args.disable_recovery,
                    )
                    results.append(result)
                    print(
                        f"[{len(results)}/{total}] {scene} seed={seed} {method}: "
                        f"success={result.success} spl={result.spl:.3f}",
                        file=sys.stderr,
                        flush=True,
                    )
        finally:
            env.stop()
    if learned is not None:
        print(
            f"learned-depth cache: {learned.cache_hits} hits, {learned.cache_misses} misses",
            file=sys.stderr,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8")
    summary = summarize(results)
    summary_path = args.output.with_suffix(".summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    by_scene = []
    for scene in scenes:
        for row in summarize([result for result in results if result.scene == scene]):
            by_scene.append({"scene": scene, **row})
    scene_path = args.output.with_suffix(".by_scene.csv")
    with scene_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(by_scene[0]))
        writer.writeheader()
        writer.writerows(by_scene)
    comparisons = paired_bootstrap_comparisons(results)
    if comparisons:
        comparison_path = args.output.with_suffix(".paired_ci.csv")
        with comparison_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
