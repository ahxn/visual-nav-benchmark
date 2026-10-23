from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize_visual_odometry(paths: list[Path]) -> tuple[dict[str, object], list[dict[str, object]]]:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    steps = [
        {"scene": run["scene"], "seed": run["seed"], **step}
        for run in runs for step in run["steps"]
    ]
    valid = [step for step in steps if step["translation_direction_error_deg"] is not None]
    summary = {
        "runs": len(runs),
        "pairs_attempted": len(steps),
        "poses_recovered": len(valid),
        "pose_recovery_rate": len(valid) / len(steps) if steps else 0.0,
        "mean_matches": sum(step["matches"] for step in steps) / len(steps) if steps else 0.0,
        "mean_inlier_ratio": sum(step["inlier_ratio"] for step in steps) / len(steps) if steps else 0.0,
        "mean_translation_direction_error_deg": (
            sum(step["translation_direction_error_deg"] for step in valid) / len(valid)
            if valid else None
        ),
        "mean_rotation_error_deg": (
            sum(step["rotation_error_deg"] for step in valid) / len(valid)
            if valid else None
        ),
    }
    by_scene = []
    for run in runs:
        by_scene.append({key: value for key, value in run.items() if key != "steps"})
    return summary, by_scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine visual-odometry result files")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/visual_odometry_summary.json"))
    args = parser.parse_args()
    summary, by_scene = summarize_visual_odometry(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".by_scene.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(by_scene[0]))
        writer.writeheader()
        writer.writerows(by_scene)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
