from __future__ import annotations

import argparse
import csv
from pathlib import Path


METHOD_LABELS = {
    "simulator_depth": "Simulator",
    "classical_sparse_features": "Sparse SIFT",
    "classical_stereo": "Classical stereo",
    "learned_metric_depth": "Learned depth",
}
METHOD_COLORS = {
    "simulator_depth": (90, 110, 130),
    "classical_sparse_features": (46, 158, 104),
    "classical_stereo": (42, 112, 205),
    "learned_metric_depth": (230, 126, 34),
}


def _draw_panel(draw, box, rows, metric, title, maximum, higher_is_better, font, small_font):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=14, fill=(250, 250, 252), outline=(215, 218, 224), width=2)
    draw.text((x0 + 20, y0 + 15), title, fill=(25, 28, 35), font=font)
    direction = "higher is better" if higher_is_better else "lower is better"
    draw.text((x0 + 20, y0 + 48), direction, fill=(95, 100, 110), font=small_font)
    chart_top, chart_bottom = y0 + 85, y1 - 70
    chart_left, chart_right = x0 + 58, x1 - 24
    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=(150, 155, 165), width=2)
    draw.line((chart_left, chart_top, chart_left, chart_bottom), fill=(150, 155, 165), width=2)
    draw.text((x0 + 10, chart_top - 8), f"{maximum:g}", fill=(90, 95, 105), font=small_font)
    draw.text((x0 + 30, chart_bottom - 8), "0", fill=(90, 95, 105), font=small_font)
    slot = (chart_right - chart_left) / len(rows)
    bar_width = slot * 0.52
    for index, row in enumerate(rows):
        value = float(row[metric])
        height = min(value / maximum, 1.0) * (chart_bottom - chart_top)
        center = chart_left + slot * (index + 0.5)
        left, right = center - bar_width / 2, center + bar_width / 2
        top = chart_bottom - height
        color = METHOD_COLORS[row["method"]]
        draw.rounded_rectangle((left, top, right, chart_bottom), radius=6, fill=color)
        value_text = f"{value:.3f}" if value < 1 else f"{value:.2f}"
        value_box = draw.textbbox((0, 0), value_text, font=small_font)
        draw.text(
            (center - (value_box[2] - value_box[0]) / 2, max(chart_top, top - 24)),
            value_text, fill=(35, 38, 45), font=small_font,
        )
        label = METHOD_LABELS[row["method"]]
        if " " in label:
            first, second = label.split(" ", 1)
            label = f"{first}\n{second}"
        label_box = draw.multiline_textbbox((0, 0), label, font=small_font, align="center")
        draw.multiline_text(
            (center - (label_box[2] - label_box[0]) / 2, chart_bottom + 10),
            label, fill=(55, 60, 70), font=small_font, align="center", spacing=1,
        )


def create_result_figure(summary_csv: Path, output: Path, coverage_panel: bool = False) -> None:
    from PIL import Image, ImageDraw, ImageFont

    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_method = {row["method"]: row for row in rows}
    ordered = [by_method[name] for name in METHOD_LABELS if name in by_method]
    if len(ordered) < 3:
        raise ValueError("summary must contain at least three recognized methods")

    image = Image.new("RGB", (1400, 940), (238, 241, 246))
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=30)
    panel_font = ImageFont.load_default(size=24)
    small_font = ImageFont.load_default(size=17)
    draw.text((45, 28), "Held-out AI2-THOR benchmark: 20 paired episodes per method", fill=(20, 24, 32), font=title_font)
    draw.text((47, 70), "Five seeds in each of four room categories; collision-only recovery enabled", fill=(75, 80, 90), font=small_font)

    fourth_panel = (
        ((710, 520, 1365, 905), "mean_depth_coverage", "Valid depth coverage", 1.0, True)
        if coverage_panel else
        ((710, 520, 1365, 905), "mean_depth_abs_rel", "Depth absolute-relative error", 1.5, False)
    )
    panels = [
        ((35, 115, 690, 500), "mean_spl", "Navigation SPL", 1.0, True),
        ((710, 115, 1365, 500), "mean_occupancy_iou", "Occupancy IoU", 0.4, True),
        ((35, 520, 690, 905), "mean_collisions", "Collisions per episode", 5.0, False),
        fourth_panel,
    ]
    for box, metric, title, maximum, higher in panels:
        _draw_panel(draw, box, ordered, metric, title, maximum, higher, panel_font, small_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a benchmark summary figure")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark_summary.png"))
    parser.add_argument("--coverage-panel", action="store_true")
    args = parser.parse_args()
    create_result_figure(args.summary, args.output, args.coverage_panel)


if __name__ == "__main__":
    main()
