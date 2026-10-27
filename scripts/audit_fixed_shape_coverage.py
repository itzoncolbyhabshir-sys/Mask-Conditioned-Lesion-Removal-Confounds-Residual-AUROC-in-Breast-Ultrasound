"""Audit fixed-shape lesion coverage for the main-text shape-and-coverage analysis."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fixed_shape_controls import fixed_shape_mask
from src.conditions import load_mask

RESULTS_DIR = ROOT / "results"
MANIFEST_PATH = ROOT / "manifest.csv"

def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
        "p05": float(np.nanpercentile(arr, 5)),
        "p25": float(np.nanpercentile(arr, 25)),
        "p75": float(np.nanpercentile(arr, 75)),
        "p95": float(np.nanpercentile(arr, 95)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "perfect_rate": float(np.mean(arr >= 0.999)),
        "near_complete_rate": float(np.mean(arr >= 0.95)),
        "below_0_90_rate": float(np.mean(arr < 0.90)),
    }


def audit_fixed_shape_coverage(dataset: str) -> dict[str, Any]:
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[manifest["dataset_name"] == dataset].copy()
    rows: list[dict[str, float | str]] = []
    for row in manifest.itertuples(index=False):
        lesion_mask = load_mask(Path(row.mask_path))
        dilated_mask = load_mask(Path(row.dilated_mask_path))
        lesion_mask = (lesion_mask > 0).astype(np.uint8)
        dilated_mask = (dilated_mask > 0).astype(np.uint8)
        for shape in ("circle", "square", "ellipse"):
            fixed = fixed_shape_mask(dilated_mask, shape).astype(np.uint8)
            lesion_den = max(1, int(lesion_mask.sum()))
            dilated_den = max(1, int(dilated_mask.sum()))
            rows.append(
                {
                    "dataset": dataset,
                    "image_path": str(row.image_path),
                    "patient_id": str(row.patient_id),
                    "shape": shape,
                    "lesion_coverage": float(((fixed > 0) & (lesion_mask > 0)).sum() / lesion_den),
                    "dilated_coverage": float(((fixed > 0) & (dilated_mask > 0)).sum() / dilated_den),
                    "fixed_area_ratio_vs_dilated": float(fixed.sum() / dilated_den),
                }
            )
    frame = pd.DataFrame(rows)
    out: dict[str, Any] = {"dataset": dataset, "n_images": int(len(manifest)), "shapes": {}}
    for shape, group in frame.groupby("shape"):
        out["shapes"][shape] = {
            "lesion_coverage": summarize(group["lesion_coverage"].tolist()),
            "dilated_coverage": summarize(group["dilated_coverage"].tolist()),
            "fixed_area_ratio_vs_dilated": summarize(group["fixed_area_ratio_vs_dilated"].tolist()),
        }
    return out


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Fixed-shape lesion coverage", "",
        "Fixed shapes approximately match the dilated-mask area and center before rasterization. Coverage below 1 means the edit leaves some lesion pixels visible.", "",
    ]
    for item in report["fixed_shape_coverage"]:
        lines.extend([f"## {item['dataset']}", "", "| Shape | Median lesion coverage | IQR | Coverage <0.90 |", "|---|---:|---:|---:|"])
        for shape, stats in item["shapes"].items():
            value = stats["lesion_coverage"]
            lines.append(f"| {shape} | {value['median']:.3f} | {value['p25']:.3f}–{value['p75']:.3f} | {value['below_0_90_rate']:.3f} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict[str, Any]:
    report = {"fixed_shape_coverage": [audit_fixed_shape_coverage(dataset) for dataset in ("BUS-BRA", "BUS-UCLM")]}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "fixed_shape_coverage.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, RESULTS_DIR / "fixed_shape_coverage.md")
    print(f"Wrote {path}")
    return report


if __name__ == "__main__":
    main()
