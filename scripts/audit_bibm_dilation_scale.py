"""Quantify the 5-pixel C4 dilation scale for the BIBM manuscript."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conditions import load_mask
from src.config import MASK_CONFIG, RESULTS_DIR


DATASETS = ["BUS-BRA", "BUS-UCLM"]


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "q1": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "q3": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def audit_dataset(dataset: str) -> dict[str, Any]:
    manifest = pd.read_csv(ROOT / "manifest.csv")
    manifest = manifest[manifest["dataset_name"].eq(dataset)].copy()
    kernel = int(MASK_CONFIG["dilation_kernel"])
    radius = kernel // 2

    rows: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        mask = (load_mask(Path(row.mask_path)) > 0).astype(np.uint8)
        dilated_mask = (load_mask(Path(row.dilated_mask_path)) > 0).astype(np.uint8)
        width, height = image_size(row.image_path)
        short_edge = float(min(width, height))
        original_area = float(mask.sum())
        dilated_area = float(dilated_mask.sum())
        if original_area <= 0:
            continue
        equivalent_diameter = float(np.sqrt(4.0 * original_area / np.pi))
        rows.append(
            {
                "image_path": str(row.image_path),
                "patient_id": str(row.patient_id),
                "width": int(width),
                "height": int(height),
                "short_edge": short_edge,
                "original_mask_area": original_area,
                "dilated_mask_area": dilated_area,
                "area_increase_fraction": float((dilated_area - original_area) / original_area),
                "kernel_fraction_of_short_edge": float(kernel / short_edge),
                "radius_fraction_of_short_edge": float(radius / short_edge),
                "kernel_fraction_of_equivalent_lesion_diameter": float(kernel / equivalent_diameter),
                "radius_fraction_of_equivalent_lesion_diameter": float(radius / equivalent_diameter),
            }
        )

    frame = pd.DataFrame(rows)
    return {
        "dataset": dataset,
        "n_images": int(len(frame)),
        "dilation_kernel_pixels": kernel,
        "approx_radius_pixels": radius,
        "image_width": summarize(frame["width"].tolist()),
        "image_height": summarize(frame["height"].tolist()),
        "short_edge": summarize(frame["short_edge"].tolist()),
        "area_increase_fraction": summarize(frame["area_increase_fraction"].tolist()),
        "kernel_fraction_of_short_edge": summarize(frame["kernel_fraction_of_short_edge"].tolist()),
        "radius_fraction_of_short_edge": summarize(frame["radius_fraction_of_short_edge"].tolist()),
        "kernel_fraction_of_equivalent_lesion_diameter": summarize(
            frame["kernel_fraction_of_equivalent_lesion_diameter"].tolist()
        ),
        "radius_fraction_of_equivalent_lesion_diameter": summarize(
            frame["radius_fraction_of_equivalent_lesion_diameter"].tolist()
        ),
    }


def fmt_summary(stats: dict[str, float], percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return (
        f"{stats['median'] * scale:.2f}{suffix} "
        f"[{stats['q1'] * scale:.2f}{suffix}, {stats['q3'] * scale:.2f}{suffix}]"
    )


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# BIBM C4 dilation-scale audit",
        "",
        "C4 uses the saved dilated lesion mask in the original image/mask space before encoder-specific resizing.",
        "",
        "| Dataset | Images | Width median [IQR] | Height median [IQR] | Area increase median [IQR] | Kernel/short edge median [IQR] | Radius/lesion diameter median [IQR] |",
        "| --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for dataset, stats in report["datasets"].items():
        lines.append(
            f"| {dataset} | {stats['n_images']} | "
            f"{fmt_summary(stats['image_width'])} | "
            f"{fmt_summary(stats['image_height'])} | "
            f"{fmt_summary(stats['area_increase_fraction'], percent=True)} | "
            f"{fmt_summary(stats['kernel_fraction_of_short_edge'], percent=True)} | "
            f"{fmt_summary(stats['radius_fraction_of_equivalent_lesion_diameter'], percent=True)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    report = {
        "source": "BIBM final closure dilation-scale audit",
        "note": "Dilation is applied in original mask space before encoder resizing.",
        "datasets": {dataset: audit_dataset(dataset) for dataset in DATASETS},
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / "bibm_dilation_scale_audit.json"
    md_path = RESULTS_DIR / "bibm_dilation_scale_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
