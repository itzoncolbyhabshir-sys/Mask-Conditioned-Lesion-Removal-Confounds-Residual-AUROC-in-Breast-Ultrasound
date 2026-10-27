"""Audit encoder preprocessing and crop coverage for the BIBM manuscript."""

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

from src.config import RESULTS_DIR  # noqa: E402
from src.conditions import load_mask  # noqa: E402


DATASETS = ["BUS-BRA", "BUS-UCLM"]
PREPROCESSING = [
    {
        "encoder": "ResNet-50",
        "input_geometry": "RGB conversion from grayscale-compatible source; custom Resize short side to 224",
        "crop_padding": "CenterCrop 224 x 224",
        "interpolation": "bilinear image resize; ImageNet normalization",
        "feature": "global average pooled penultimate feature",
        "feature_dim": 2048,
        "crop_audit": "resize_short_center_crop_224",
    },
    {
        "encoder": "DINOv3",
        "input_geometry": "grayscale-to-3-channel, timm resolved eval transform, input 256 x 256",
        "crop_padding": "Resize short side to 256 and CenterCrop 256 x 256 (crop_pct=1.0)",
        "interpolation": "bicubic image resize with antialias; ImageNet normalization",
        "feature": "CLS token",
        "feature_dim": 1024,
        "crop_audit": "resize_short_center_crop_256",
    },
    {
        "encoder": "MedSAM2",
        "input_geometry": "RGB conversion and direct Resize 512 x 512",
        "crop_padding": "no crop; aspect ratio is stretched to square",
        "interpolation": "bilinear image resize; masks/conditions use nearest neighbor where mask semantics are required",
        "feature": "spatial mean of image_encoder(images)[\"vision_features\"]",
        "feature_dim": 256,
        "crop_audit": "no_crop",
    },
    {
        "encoder": "UltraSam",
        "input_geometry": "RGB conversion, keep-ratio resize to 1024 long side",
        "crop_padding": "zero padding on 1024 x 1024 canvas",
        "interpolation": "bilinear image resize; ultrasound checkpoint normalization",
        "feature": "mean pooled encoder tokens",
        "feature_dim": 768,
        "crop_audit": "no_crop",
    },
]


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def center_crop_window_original(w: int, h: int, target: int) -> tuple[float, float, float, float]:
    """Return the original-coordinate crop window induced by Resize(short=target)+CenterCrop(target)."""
    short = min(w, h)
    if short <= 0:
        return 0.0, 0.0, float(w), float(h)
    scale = target / float(short)
    resized_w = w * scale
    resized_h = h * scale
    left_resized = max(0.0, (resized_w - target) / 2.0)
    top_resized = max(0.0, (resized_h - target) / 2.0)
    x0 = left_resized / scale
    y0 = top_resized / scale
    x1 = (left_resized + target) / scale
    y1 = (top_resized + target) / scale
    return x0, y0, min(float(w), x1), min(float(h), y1)


def mask_crop_coverage(mask_path: str | Path, target: int | None) -> float:
    mask = load_mask(Path(mask_path)) > 0
    if mask.sum() == 0:
        return 1.0
    if target is None:
        return 1.0
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = center_crop_window_original(w, h, target)
    ys, xs = np.nonzero(mask)
    inside = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
    return float(inside.mean())


def describe(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "q1": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "q3": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "below_0_99_rate": float(np.mean(arr < 0.99)),
        "below_0_95_rate": float(np.mean(arr < 0.95)),
        "below_0_90_rate": float(np.mean(arr < 0.90)),
    }


def audit_dataset(dataset: str) -> dict[str, Any]:
    manifest = pd.read_csv(ROOT / "manifest.csv")
    manifest = manifest[manifest["dataset_name"].eq(dataset)].copy()
    widths: list[int] = []
    heights: list[int] = []
    original_coverages = {224: [], 256: []}
    dilated_coverages = {224: [], 256: []}

    for row in manifest.itertuples(index=False):
        w, h = image_size(row.image_path)
        widths.append(int(w))
        heights.append(int(h))
        for target in (224, 256):
            original_coverages[target].append(mask_crop_coverage(row.mask_path, target))
            dilated_coverages[target].append(mask_crop_coverage(row.dilated_mask_path, target))

    return {
        "n_images": int(len(manifest)),
        "n_patients": int(manifest["patient_id"].nunique()),
        "image_width": {
            "min": int(np.min(widths)),
            "median": float(np.median(widths)),
            "max": int(np.max(widths)),
        },
        "image_height": {
            "min": int(np.min(heights)),
            "median": float(np.median(heights)),
            "max": int(np.max(heights)),
        },
        "short_edge": {
            "min": int(np.min(np.minimum(widths, heights))),
            "median": float(np.median(np.minimum(widths, heights))),
            "max": int(np.max(np.minimum(widths, heights))),
        },
        "crop_coverage": {
            "ResNet-50_original_mask": describe(original_coverages[224]),
            "ResNet-50_dilated_mask": describe(dilated_coverages[224]),
            "DINOv3_original_mask": describe(original_coverages[256]),
            "DINOv3_dilated_mask": describe(dilated_coverages[256]),
            "MedSAM2_original_mask": describe([1.0] * len(manifest)),
            "MedSAM2_dilated_mask": describe([1.0] * len(manifest)),
            "UltraSam_original_mask": describe([1.0] * len(manifest)),
            "UltraSam_dilated_mask": describe([1.0] * len(manifest)),
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# BIBM preprocessing and crop audit",
        "",
        "Encoder preprocessing was read from the project feature-extraction code and summarized for manuscript reproducibility. Mask resizing for condition generation uses nearest-neighbor interpolation when mask semantics are required.",
        "",
        "## Encoder preprocessing",
        "",
        "| Encoder | Input geometry | Crop/padding | Interpolation/normalization | Feature | Dim |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for row in report["encoders"]:
        lines.append(
            f"| {row['encoder']} | {row['input_geometry']} | {row['crop_padding']} | "
            f"{row['interpolation']} | {row['feature']} | {row['feature_dim']} |"
        )
    lines.extend(
        [
            "",
            "## Crop coverage for lesion masks",
            "",
            "| Dataset | Encoder/mask | Median | IQR | Min | <0.90 rate |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for dataset, stats in report["datasets"].items():
        for key, cov in stats["crop_coverage"].items():
            lines.append(
                f"| {dataset} | {key} | {cov['median']:.3f} | "
                f"{cov['q1']:.3f}--{cov['q3']:.3f} | {cov['min']:.3f} | "
                f"{cov['below_0_90_rate']:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    report = {
        "source": "BIBM preprocessing and crop audit",
        "note": "C4 dilation is applied in original image/mask space before encoder-specific resizing; binary masks use nearest-neighbor resizing when resizing is required.",
        "encoders": PREPROCESSING,
        "datasets": {dataset: audit_dataset(dataset) for dataset in DATASETS},
    }
    json_path = RESULTS_DIR / "bibm_preprocessing_audit.json"
    md_path = RESULTS_DIR / "bibm_preprocessing_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
