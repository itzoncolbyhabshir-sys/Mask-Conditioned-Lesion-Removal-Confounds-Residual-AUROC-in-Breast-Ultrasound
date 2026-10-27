"""Run the fixed-shape lesion-removal controls reported in main-text Figure 3."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.refresh_bibm_fixed_shape_statistics import (
    C4_CONDITION, FIXED_CONDITIONS, endpoint_statistics, compare_conditions,
)
from src.conditions import inpaint_telea, load_image, load_mask
from src.config import IMAGES_DIR, RESULTS_DIR, SEEDS
from src.features import run_extraction
from src.probe import run_probe_experiment

def fixed_shape_mask(mask: np.ndarray, shape: str) -> np.ndarray:
    """Create a fixed shape nominally matching the input mask's area and center."""
    binary = (mask > 0).astype(np.uint8)
    area = int(binary.sum())
    output = np.zeros_like(binary)
    if area == 0:
        return output
    ys, xs = np.where(binary > 0)
    cx = int(round(float(xs.mean())))
    cy = int(round(float(ys.mean())))

    if shape == "circle":
        radius = max(1, int(round(math.sqrt(area / math.pi))))
        cv2.circle(output, (cx, cy), radius, 1, thickness=-1)
    elif shape == "square":
        side = max(2, int(round(math.sqrt(area))))
        half = side // 2
        cv2.rectangle(output, (cx - half, cy - half), (cx + half, cy + half), 1, -1)
    elif shape == "ellipse":
        # Fixed 2:1 aspect ratio; axes are selected to match the mask area.
        minor = max(1, int(round(math.sqrt(area / (2 * math.pi)))))
        major = max(2, 2 * minor)
        cv2.ellipse(output, (cx, cy), (major, minor), 0, 0, 360, 1, -1)
    else:
        raise ValueError(f"Unknown fixed shape: {shape}")
    return output

def main(model: str = "DINOv3", dataset: str = "BUS-BRA", force_features: bool = False) -> dict:
    manifest = pd.read_csv(ROOT / "manifest.csv")
    manifest = manifest[manifest["dataset_name"] == dataset].copy().reset_index(drop=True)
    if manifest.empty:
        raise ValueError(f"No manifest records for {dataset}")
    fixed_dirs = {
        shape: IMAGES_DIR / dataset / condition
        for shape, condition in FIXED_CONDITIONS.items()
    }
    for directory in fixed_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    used_names = set()
    for row in manifest.itertuples(index=False):
        image_path = Path(row.image_path)
        image = load_image(image_path, grayscale=True)
        target_mask = load_mask(Path(row.dilated_mask_path))
        output_name = f"{image_path.stem}.png"
        if output_name in used_names:
            raise ValueError(f"Duplicate condition image name: {output_name}")
        used_names.add(output_name)
        for shape, directory in fixed_dirs.items():
            intervention_mask = fixed_shape_mask(target_mask, shape)
            Image.fromarray(inpaint_telea(image, intervention_mask)).save(directory / output_name)

    fixed_results = {}
    for shape, condition in FIXED_CONDITIONS.items():
        run_extraction(model, dataset, condition, batch_size=4, force=force_features)
        run_probe_experiment(model, dataset, condition, seeds=SEEDS)
        fixed_results[shape] = {
            "condition": condition,
            "endpoint": endpoint_statistics(model, dataset, condition),
            "c4_minus_fixed": compare_conditions(model, dataset, C4_CONDITION, condition),
        }
    report = {
        "model": model,
        "dataset": dataset,
        "n_images": len(manifest),
        "n_target_patients": int(manifest["patient_id"].nunique()),
        "fixed_shape_removal": fixed_results,
        "fixed_shape_statistics_note": "Stratified paired patient bootstrap; fixed shapes jointly change shape and lesion coverage.",
    }
    suffix_parts = []
    if model != "DINOv3":
        suffix_parts.append(model.lower().replace("-", "_"))
    if dataset != "BUS-BRA":
        suffix_parts.append(dataset.lower().replace("-", "_"))
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"fixed_shape_controls{suffix}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["DINOv3", "MedSAM2", "UltraSam"], default="DINOv3")
    parser.add_argument("--dataset", choices=["BUS-BRA", "BUS-UCLM"], default="BUS-BRA")
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()
    main(args.model, args.dataset, args.force_features)
