"""Interpretable mask-geometry baseline for the BIBM manuscript.

This script trains a simple logistic probe on hand-engineered lesion-mask
geometry using the same registered outer folds as the image-feature probes.
It is a low-cost audit check: if these explicit geometry features are
label-decodable, the high C5/C6 controls are easier to interpret.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import N_SPLITS, PROBE_CONFIG, SEEDS
from src.conditions import load_mask
from src.probe import train_probe
from src.statistics import (
    aggregate_patient_predictions,
    align_patient_predictions,
    bootstrap_auroc_ci,
    load_predictions,
    paired_bootstrap_difference,
    permutation_test,
)


RESULTS_DIR = ROOT / "results"
MANIFEST_PATH = ROOT / "manifest.csv"
DATASETS = ["BUS-BRA", "BUS-UCLM"]
REFERENCE_MODEL = "DINOv3"
REFERENCE_CONDITION = "C4_lesion-removed-inpaint"
FEATURE_NAMES = [
    "sqrt_normalized_area",
    "normalized_perimeter",
    "compactness",
    "eccentricity",
    "solidity",
    "bbox_aspect_ratio",
    "centroid_x",
    "centroid_y",
    "lesion_to_image_area_ratio",
]


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    return float(num / den) if den else default


def mask_geometry_features(mask_path: str | Path) -> list[float]:
    mask = (load_mask(Path(mask_path)) > 0).astype(np.uint8)
    h, w = mask.shape[:2]
    image_area = float(max(1, h * w))
    area = float(mask.sum())
    ys, xs = np.nonzero(mask)
    if area <= 0 or len(xs) == 0:
        return [0.0] * len(FEATURE_NAMES)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(contour, closed=True) for contour in contours))
    contour_area = float(sum(cv2.contourArea(contour) for contour in contours))
    if contour_area <= 0:
        contour_area = area
    hull_area = 0.0
    for contour in contours:
        if len(contour) >= 3:
            hull_area += float(cv2.contourArea(cv2.convexHull(contour)))
    if hull_area <= 0:
        hull_area = contour_area

    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    bbox_w = x1 - x0 + 1.0
    bbox_h = y1 - y0 + 1.0
    coords = np.column_stack([xs.astype(float), ys.astype(float)])
    if len(coords) >= 3:
        cov = np.cov(coords, rowvar=False)
        eigvals = np.sort(np.linalg.eigvalsh(cov))
        eccentricity = (
            math.sqrt(max(0.0, 1.0 - safe_div(eigvals[0], eigvals[-1], 1.0)))
            if eigvals[-1] > 0
            else 0.0
        )
    else:
        eccentricity = 0.0

    lesion_to_image_area_ratio = safe_div(area, image_area)
    return [
        math.sqrt(lesion_to_image_area_ratio),
        safe_div(perimeter, math.sqrt(image_area)),
        safe_div(4.0 * math.pi * contour_area, perimeter * perimeter),
        eccentricity,
        safe_div(contour_area, hull_area, 1.0),
        safe_div(bbox_w, bbox_h, 1.0),
        safe_div(float(xs.mean()), max(1.0, w - 1.0)),
        safe_div(float(ys.mean()), max(1.0, h - 1.0)),
        lesion_to_image_area_ratio,
    ]


def build_feature_table(dataset: str) -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[manifest["dataset_name"].eq(dataset)].copy()
    rows: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        features = mask_geometry_features(row.mask_path)
        out = {
            "image_path": str(row.image_path),
            "patient_id": str(row.patient_id),
            "label": str(row.label),
            "split_id": int(row.split_id),
        }
        out.update(dict(zip(FEATURE_NAMES, features)))
        rows.append(out)
    return pd.DataFrame(rows)


def evaluate_dataset(dataset: str) -> tuple[dict[str, Any], pd.DataFrame]:
    table = build_feature_table(dataset)
    X = table[FEATURE_NAMES].to_numpy(dtype=float)
    y = table["label"].eq("malignant").astype(int).to_numpy()
    groups = table["patient_id"].astype(str).to_numpy()
    splits = table["split_id"].astype(int).to_numpy()
    all_predictions: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []

    for fold in range(N_SPLITS):
        train_mask = splits != fold
        test_mask = splits == fold
        if not train_mask.any() or not test_mask.any():
            continue
        if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[test_mask])) < 2:
            continue
        for seed in SEEDS:
            clf, scaler = train_probe(
                X[train_mask],
                y[train_mask],
                groups[train_mask],
                C_grid=PROBE_CONFIG["C_grid"],
                inner_cv=PROBE_CONFIG["inner_cv"],
                class_weight=PROBE_CONFIG["class_weight"],
                max_iter=PROBE_CONFIG["max_iter"],
                random_state=seed,
            )
            y_prob = clf.predict_proba(scaler.transform(X[test_mask]))[:, 1]
            pred = table.loc[test_mask, ["image_path", "patient_id"]].copy()
            pred["y_true"] = y[test_mask]
            pred["y_prob"] = y_prob
            pred["fold"] = fold
            pred["seed"] = seed
            all_predictions.append(pred)
            patient = aggregate_patient_predictions(pred)
            if len(np.unique(patient["y_true"])) == 2:
                fold_summaries.append(
                    {
                        "fold": int(fold),
                        "seed": int(seed),
                        "n_test_patients": int(len(patient)),
                        "auroc": float(roc_auc_score(patient["y_true"], patient["y_prob"])),
                    }
                )

    predictions = pd.concat(all_predictions, ignore_index=True)
    patient_predictions = aggregate_patient_predictions(predictions)
    y_true = patient_predictions["y_true"].to_numpy(dtype=int)
    y_prob = patient_predictions["y_prob"].to_numpy(dtype=float)
    patient_ids = patient_predictions["patient_id"].astype(str).to_numpy()
    boot = bootstrap_auroc_ci(y_true, y_prob, patient_ids)
    perm = permutation_test(y_true, y_prob, patient_ids)

    reference = aggregate_patient_predictions(
        load_predictions(REFERENCE_MODEL, dataset, REFERENCE_CONDITION)
    )
    aligned = align_patient_predictions(
        patient_predictions,
        reference,
        first_prob_name="y_prob_geometry",
        second_prob_name="y_prob_c4",
    )
    paired = paired_bootstrap_difference(
        aligned["y_true"].to_numpy(dtype=int),
        aligned["y_prob_geometry"].to_numpy(dtype=float),
        aligned["y_prob_c4"].to_numpy(dtype=float),
        aligned["patient_id"].astype(str).to_numpy(),
    )

    summary = {
        "dataset": dataset,
        "model": "interpretable_mask_geometry_logistic_probe",
        "feature_names": FEATURE_NAMES,
        "n_images": int(len(table)),
        "n_patients": int(patient_predictions["patient_id"].nunique()),
        "n_benign_patients": int((patient_predictions["y_true"] == 0).sum()),
        "n_malignant_patients": int((patient_predictions["y_true"] == 1).sum()),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "ci_lower": float(boot["ci_lower"]),
        "ci_upper": float(boot["ci_upper"]),
        "p_value": float(perm["p_value"]),
        "delta_vs_dinov3_c4": {
            "convention": "geometry_feature_baseline_minus_C4",
            "delta_auroc": float(paired["delta_auroc"]),
            "ci_lower": float(paired["ci_lower"]),
            "ci_upper": float(paired["ci_upper"]),
            "p_value": float(paired["p_value"]),
        },
        "fold_seed_aurocs": fold_summaries,
    }
    predictions["condition"] = "mask_geometry_features"
    predictions["dataset"] = dataset
    return summary, predictions


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Interpretable mask-geometry baseline",
        "",
        "Logistic regression on explicit original-mask geometry features, using the same outer folds as the image-feature probes. The C4 dilated intervention mask is not used.",
        "",
        "| Dataset | Patients | AUROC [95% CI] | Delta vs DINOv3 C4 |",
        "| --- | ---: | --- | --- |",
    ]
    for dataset, stats in report["datasets"].items():
        delta = stats["delta_vs_dinov3_c4"]
        lines.append(
            f"| {dataset} | {stats['n_patients']} | "
            f"{stats['auroc']:.3f} [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}] | "
            f"{delta['delta_auroc']:+.3f} [{delta['ci_lower']:+.3f}, {delta['ci_upper']:+.3f}] |"
        )
    lines.extend(["", "## Features", "", ", ".join(FEATURE_NAMES)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summaries: dict[str, Any] = {}
    prediction_frames: list[pd.DataFrame] = []
    for dataset in DATASETS:
        summary, predictions = evaluate_dataset(dataset)
        summaries[dataset] = summary
        prediction_frames.append(predictions)

    report = {
        "source": "mask geometry logistic-probe baseline for BIBM final closure",
        "mask_source": "original lesion mask, not the C4 dilated intervention mask",
        "normalization": "area is normalized by image area, perimeter by sqrt(image area), and centroid coordinates by image width/height; standardization is fitted inside each training fold",
        "datasets": summaries,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / "bibm_geometry_feature_baseline.json"
    md_path = RESULTS_DIR / "bibm_geometry_feature_baseline.md"
    pred_path = RESULTS_DIR / "bibm_geometry_feature_baseline_predictions.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    pd.concat(prediction_frames, ignore_index=True).to_csv(pred_path, index=False)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {pred_path}")


if __name__ == "__main__":
    main()
