"""BUS-UCLM patient-label and pooling sensitivity for the BIBM manuscript.

The main probe is trained on image-level lesion labels and reported at the
patient level with an any-malignant target. BUS-UCLM contains a small set of
mixed-label patients, so this script quantifies whether the main DINOv3
direction depends on those patients or on mean patient pooling.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.statistics import bootstrap_auroc_ci, load_predictions, permutation_test


RESULTS_DIR = ROOT / "results"
MANIFEST_PATH = ROOT / "manifest.csv"
MODEL = "DINOv3"
DATASET = "BUS-UCLM"
CONDITIONS = [
    "C1_full",
    "C2_lesion-only",
    "C4_lesion-removed-inpaint",
    "C5d_dilated-mask-only",
    "C6d_dilated-geometry-rendering",
    "explicit_dilated_geometry",
]
DISPLAY_NAMES = {
    "C1_full": "Full image (C1)",
    "C2_lesion-only": "Lesion-only (C2)",
    "C4_lesion-removed-inpaint": "Lesion-removed (C4)",
    "C5d_dilated-mask-only": "Dilated mask-only (C5d)",
    "C6d_dilated-geometry-rendering": "Dilated geometry rendering (C6d)",
    "explicit_dilated_geometry": "Explicit dilated geometry",
}


def patient_categories(manifest: pd.DataFrame) -> tuple[dict[str, str], dict[str, int]]:
    categories: dict[str, str] = {}
    targets: dict[str, int] = {}
    for patient_id, group in manifest.groupby("patient_id", sort=True):
        labels = set(group["label"].astype(str))
        if labels == {"benign"}:
            category = "benign_only"
        elif labels == {"malignant"}:
            category = "malignant_only"
        else:
            category = "mixed"
        categories[str(patient_id)] = category
        targets[str(patient_id)] = int("malignant" in labels)
    return categories, targets


def image_seed_average(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(["image_path", "patient_id"], as_index=False)
        .agg(y_true=("y_true", "max"), y_prob=("y_prob", "mean"))
        .sort_values(["patient_id", "image_path"], kind="stable")
        .reset_index(drop=True)
    )


def noisy_or(values: pd.Series) -> float:
    clipped = np.clip(values.to_numpy(dtype=float), 0.0, 1.0)
    return float(1.0 - np.prod(1.0 - clipped))


POOLING: dict[str, Callable[[pd.Series], float] | str] = {
    "mean": "mean",
    "max": "max",
    "noisy_or": noisy_or,
}


def aggregate_patient_predictions(
    predictions: pd.DataFrame,
    categories: dict[str, str],
    targets: dict[str, int],
    pooling: str,
    exclude_mixed: bool,
) -> pd.DataFrame:
    averaged = image_seed_average(predictions)
    if exclude_mixed:
        averaged = averaged[
            averaged["patient_id"].astype(str).map(categories).ne("mixed")
        ].copy()
    reducer = POOLING[pooling]
    patient = (
        averaged.groupby("patient_id", as_index=False)
        .agg(y_prob=("y_prob", reducer))
        .sort_values("patient_id", kind="stable")
        .reset_index(drop=True)
    )
    patient["patient_id"] = patient["patient_id"].astype(str)
    patient["patient_category"] = patient["patient_id"].map(categories)
    patient["y_true"] = patient["patient_id"].map(targets).astype(int)
    return patient[["patient_id", "patient_category", "y_true", "y_prob"]]


def summarize_patient_frame(patient: pd.DataFrame) -> dict[str, float | int]:
    y_true = patient["y_true"].to_numpy(dtype=int)
    y_prob = patient["y_prob"].to_numpy(dtype=float)
    patient_ids = patient["patient_id"].astype(str).to_numpy()
    if len(np.unique(y_true)) < 2:
        return {
            "n_patients": int(len(patient)),
            "n_benign": int((y_true == 0).sum()),
            "n_malignant": int((y_true == 1).sum()),
            "auroc": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "p_value": float("nan"),
        }
    boot = bootstrap_auroc_ci(y_true, y_prob, patient_ids)
    perm = permutation_test(y_true, y_prob, patient_ids)
    return {
        "n_patients": int(len(patient)),
        "n_benign": int((y_true == 0).sum()),
        "n_malignant": int((y_true == 1).sum()),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "ci_lower": float(boot["ci_lower"]),
        "ci_upper": float(boot["ci_upper"]),
        "p_value": float(perm["p_value"]),
    }


def load_condition_predictions(condition: str) -> pd.DataFrame:
    if condition == "explicit_dilated_geometry":
        pred_path = RESULTS_DIR / "bibm_dilated_geometry_feature_predictions.csv"
        if not pred_path.exists():
            return pd.DataFrame()
        predictions = pd.read_csv(pred_path)
        dataset_col = "geometry_dataset" if "geometry_dataset" in predictions.columns else "dataset"
        return predictions[predictions[dataset_col].eq(DATASET)].copy()
    return load_predictions(MODEL, DATASET, condition)


def format_metric(stats: dict[str, float | int]) -> str:
    return (
        f"{stats['auroc']:.3f} "
        f"[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]"
    )


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# BUS-UCLM patient-label sensitivity",
        "",
        "Probe training uses image-level lesion labels. Patient-level evaluation uses an any-malignant target.",
        "",
        "## Cohort accounting",
        "",
        "| Patients | Benign-only | Malignant-only | Mixed | Final benign target | Final malignant target |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {report['cohort']['n_patients']} | {report['cohort']['benign_only']} | "
            f"{report['cohort']['malignant_only']} | {report['cohort']['mixed']} | "
            f"{report['cohort']['final_benign_target']} | {report['cohort']['final_malignant_target']} |"
        ),
        "",
        "## DINOv3 BUS-UCLM sensitivity",
        "",
        "| Condition | Pooling | Mixed patients | Patients | AUROC [95% CI] |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for condition, variants in report["conditions"].items():
        for key, stats in variants.items():
            pooling, mixed_status = key.split("__")
            lines.append(
                f"| {DISPLAY_NAMES.get(condition, condition)} | {pooling} | {mixed_status.replace('_', ' ')} | "
                f"{stats['n_patients']} | {format_metric(stats)} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[manifest["dataset_name"].eq(DATASET)].copy()
    categories, targets = patient_categories(manifest)
    cohort = {
        "dataset": DATASET,
        "n_images": int(len(manifest)),
        "n_patients": int(len(categories)),
        "benign_only": int(sum(1 for v in categories.values() if v == "benign_only")),
        "malignant_only": int(sum(1 for v in categories.values() if v == "malignant_only")),
        "mixed": int(sum(1 for v in categories.values() if v == "mixed")),
        "final_benign_target": int(sum(1 for v in targets.values() if v == 0)),
        "final_malignant_target": int(sum(1 for v in targets.values() if v == 1)),
    }
    report: dict = {
        "source": "DINOv3 BUS-UCLM patient-label and pooling sensitivity",
        "cohort": cohort,
        "conditions": {},
    }
    for condition in CONDITIONS:
        predictions = load_condition_predictions(condition)
        if predictions.empty:
            continue
        condition_report: dict[str, dict[str, float | int]] = {}
        for pooling in POOLING:
            for exclude_mixed in (False, True):
                patient = aggregate_patient_predictions(
                    predictions,
                    categories,
                    targets,
                    pooling=pooling,
                    exclude_mixed=exclude_mixed,
                )
                key = f"{pooling}__{'mixed_excluded' if exclude_mixed else 'mixed_included'}"
                condition_report[key] = summarize_patient_frame(patient)
        report["conditions"][condition] = condition_report

    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / "bibm_patient_label_sensitivity.json"
    md_path = RESULTS_DIR / "bibm_patient_label_sensitivity.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
