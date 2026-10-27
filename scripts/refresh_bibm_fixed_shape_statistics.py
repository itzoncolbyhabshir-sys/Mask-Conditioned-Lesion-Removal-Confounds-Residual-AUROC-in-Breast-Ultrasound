"""Refresh the main-text fixed-shape statistics from stored OOF predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.statistics import (  # noqa: E402
    aggregate_patient_predictions,
    bootstrap_auroc_ci,
    load_predictions,
    paired_bootstrap_difference,
)


C4_CONDITION = "C4_lesion-removed-inpaint"
FIXED_CONDITIONS = {
    "circle": "C10_fixed-circle-inpaint",
    "square": "C11_fixed-square-inpaint",
    "ellipse": "C12_fixed-ellipse-inpaint",
}
SETTINGS = [
    ("DINOv3", "BUS-BRA", "fixed_shape_controls.json"),
    ("MedSAM2", "BUS-BRA", "fixed_shape_controls_medsam2.json"),
    ("DINOv3", "BUS-UCLM", "fixed_shape_controls_bus_uclm.json"),
    ("UltraSam", "BUS-BRA", "fixed_shape_controls_ultrasam.json"),
]
RESULTS_DIR = ROOT / "results"


def patient_predictions(model: str, dataset: str, condition: str):
    predictions = load_predictions(model, dataset, condition)
    if predictions.empty:
        raise RuntimeError(f"Missing predictions for {model}/{dataset}/{condition}")
    return aggregate_patient_predictions(predictions)


def endpoint_statistics(model: str, dataset: str, condition: str) -> dict[str, Any]:
    patient = patient_predictions(model, dataset, condition)
    y_true = patient["y_true"].to_numpy(dtype=int)
    y_prob = patient["y_prob"].to_numpy(dtype=float)
    patient_ids = patient["patient_id"].astype(str).to_numpy()
    boot = bootstrap_auroc_ci(y_true, y_prob, patient_ids)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "bootstrap_mean": float(boot["auroc_mean"]),
        "ci_lower": float(boot["ci_lower"]),
        "ci_upper": float(boot["ci_upper"]),
        "permutation_p": None,
        "p_value_note": "not recomputed; manuscript figure uses AUROC and stratified patient bootstrap CI",
    }


def compare_conditions(model: str, dataset: str, first: str, second: str) -> dict[str, Any]:
    left = patient_predictions(model, dataset, first).rename(
        columns={"y_true": "y_true_left", "y_prob": "y_prob_left"}
    )
    right = patient_predictions(model, dataset, second).rename(
        columns={"y_true": "y_true_right", "y_prob": "y_prob_right"}
    )
    aligned = left.merge(right, on="patient_id", validate="one_to_one")
    if not np.array_equal(aligned["y_true_left"], aligned["y_true_right"]):
        raise RuntimeError(f"Label mismatch between {model}/{dataset}/{first} and {second}")
    paired = paired_bootstrap_difference(
        aligned["y_true_left"].to_numpy(dtype=int),
        aligned["y_prob_left"].to_numpy(dtype=float),
        aligned["y_prob_right"].to_numpy(dtype=float),
        aligned["patient_id"].astype(str).to_numpy(),
    )
    return {
        "delta_auroc": float(paired["delta_auroc"]),
        "observed_delta_auroc": float(paired["observed_delta_auroc"]),
        "ci_lower": float(paired["ci_lower"]),
        "ci_upper": float(paired["ci_upper"]),
        "p_value": float(paired["p_value"]),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for model, dataset, filename in SETTINGS:
        path = RESULTS_DIR / filename
        report = {"model": model, "dataset": dataset}
        fixed_results: dict[str, Any] = {}
        for shape, condition in FIXED_CONDITIONS.items():
            fixed_results[shape] = {
                "condition": condition,
                "endpoint": endpoint_statistics(model, dataset, condition),
                "c4_minus_fixed": compare_conditions(model, dataset, C4_CONDITION, condition),
            }
        report["model"] = model
        report["dataset"] = dataset
        report["fixed_shape_removal"] = fixed_results
        report["fixed_shape_statistics_note"] = (
            "Fixed-shape statistics refreshed from prediction CSVs using stratified patient bootstrap."
        )
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Refreshed {path}")


if __name__ == "__main__":
    main()
