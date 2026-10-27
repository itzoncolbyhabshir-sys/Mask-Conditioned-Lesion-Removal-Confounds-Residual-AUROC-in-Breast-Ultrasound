"""Patient-level inner-CV tuning sensitivity for primary BIBM controls.

The manuscript-facing probes select the logistic penalty with image-level AUROC
inside group-aware inner folds, then report patient-level AUROC after
aggregation. This lightweight sensitivity keeps the same outer folds and
endpoints, but selects ``C`` using patient-level AUROC inside each inner fold.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_bibm_dilated_mask_controls import build_dilated_geometry_feature_table  # noqa: E402
from run_bibm_geometry_feature_baseline import FEATURE_NAMES  # noqa: E402
from src.config import N_SPLITS, PROBE_CONFIG, RESULTS_DIR, SEEDS  # noqa: E402
from src.probe import run_probe_experiment, train_probe  # noqa: E402
from src.statistics import (  # noqa: E402
    aggregate_patient_predictions,
    align_patient_predictions,
    bootstrap_auroc_ci,
    load_predictions,
    paired_bootstrap_difference,
)


MODEL = "DINOv3"
DATASET = "BUS-BRA"
CONDITIONS = [
    "C4_lesion-removed-inpaint",
    "C5d_dilated-mask-only",
    "C6d_dilated-geometry-rendering",
]
DISPLAY_NAMES = {
    "C4_lesion-removed-inpaint": "Lesion-removed (C4)",
    "C5d_dilated-mask-only": "Dilated mask-only (C5d)",
    "C6d_dilated-geometry-rendering": "Dilated geometry rendering (C6d)",
    "dilated_mask_geometry_features": "Explicit dilated-mask geometry",
}
TAG_SUFFIX = "__patient-level-tuned"


def summarize_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    patient = aggregate_patient_predictions(predictions)
    y_true = patient["y_true"].to_numpy(dtype=int)
    y_prob = patient["y_prob"].to_numpy(dtype=float)
    patient_ids = patient["patient_id"].astype(str).to_numpy()
    boot = bootstrap_auroc_ci(y_true, y_prob, patient_ids)
    return {
        "n_patients": int(len(patient)),
        "n_benign_patients": int((patient["y_true"] == 0).sum()),
        "n_malignant_patients": int((patient["y_true"] == 1).sum()),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "ci_lower": float(boot["ci_lower"]),
        "ci_upper": float(boot["ci_upper"]),
        "patient_predictions": patient,
    }


def paired_delta(control: pd.DataFrame, c4: pd.DataFrame) -> dict[str, float]:
    aligned = align_patient_predictions(
        control,
        c4,
        first_prob_name="y_prob_control",
        second_prob_name="y_prob_c4",
    )
    paired = paired_bootstrap_difference(
        aligned["y_true"].to_numpy(dtype=int),
        aligned["y_prob_control"].to_numpy(dtype=float),
        aligned["y_prob_c4"].to_numpy(dtype=float),
        aligned["patient_id"].astype(str).to_numpy(),
    )
    return {
        "convention": "control_minus_patient_level_tuned_C4",
        "delta_auroc": float(paired["delta_auroc"]),
        "ci_lower": float(paired["ci_lower"]),
        "ci_upper": float(paired["ci_upper"]),
        "p_value": float(paired["p_value"]),
    }


def run_image_condition_probes() -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for condition in CONDITIONS:
        tag = f"{condition}{TAG_SUFFIX}"
        run_probe_experiment(
            model_name=MODEL,
            dataset_name=DATASET,
            condition=condition,
            seeds=SEEDS,
            result_tag=tag,
            inner_selection_metric="patient_auroc",
        )
        predictions = load_predictions(MODEL, DATASET, tag)
        if predictions.empty:
            raise RuntimeError(f"No predictions generated for {MODEL}/{DATASET}/{tag}")
        summary = summarize_predictions(predictions)
        summaries[condition] = {
            key: value for key, value in summary.items() if key != "patient_predictions"
        }
        summaries[condition]["_patient_predictions"] = summary["patient_predictions"]
    return summaries


def run_patient_tuned_dilated_geometry_probe(c4_patient_predictions: pd.DataFrame) -> dict[str, Any]:
    table = build_dilated_geometry_feature_table(DATASET)
    X = table[FEATURE_NAMES].to_numpy(dtype=float)
    y = table["label"].eq("malignant").astype(int).to_numpy()
    groups = table["patient_id"].astype(str).to_numpy()
    splits = table["split_id"].astype(int).to_numpy()

    all_predictions: list[pd.DataFrame] = []
    fold_metadata: list[dict[str, Any]] = []
    for fold in range(N_SPLITS):
        train_mask = splits != fold
        test_mask = splits == fold
        if not train_mask.any() or not test_mask.any():
            continue
        if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[test_mask])) < 2:
            continue
        for seed in SEEDS:
            clf, scaler, metadata = train_probe(
                X[train_mask],
                y[train_mask],
                groups[train_mask],
                C_grid=PROBE_CONFIG["C_grid"],
                inner_cv=PROBE_CONFIG["inner_cv"],
                class_weight=PROBE_CONFIG["class_weight"],
                max_iter=PROBE_CONFIG["max_iter"],
                random_state=seed,
                inner_selection_metric="patient_auroc",
                return_metadata=True,
            )
            y_prob = clf.predict_proba(scaler.transform(X[test_mask]))[:, 1]
            pred = table.loc[test_mask, ["image_path", "patient_id"]].copy()
            pred["y_true"] = y[test_mask]
            pred["y_prob"] = y_prob
            pred["fold"] = fold
            pred["seed"] = seed
            pred["condition"] = "dilated_mask_geometry_features"
            all_predictions.append(pred)
            fold_metadata.append({"fold": fold, "seed": seed, **metadata})

    if not all_predictions:
        raise RuntimeError("No patient-level-tuned dilated geometry predictions were produced")
    predictions = pd.concat(all_predictions, ignore_index=True)
    pred_path = RESULTS_DIR / "bibm_dilated_geometry_feature_patient_level_tuned_predictions.csv"
    predictions.to_csv(pred_path, index=False)
    summary = summarize_predictions(predictions)
    patient = summary["patient_predictions"]
    delta = paired_delta(patient, c4_patient_predictions)
    return {
        "n_patients": summary["n_patients"],
        "n_benign_patients": summary["n_benign_patients"],
        "n_malignant_patients": summary["n_malignant_patients"],
        "auroc": summary["auroc"],
        "ci_lower": summary["ci_lower"],
        "ci_upper": summary["ci_upper"],
        "delta_vs_patient_level_tuned_c4": delta,
        "feature_names": FEATURE_NAMES,
        "prediction_file": str(pred_path.relative_to(ROOT)),
        "fold_metadata": fold_metadata,
        "_patient_predictions": patient,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# BIBM patient-level inner-CV tuning sensitivity",
        "",
        "The logistic penalty was selected with patient-level AUROC inside group-aware inner folds. Outer folds, labels, and patient-level reporting match the main protocol.",
        "",
        "| Endpoint | Patients B/M | AUROC [95% CI] | Delta vs patient-tuned C4 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in [
        "C4_lesion-removed-inpaint",
        "C5d_dilated-mask-only",
        "C6d_dilated-geometry-rendering",
        "dilated_mask_geometry_features",
    ]:
        stats = report["endpoints"][key]
        if key == "C4_lesion-removed-inpaint":
            delta_text = "reference"
        else:
            delta = stats["delta_vs_patient_level_tuned_c4"]
            delta_text = (
                f"{delta['delta_auroc']:+.3f} "
                f"[{delta['ci_lower']:+.3f}, {delta['ci_upper']:+.3f}]"
            )
        lines.append(
            f"| {DISPLAY_NAMES[key]} | "
            f"{stats['n_benign_patients']}/{stats['n_malignant_patients']} | "
            f"{stats['auroc']:.3f} [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}] | "
            f"{delta_text} |"
        )
    lines.extend(["", "## Claim gate", "", report["claim_gate"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def strip_private_frames(report: dict[str, Any]) -> dict[str, Any]:
    clean = {
        "source": report["source"],
        "model": report["model"],
        "dataset": report["dataset"],
        "inner_selection_metric": report["inner_selection_metric"],
        "endpoints": {},
        "claim_gate": report["claim_gate"],
    }
    for name, endpoint in report["endpoints"].items():
        clean["endpoints"][name] = {
            key: value for key, value in endpoint.items() if key != "_patient_predictions"
        }
    return clean


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    image_summaries = run_image_condition_probes()
    c4_patient = image_summaries["C4_lesion-removed-inpaint"]["_patient_predictions"]
    for condition in ["C5d_dilated-mask-only", "C6d_dilated-geometry-rendering"]:
        image_summaries[condition]["delta_vs_patient_level_tuned_c4"] = paired_delta(
            image_summaries[condition]["_patient_predictions"],
            c4_patient,
        )
    geometry = run_patient_tuned_dilated_geometry_probe(c4_patient)

    endpoints = {
        **image_summaries,
        "dilated_mask_geometry_features": geometry,
    }
    if (
        endpoints["C5d_dilated-mask-only"]["delta_vs_patient_level_tuned_c4"]["delta_auroc"] >= -0.01
        and endpoints["C6d_dilated-geometry-rendering"]["delta_vs_patient_level_tuned_c4"]["delta_auroc"] >= -0.01
        and endpoints["dilated_mask_geometry_features"]["delta_vs_patient_level_tuned_c4"]["delta_auroc"] >= -0.02
    ):
        gate = (
            "Patient-level inner-CV tuning preserves the qualitative source-ambiguity "
            "pattern for the primary DINOv3/BUS-BRA controls."
        )
    else:
        gate = (
            "Patient-level inner-CV tuning weakens at least one primary control; "
            "the manuscript should downgrade the inner-CV robustness wording."
        )

    report = {
        "source": "BIBM final patient-level inner-CV tuning sensitivity",
        "model": MODEL,
        "dataset": DATASET,
        "inner_selection_metric": "patient_auroc",
        "endpoints": endpoints,
        "claim_gate": gate,
    }
    json_path = RESULTS_DIR / "bibm_patient_level_tuning_sensitivity.json"
    md_path = RESULTS_DIR / "bibm_patient_level_tuning_sensitivity.md"
    json_path.write_text(json.dumps(strip_private_frames(report), indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
