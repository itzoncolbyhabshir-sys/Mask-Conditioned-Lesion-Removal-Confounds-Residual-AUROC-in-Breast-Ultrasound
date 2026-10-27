"""Fully covered crop sensitivity for BIBM geometry analyses.

The main dilated geometry targets are computed in the original image/mask
space, while DINOv3 uses a resize-then-center-crop transform. This sensitivity
restricts the primary DINOv3/BUS-BRA evidence to images whose dilated mask is
almost fully retained by the DINOv3 crop, then recomputes the explicit dilated
geometry endpoint and the C4 geometry-recoverability summary from existing
out-of-fold predictions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_bibm_preprocessing import mask_crop_coverage  # noqa: E402
from scripts.run_bibm_geometry_feature_baseline import FEATURE_NAMES  # noqa: E402
from scripts.run_bibm_geometry_recoverability import CORE_TARGETS, patient_cluster_bootstrap  # noqa: E402
from src.statistics import (  # noqa: E402
    aggregate_patient_predictions,
    align_patient_predictions,
    bootstrap_auroc_ci,
    load_predictions,
    paired_bootstrap_difference,
)


MODEL = "DINOv3"
DATASET = "BUS-BRA"
C4_CONDITION = "C4_lesion-removed-inpaint"
CONTROL_CONDITIONS = [
    C4_CONDITION,
    "C5d_dilated-mask-only",
    "C6d_dilated-geometry-rendering",
]
DISPLAY_NAMES = {
    C4_CONDITION: "Lesion-removed (C4)",
    "C5d_dilated-mask-only": "Dilated mask-only (C5d)",
    "C6d_dilated-geometry-rendering": "Dilated geometry rendering (C6d)",
    "dilated_mask_geometry_features": "Explicit dilated-mask geometry",
}
THRESHOLD = 0.95
DINO_TARGET = 256
RESULTS_DIR = ROOT / "results"
MANIFEST_PATH = ROOT / "manifest.csv"


def _norm_path(path: str | Path) -> str:
    return str(Path(path).resolve()).casefold()


def _match_key(image_path: str | Path, patient_id: str) -> str:
    return f"{str(patient_id)}::{Path(str(image_path)).stem.casefold()}"


def build_coverage_table() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[manifest["dataset_name"].eq(DATASET)].copy()
    rows: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        coverage = mask_crop_coverage(row.dilated_mask_path, DINO_TARGET)
        rows.append(
            {
                "image_path": str(row.image_path),
                "image_key": _norm_path(row.image_path),
                "match_key": _match_key(row.image_path, str(row.patient_id)),
                "patient_id": str(row.patient_id),
                "y_true": int(str(row.label) == "malignant"),
                "coverage": float(coverage),
                "fully_covered": bool(coverage >= THRESHOLD),
            }
        )
    return pd.DataFrame(rows)


def _filter_predictions_to_images(frame: pd.DataFrame, retained_keys: set[str]) -> pd.DataFrame:
    if "image_path" not in frame.columns:
        raise ValueError("Prediction frame must contain image_path for fully-covered filtering")
    out = frame.copy()
    out["match_key"] = [
        _match_key(path, patient)
        for path, patient in zip(out["image_path"], out["patient_id"])
    ]
    out = out[out["match_key"].isin(retained_keys)].copy()
    return out.drop(columns=["match_key"])


def endpoint_summary(patient: pd.DataFrame) -> dict[str, Any]:
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
    }


def _prediction_endpoint_subset(
    predictions: pd.DataFrame,
    retained_keys: set[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    filtered = _filter_predictions_to_images(predictions, retained_keys)
    patient = aggregate_patient_predictions(filtered)
    return endpoint_summary(patient), patient


def endpoint_subset(retained_keys: set[str]) -> dict[str, Any]:
    geom_path = RESULTS_DIR / "bibm_dilated_geometry_feature_predictions.csv"
    if not geom_path.exists():
        raise RuntimeError(f"Missing {geom_path}; run run_bibm_dilated_mask_controls.py first")

    endpoint_patients: dict[str, pd.DataFrame] = {}
    endpoints: dict[str, Any] = {}
    for condition in CONTROL_CONDITIONS:
        predictions = load_predictions(MODEL, DATASET, condition)
        if predictions.empty:
            raise RuntimeError(f"Missing predictions for {MODEL}/{DATASET}/{condition}")
        stats, patient = _prediction_endpoint_subset(predictions, retained_keys)
        endpoints[condition] = stats
        endpoint_patients[condition] = patient

    geom = pd.read_csv(geom_path)
    dataset_col = "geometry_dataset" if "geometry_dataset" in geom.columns else "dataset"
    geom = geom[geom[dataset_col].eq(DATASET)].copy()
    geom_stats, geom_patient = _prediction_endpoint_subset(geom, retained_keys)
    endpoints["dilated_mask_geometry_features"] = geom_stats
    endpoint_patients["dilated_mask_geometry_features"] = geom_patient

    c4_patient = endpoint_patients[C4_CONDITION]
    for key in [
        "C5d_dilated-mask-only",
        "C6d_dilated-geometry-rendering",
        "dilated_mask_geometry_features",
    ]:
        aligned = align_patient_predictions(
            endpoint_patients[key],
            c4_patient,
            first_prob_name="y_prob_control",
            second_prob_name="y_prob_c4",
        )
        paired = paired_bootstrap_difference(
            aligned["y_true"].to_numpy(dtype=int),
            aligned["y_prob_control"].to_numpy(dtype=float),
            aligned["y_prob_c4"].to_numpy(dtype=float),
            aligned["patient_id"].astype(str).to_numpy(),
        )
        endpoints[key]["delta_vs_c4"] = {
            "convention": f"{key}_minus_C4",
            "delta_auroc": float(paired["delta_auroc"]),
            "ci_lower": float(paired["ci_lower"]),
            "ci_upper": float(paired["ci_upper"]),
            "p_value": float(paired["p_value"]),
        }

    return {
        "endpoints": endpoints,
        "explicit_dilated_geometry": endpoints["dilated_mask_geometry_features"],
        "c4_reference": endpoints[C4_CONDITION],
        "delta_vs_c4": endpoints["dilated_mask_geometry_features"]["delta_vs_c4"],
    }


def _metric_values(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    if len(y_true) < 2 or float(np.var(y_true)) <= 1e-12:
        return {"spearman": float("nan"), "r2": float("nan"), "standardized_mae": float("nan")}
    rho = float(spearmanr(y_true, y_pred).correlation)
    r2 = float(r2_score(y_true, y_pred))
    smae = float(np.mean(np.abs(y_pred - y_true)) / np.std(y_true))
    return {"spearman": rho, "r2": r2, "standardized_mae": smae}


def recoverability_subset(retained_keys: set[str]) -> dict[str, Any]:
    pred_path = RESULTS_DIR / "bibm_geometry_recoverability_predictions.csv"
    if not pred_path.exists():
        raise RuntimeError(f"Missing {pred_path}; run run_bibm_geometry_recoverability.py first")
    preds = pd.read_csv(pred_path)
    preds = preds[preds["condition"].eq(C4_CONDITION)].copy()
    preds = _filter_predictions_to_images(preds, retained_keys)
    target_summaries: dict[str, Any] = {}
    for target in FEATURE_NAMES:
        frame = preds[preds["target"].eq(target)].copy()
        observed = _metric_values(frame)
        boot = patient_cluster_bootstrap(frame) if not frame.empty else {}
        target_summaries[target] = {
            key: {
                "value": float(observed[key]),
                "ci_lower": float(boot.get(key, {}).get("ci_lower", float("nan"))),
                "ci_upper": float(boot.get(key, {}).get("ci_upper", float("nan"))),
            }
            for key in ("spearman", "r2", "standardized_mae")
        }
    core = {target: target_summaries[target] for target in CORE_TARGETS}
    return {
        "n_prediction_rows": int(len(preds)),
        "n_images": int(preds["image_path"].nunique()),
        "n_patients": int(preds["patient_id"].nunique()),
        "targets": target_summaries,
        "core_targets": core,
        "median_core_spearman": float(np.nanmedian([
            stats["spearman"]["value"] for stats in core.values()
        ])),
        "median_core_r2": float(np.nanmedian([
            stats["r2"]["value"] for stats in core.values()
        ])),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    cov = report["coverage"]
    endpoints = report["endpoints"]
    rec = report["recoverability"]
    lines = [
        "# Fully-covered geometry sensitivity",
        "",
        f"Primary DINOv3/BUS-BRA images were filtered to dilated-mask crop coverage >= {THRESHOLD:.2f}.",
        "",
        "| Images retained | Patients retained | Below 0.95 rate | Below 0.90 rate |",
        "| ---: | ---: | ---: | ---: |",
        (
            f"| {cov['retained_images']} / {cov['total_images']} | "
            f"{cov['retained_patients']} / {cov['total_patients']} | "
            f"{cov['below_0_95_rate']:.3f} | {cov['below_0_90_rate']:.3f} |"
        ),
        "",
        "## Main endpoints on retained images",
        "",
        "| Endpoint | Patients B/M | AUROC [95% CI] | Delta vs C4 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in [
        C4_CONDITION,
        "C5d_dilated-mask-only",
        "C6d_dilated-geometry-rendering",
        "dilated_mask_geometry_features",
    ]:
        stats = endpoints[key]
        if key == C4_CONDITION:
            delta_text = "--"
        else:
            delta = stats["delta_vs_c4"]
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
    lines.extend([
        "",
        "## C4 recoverability on retained images",
        "",
        f"Median core Spearman: {rec['median_core_spearman']:.3f}; median core R2: {rec['median_core_r2']:.3f}.",
        "",
        "| Core target | Spearman | R2 |",
        "| --- | ---: | ---: |",
    ])
    for target, stats in rec["core_targets"].items():
        lines.append(
            f"| {target} | {stats['spearman']['value']:.3f} | {stats['r2']['value']:.3f} |"
        )
    lines.extend(["", "## Claim gate", "", report["claim_gate"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    coverage = build_coverage_table()
    retained = coverage[coverage["fully_covered"]].copy()
    retained_keys = set(retained["match_key"].astype(str))
    if not retained_keys:
        raise RuntimeError("No fully-covered images retained; threshold is too strict")

    endpoint_report = endpoint_subset(retained_keys)
    recoverability = recoverability_subset(retained_keys)
    coverage_summary = {
        "encoder": MODEL,
        "dataset": DATASET,
        "threshold": THRESHOLD,
        "crop_target": DINO_TARGET,
        "total_images": int(len(coverage)),
        "retained_images": int(len(retained)),
        "total_patients": int(coverage["patient_id"].nunique()),
        "retained_patients": int(retained["patient_id"].nunique()),
        "coverage_min": float(coverage["coverage"].min()),
        "coverage_median": float(coverage["coverage"].median()),
        "coverage_q1": float(coverage["coverage"].quantile(0.25)),
        "coverage_q3": float(coverage["coverage"].quantile(0.75)),
        "below_0_95_rate": float((coverage["coverage"] < 0.95).mean()),
        "below_0_90_rate": float((coverage["coverage"] < 0.90).mean()),
    }
    if (
        endpoint_report["endpoints"]["C5d_dilated-mask-only"]["auroc"] >= 0.85
        and endpoint_report["endpoints"]["C6d_dilated-geometry-rendering"]["auroc"] >= 0.85
        and endpoint_report["endpoints"]["dilated_mask_geometry_features"]["auroc"] >= 0.85
        and recoverability["median_core_spearman"] >= 0.50
    ):
        gate = (
            "The fully-covered subset preserves the direction of the primary "
            "C4-aligned mask-control, geometry endpoint, and C4 recoverability findings."
        )
    else:
        gate = (
            "The fully-covered subset weakens at least one geometry endpoint; "
            "the manuscript should downgrade recoverability wording."
        )

    report = {
        "source": "BIBM fully-covered DINOv3/BUS-BRA geometry sensitivity",
        "coverage": coverage_summary,
        "endpoints": endpoint_report["endpoints"],
        "explicit_geometry_endpoint": endpoint_report,
        "recoverability": recoverability,
        "claim_gate": gate,
    }
    json_path = RESULTS_DIR / "bibm_fully_covered_geometry_sensitivity.json"
    md_path = RESULTS_DIR / "bibm_fully_covered_geometry_sensitivity.md"
    csv_path = RESULTS_DIR / "bibm_fully_covered_geometry_coverage.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    coverage.drop(columns=["image_key", "match_key"]).to_csv(csv_path, index=False)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
