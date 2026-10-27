"""Summarize the manuscript-facing inpainting-operator sensitivity check.

This is intentionally small: it promotes the already-run DINOv3/BUS-BRA
Navier-Stokes C4 condition into a compact paper table without opening a new
operator matrix. If cached Navier-Stokes features are available, the script also
reuses the geometry-recoverability pipeline to report the footprint-focused
median Spearman for the second operator.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_bibm_geometry_recoverability import (  # noqa: E402
    CORE_TARGETS,
    MANUSCRIPT_TARGETS,
    evaluate_condition,
)
from src.config import RESULTS_DIR  # noqa: E402
from src.statistics import (  # noqa: E402
    aggregate_patient_predictions,
    align_patient_predictions,
    bootstrap_auroc_ci,
    load_predictions,
    paired_bootstrap_difference,
)


MODEL = "DINOv3"
DATASET = "BUS-BRA"
FULL_CONDITION = "C1_full"
OPERATORS = {
    "Telea": "C4_lesion-removed-inpaint",
    "Navier-Stokes": "C4_Navier-Stokes",
}


def summarize_condition(condition: str) -> dict[str, Any]:
    predictions = load_predictions(MODEL, DATASET, condition)
    if predictions.empty:
        raise RuntimeError(f"Missing predictions for {MODEL}/{DATASET}/{condition}")
    patient = aggregate_patient_predictions(predictions)
    y_true = patient["y_true"].to_numpy(dtype=int)
    y_prob = patient["y_prob"].to_numpy(dtype=float)
    patient_ids = patient["patient_id"].astype(str).to_numpy()
    boot = bootstrap_auroc_ci(y_true, y_prob, patient_ids)
    return {
        "condition": condition,
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "ci_lower": float(boot["ci_lower"]),
        "ci_upper": float(boot["ci_upper"]),
        "n_patients": int(len(patient)),
        "n_benign_patients": int((patient["y_true"] == 0).sum()),
        "n_malignant_patients": int((patient["y_true"] == 1).sum()),
        "patient_predictions": patient,
    }


def paired_delta(first: Any, second: Any) -> dict[str, Any]:
    aligned = align_patient_predictions(
        first,
        second,
        first_prob_name="first",
        second_prob_name="second",
    )
    stats = paired_bootstrap_difference(
        aligned["y_true"].to_numpy(dtype=int),
        aligned["first"].to_numpy(dtype=float),
        aligned["second"].to_numpy(dtype=float),
        aligned["patient_id"].astype(str).to_numpy(),
    )
    return {
        "delta_auroc": float(stats["observed_delta_auroc"]),
        "ci_lower": float(stats["ci_lower"]),
        "ci_upper": float(stats["ci_upper"]),
    }


def safe_recoverability(condition: str) -> dict[str, Any]:
    try:
        summary, _ = evaluate_condition(condition)
    except Exception as exc:  # pragma: no cover - manuscript audit fallback
        return {
            "available": False,
            "reason": str(exc),
        }

    footprint = [
        summary["core_targets"][target]["spearman"]["value"]
        for target in CORE_TARGETS
    ]
    nonredundant = [
        summary["targets"][target]["spearman"]["value"]
        for target in MANUSCRIPT_TARGETS
    ]
    return {
        "available": True,
        "median_footprint_spearman": float(np.nanmedian(footprint)),
        "median_nonredundant_spearman": float(np.nanmedian(nonredundant)),
        "target_spearman": {
            target: float(summary["targets"][target]["spearman"]["value"])
            for target in MANUSCRIPT_TARGETS
        },
    }


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# BIBM operator sensitivity",
        "",
        "DINOv3/BUS-BRA patient-level operator sensitivity for the lesion-removed C4 endpoint. This is a focused sensitivity check, not a full inpainting-operator matrix.",
        "",
        "| Operator | C4 AUROC [95% CI] | C4-Full delta | Geometry recoverability |",
        "| --- | ---: | ---: | ---: |",
    ]
    for operator, row in report["operators"].items():
        rec = row["geometry_recoverability"]
        if rec.get("available"):
            rec_text = f"{rec['median_footprint_spearman']:.3f} footprint rho"
        else:
            rec_text = "not assessed"
        lines.append(
            f"| {operator} | {_fmt(row['c4_auroc'])} "
            f"[{_fmt(row['c4_ci_lower'])}, {_fmt(row['c4_ci_upper'])}] | "
            f"{row['c4_minus_full_delta']:+.3f} | {rec_text} |"
        )
    if "navier_minus_telea" in report:
        delta = report["navier_minus_telea"]
        lines.extend(
            [
                "",
                f"Navier-Stokes minus Telea C4 AUROC delta: {delta['delta_auroc']:+.3f} "
                f"(95% CI {delta['ci_lower']:+.3f} to {delta['ci_upper']:+.3f}).",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    full = summarize_condition(FULL_CONDITION)
    report: dict[str, Any] = {
        "source": "BIBM operator sensitivity; DINOv3/BUS-BRA only",
        "model": MODEL,
        "dataset": DATASET,
        "full_condition": FULL_CONDITION,
        "operators": {},
    }
    operator_summaries: dict[str, dict[str, Any]] = {}
    for operator, condition in OPERATORS.items():
        condition_summary = summarize_condition(condition)
        operator_summaries[operator] = condition_summary
        delta_full = paired_delta(
            condition_summary["patient_predictions"],
            full["patient_predictions"],
        )
        recoverability = safe_recoverability(condition)
        report["operators"][operator] = {
            "condition": condition,
            "c4_auroc": condition_summary["auroc"],
            "c4_ci_lower": condition_summary["ci_lower"],
            "c4_ci_upper": condition_summary["ci_upper"],
            "c4_minus_full_delta": delta_full["delta_auroc"],
            "c4_minus_full_ci_lower": delta_full["ci_lower"],
            "c4_minus_full_ci_upper": delta_full["ci_upper"],
            "n_patients": condition_summary["n_patients"],
            "geometry_recoverability": recoverability,
        }

    if "Telea" in operator_summaries and "Navier-Stokes" in operator_summaries:
        report["navier_minus_telea"] = paired_delta(
            operator_summaries["Navier-Stokes"]["patient_predictions"],
            operator_summaries["Telea"]["patient_predictions"],
        )

    # Drop non-serializable patient prediction frames before writing.
    json_path = RESULTS_DIR / "bibm_operator_sensitivity.json"
    md_path = RESULTS_DIR / "bibm_operator_sensitivity.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
