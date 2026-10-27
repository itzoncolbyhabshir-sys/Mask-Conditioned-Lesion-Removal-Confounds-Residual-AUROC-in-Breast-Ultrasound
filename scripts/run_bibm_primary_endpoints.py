"""Refresh primary BIBM endpoint summary from prediction CSV files."""

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

from src.config import RESULTS_DIR  # noqa: E402
from src.statistics import (  # noqa: E402
    aggregate_patient_predictions,
    bootstrap_auroc_ci,
    load_predictions,
)


SETTINGS: dict[str, dict[str, Any]] = {
    "DINOv3/BUS-BRA": {
        "role": "primary_audit",
        "conditions": [
            "C1_full",
            "C2_lesion-only",
            "C4_lesion-removed-inpaint",
            "C5_mask-only",
            "C6_shape-only",
        ],
    },
    "ResNet-50/BUS-BRA": {
        "role": "encoder_replication_partial",
        "conditions": [
            "C1_full",
            "C2_lesion-only",
            "C4_lesion-removed-inpaint",
        ],
    },
    "MedSAM2/BUS-BRA": {
        "role": "encoder_replication",
        "conditions": [
            "C1_full",
            "C2_lesion-only",
            "C4_lesion-removed-inpaint",
            "C5_mask-only",
            "C6_shape-only",
        ],
    },
    "UltraSam/BUS-BRA": {
        "role": "sensitivity_known_dataset_level_training_data_exposure",
        "conditions": [
            "C1_full",
            "C2_lesion-only",
            "C4_lesion-removed-inpaint",
            "C5_mask-only",
            "C6_shape-only",
        ],
    },
    "DINOv3/BUS-UCLM": {
        "role": "small_independent_dataset_sensitivity",
        "conditions": [
            "C1_full",
            "C2_lesion-only",
            "C4_lesion-removed-inpaint",
            "C5_mask-only",
            "C6_shape-only",
        ],
    },
}


def summarize_condition(
    model: str,
    dataset: str,
    condition: str,
) -> dict[str, Any] | None:
    predictions = load_predictions(model, dataset, condition)
    if predictions.empty:
        return None
    patient = aggregate_patient_predictions(predictions)
    y_true = patient["y_true"].to_numpy(dtype=int)
    y_prob = patient["y_prob"].to_numpy(dtype=float)
    patient_ids = patient["patient_id"].astype(str).to_numpy()
    if len(np.unique(y_true)) < 2:
        return None
    boot = bootstrap_auroc_ci(y_true, y_prob, patient_ids)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "ci_lower": float(boot["ci_lower"]),
        "ci_upper": float(boot["ci_upper"]),
        "n_patients": int(len(patient)),
        "n_benign_patients": int((patient["y_true"] == 0).sum()),
        "n_malignant_patients": int((patient["y_true"] == 1).sum()),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# BIBM primary endpoints",
        "",
        "Patient-level AUROC summaries generated from stored prediction CSV files. Bootstrap CIs use stratified patient resampling.",
        "",
        "| Setting | Condition | Patients | AUROC [95% CI] |",
        "| --- | --- | ---: | ---: |",
    ]
    for setting, stats in report["settings"].items():
        for condition, values in stats["conditions"].items():
            lines.append(
                f"| {setting} | {condition} | {values['n_patients']} | "
                f"{values['auroc']:.3f} [{values['ci_lower']:.3f}, {values['ci_upper']:.3f}] |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    output: dict[str, Any] = {
        "source": "patient-level predictions aggregated with src.statistics; stratified patient bootstrap defaults; generated for BIBM accept-zone manuscript",
        "settings": {},
    }
    for setting, config in SETTINGS.items():
        model, dataset = setting.split("/", 1)
        condition_stats = {}
        for condition in config["conditions"]:
            stats = summarize_condition(
                model,
                dataset,
                condition,
            )
            if stats is not None:
                condition_stats[condition] = stats
        output["settings"][setting] = {
            "role": config["role"],
            "conditions": condition_stats,
        }

    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / "bibm_primary_endpoints.json"
    md_path = RESULTS_DIR / "bibm_primary_endpoints.md"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_markdown(output, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
