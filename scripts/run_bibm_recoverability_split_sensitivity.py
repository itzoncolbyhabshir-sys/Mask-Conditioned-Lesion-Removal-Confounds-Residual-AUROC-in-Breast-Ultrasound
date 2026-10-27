"""Split-registry sensitivity for C4 geometry recoverability.

The main recoverability analysis uses the fixed patient-disjoint split registry
stored in ``manifest.csv``. This script checks whether the C4 recoverability
summary is stable under alternate patient-disjoint five-fold registries created
with seeds 2026--2030. It reuses cached DINOv3/BUS-BRA C4 features and the same
Ridge target protocol; it does not extract new encoder features.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_bibm_geometry_recoverability import (  # noqa: E402
    CORE_TARGETS,
    FEATURE_NAMES,
    fit_predict_target,
    load_feature_table,
    metric_values,
)
from src.config import N_SPLITS, RESULTS_DIR, SEEDS  # noqa: E402


MODEL = "DINOv3"
DATASET = "BUS-BRA"
CONDITION = "C4_lesion-removed-inpaint"


def _norm_path(path: str | Path) -> str:
    return str(Path(path).resolve()).casefold()


def attach_labels(table: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.read_csv(ROOT / "manifest.csv")
    manifest = manifest[manifest["dataset_name"].eq(DATASET)].copy()
    label_lookup = {
        _norm_path(row.image_path): int(str(row.label).lower() == "malignant")
        for row in manifest.itertuples(index=False)
    }
    out = table.copy()
    out["_label"] = [
        label_lookup.get(_norm_path(path), np.nan)
        for path in out["image_path"].astype(str)
    ]
    if out["_label"].isna().any():
        missing = int(out["_label"].isna().sum())
        raise RuntimeError(f"Could not attach labels for {missing} recoverability rows")
    out["_label"] = out["_label"].astype(int)
    return out


def assign_split_registry(table: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = table.copy()
    y = out["_label"].to_numpy(dtype=int)
    groups = out["patient_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=int(seed),
    )
    split_id = np.full(len(out), -1, dtype=int)
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(out)), y, groups)):
        split_id[test_idx] = fold
    if (split_id < 0).any():
        raise RuntimeError("Split registry left rows unassigned")
    out["split_id"] = split_id
    return out


def summarize_registry(X: np.ndarray, labeled_table: pd.DataFrame, seed: int) -> dict[str, Any]:
    table = assign_split_registry(labeled_table, seed)
    targets: dict[str, Any] = {}
    for target in FEATURE_NAMES:
        predictions, _ = fit_predict_target(X, table, target)
        metrics = metric_values(predictions)
        targets[target] = {
            "spearman": float(metrics["spearman"]),
            "r2": float(metrics["r2"]),
            "standardized_mae": float(metrics["standardized_mae"]),
        }
    return {
        "seed": int(seed),
        "n_images": int(len(table)),
        "n_patients": int(table["patient_id"].nunique()),
        "median_all_target_spearman": float(np.nanmedian([
            value["spearman"] for value in targets.values()
        ])),
        "median_core_target_spearman": float(np.nanmedian([
            targets[target]["spearman"] for target in CORE_TARGETS
        ])),
        "targets": targets,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Recoverability split-registry sensitivity",
        "",
        "DINOv3/BUS-BRA C4 recoverability was rerun under alternate patient-disjoint five-fold split registries. This sensitivity output includes all nine stored dilated-mask geometry targets; manuscript Table V reports eight non-redundant targets, excluding the raw area ratio. Metrics are descriptive.",
        "",
        "| Split seed | All-nine median Spearman | Four size/location median Spearman |",
        "| ---: | ---: | ---: |",
    ]
    for row in report["seed_summaries"]:
        lines.append(
            f"| {row['seed']} | {row['median_all_target_spearman']:.3f} | "
            f"{row['median_core_target_spearman']:.3f} |"
        )
    summary = report["aggregate"]
    lines.extend(
        [
            "",
            "## Aggregate range",
            "",
            (
                f"All-nine median Spearman range: "
                f"{summary['all_target_median_min']:.3f}--{summary['all_target_median_max']:.3f}; "
                f"four size/location median Spearman range: "
                f"{summary['core_target_median_min']:.3f}--{summary['core_target_median_max']:.3f}."
            ),
            "",
            "## Claim gate",
            "",
            report["claim_gate"],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    X, table = load_feature_table(CONDITION)
    labeled_table = attach_labels(table)
    seed_summaries = [summarize_registry(X, labeled_table, seed) for seed in SEEDS]
    all_medians = np.asarray([
        row["median_all_target_spearman"] for row in seed_summaries
    ], dtype=float)
    core_medians = np.asarray([
        row["median_core_target_spearman"] for row in seed_summaries
    ], dtype=float)
    aggregate = {
        "all_target_median_mean": float(np.nanmean(all_medians)),
        "all_target_median_min": float(np.nanmin(all_medians)),
        "all_target_median_max": float(np.nanmax(all_medians)),
        "core_target_median_mean": float(np.nanmean(core_medians)),
        "core_target_median_min": float(np.nanmin(core_medians)),
        "core_target_median_max": float(np.nanmax(core_medians)),
    }
    if aggregate["all_target_median_min"] >= 0.80 and aggregate["core_target_median_min"] >= 0.90:
        gate = "Recoverability medians remain stable across alternate split registries."
    else:
        gate = "Recoverability medians vary enough across split registries to require limitation wording."
    report = {
        "source": "BIBM C4 recoverability split-registry sensitivity",
        "model": MODEL,
        "dataset": DATASET,
        "condition": CONDITION,
        "seeds": [int(seed) for seed in SEEDS],
        "target_source": "dilated intervention-mask geometry used by C4",
        "targets": FEATURE_NAMES,
        "core_targets": CORE_TARGETS,
        "seed_summaries": seed_summaries,
        "aggregate": aggregate,
        "claim_gate": gate,
    }
    json_path = RESULTS_DIR / "bibm_recoverability_split_sensitivity.json"
    md_path = RESULTS_DIR / "bibm_recoverability_split_sensitivity.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
