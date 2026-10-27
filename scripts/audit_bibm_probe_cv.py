"""Audit inner-CV fallback modes for the BIBM logistic probes.

The manuscript protocol permits only group-aware inner tuning:
group-aware 3-fold, then group-aware 2-fold, then fixed C=1.0.  This audit
counts which mode each outer-fold/seed training partition uses and explicitly
documents that image-level inner CV is not used.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import N_SPLITS, PROBE_CONFIG, RESULTS_DIR, SEEDS


DATASETS = ["BUS-BRA", "BUS-UCLM"]


def valid_group_split_exists(
    y_train: np.ndarray,
    groups_train: np.ndarray,
    n_splits: int,
    seed: int,
) -> bool:
    classes = np.unique(y_train)
    if len(classes) < 2:
        return False
    groups_per_class = [
        len(np.unique(groups_train[y_train == cls])) for cls in classes
    ]
    if n_splits > len(np.unique(groups_train)) or n_splits > min(groups_per_class):
        return False
    try:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        splits = list(splitter.split(np.zeros_like(y_train), y_train, groups_train))
    except ValueError:
        return False
    for train_idx, val_idx in splits:
        if len(np.unique(y_train[train_idx])) < 2:
            return False
        if len(np.unique(y_train[val_idx])) < 2:
            return False
    return True


def inner_cv_mode(y_train: np.ndarray, groups_train: np.ndarray, seed: int) -> str:
    if PROBE_CONFIG["inner_cv"] >= 3 and valid_group_split_exists(
        y_train, groups_train, 3, seed
    ):
        return "group3"
    if PROBE_CONFIG["inner_cv"] >= 2 and valid_group_split_exists(
        y_train, groups_train, 2, seed
    ):
        return "group2"
    return "fixed_C"


def audit_dataset(dataset: str) -> dict[str, Any]:
    manifest = pd.read_csv(ROOT / "manifest.csv")
    manifest = manifest[manifest["dataset_name"].eq(dataset)].copy()
    y = manifest["label"].eq("malignant").astype(int).to_numpy()
    groups = manifest["patient_id"].astype(str).to_numpy()
    splits = manifest["split_id"].astype(int).to_numpy()

    rows: list[dict[str, Any]] = []
    for fold in range(N_SPLITS):
        train_mask = splits != fold
        if not train_mask.any() or len(np.unique(y[train_mask])) < 2:
            continue
        for seed in SEEDS:
            mode = inner_cv_mode(y[train_mask], groups[train_mask], seed)
            rows.append(
                {
                    "dataset": dataset,
                    "fold": int(fold),
                    "seed": int(seed),
                    "inner_cv_mode": mode,
                    "n_train_images": int(train_mask.sum()),
                    "n_train_patients": int(len(np.unique(groups[train_mask]))),
                }
            )

    counts = Counter(row["inner_cv_mode"] for row in rows)
    return {
        "dataset": dataset,
        "n_outer_seed_partitions": len(rows),
        "mode_counts": {
            "group3": int(counts.get("group3", 0)),
            "group2": int(counts.get("group2", 0)),
            "fixed_C": int(counts.get("fixed_C", 0)),
            "image_level_inner_cv": 0,
        },
        "partitions": rows,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# BIBM probe inner-CV audit",
        "",
        "Allowed sequence: group-aware 3-fold -> group-aware 2-fold -> fixed C=1.0. Image-level inner CV is disabled.",
        "",
        "| Dataset | Fold-seed partitions | Group3 | Group2 | Fixed C | Image-level fallback |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, stats in report["datasets"].items():
        counts = stats["mode_counts"]
        lines.append(
            f"| {dataset} | {stats['n_outer_seed_partitions']} | "
            f"{counts['group3']} | {counts['group2']} | {counts['fixed_C']} | "
            f"{counts['image_level_inner_cv']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    report = {
        "source": "BIBM final closure inner-CV leakage audit",
        "protocol": "group3_then_group2_then_fixed_C_no_image_level_fallback",
        "datasets": {dataset: audit_dataset(dataset) for dataset in DATASETS},
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / "bibm_probe_cv_audit.json"
    md_path = RESULTS_DIR / "bibm_probe_cv_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
