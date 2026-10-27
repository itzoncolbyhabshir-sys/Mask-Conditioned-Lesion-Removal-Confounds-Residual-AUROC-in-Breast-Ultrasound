"""
05_probe.py — 线性探针训练与评测 SOP
=========================================
- For each (model, dataset, condition, fold, seed):
  * Training fold: StandardScaler fit→transform
  * LogisticRegression(class_weight=balanced, max_iter=5000, L2)
  * C selected via group-aware 3-fold CV on training fold (logspace(-3,3,7))
  * Test fold: output y_prob
- Patient-level aggregation: same patient multi-image y_prob → mean
  → patient-level (y_true, y_prob)
- All metrics computed at patient level
- Metrics: AUROC (primary), AUPRC, balanced accuracy, sensitivity/specificity

Usage: python -m src.05_probe --model DINOv3 --dataset BUS-BRA --condition C1_full
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import argparse
from typing import Dict, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, recall_score,
)
import warnings
warnings.filterwarnings("ignore")

from src.config import (
    ROOT, FEATURES_DIR, PREDICTIONS_DIR, RESULTS_DIR,
    N_SPLITS, SEEDS, PROBE_CONFIG, PRIMARY_METRIC, SECONDARY_METRICS,
)


def train_probe(
    X_train: np.ndarray, y_train: np.ndarray,
    groups_train: np.ndarray = None,
    sample_weight: np.ndarray = None,
    inner_selection_metric: str = "image_auroc",
    C_grid: list = None,
    inner_cv: int = 3,
    class_weight: str = "balanced",
    max_iter: int = 5000,
    random_state: int = 2026,
    return_metadata: bool = False,
) -> LogisticRegression:
    """
    Train logistic regression with inner CV for C selection.
    Group-aware when groups provided.
    """
    if C_grid is None:
        C_grid = PROBE_CONFIG["C_grid"]

    classes, class_counts = np.unique(y_train, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Probe training requires both benign and malignant samples")

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.shape[0] != y_train.shape[0]:
            raise ValueError("sample_weight must have one value per training sample")
    if inner_selection_metric not in {"image_auroc", "patient_auroc"}:
        raise ValueError("inner_selection_metric must be 'image_auroc' or 'patient_auroc'")
    if inner_selection_metric == "patient_auroc" and groups_train is None:
        raise ValueError("patient_auroc inner selection requires groups_train")

    best_score = -np.inf
    best_clf = None
    cv_splits = []
    cv_mode = "fixed_C"
    cv_n_splits = 0

    metadata = {
        "inner_cv_mode": "fixed_C",
        "inner_cv_n_splits": 0,
        "selected_C": None,
        "inner_cv_score": None,
        "inner_selection_metric": inner_selection_metric,
        "n_train_samples": int(len(y_train)),
        "n_train_groups": int(len(np.unique(groups_train))) if groups_train is not None else None,
        "fallback_reason": None,
    }

    if groups_train is not None:
        groups_train = np.asarray(groups_train)
        requested_splits = []
        if inner_cv >= 3:
            requested_splits.append(3)
        if inner_cv >= 2:
            requested_splits.append(2)
        for n_group_splits in requested_splits:
            groups_per_class = [
                len(np.unique(groups_train[y_train == cls])) for cls in classes
            ]
            if (
                n_group_splits > len(np.unique(groups_train))
                or n_group_splits > min(groups_per_class)
            ):
                continue
            try:
                cv = StratifiedGroupKFold(
                    n_splits=n_group_splits, shuffle=True, random_state=random_state
                )
                candidate_splits = list(cv.split(X_train, y_train, groups_train))
            except ValueError:
                candidate_splits = []

            candidate_splits = [
                (train_idx, val_idx)
                for train_idx, val_idx in candidate_splits
                if len(np.unique(y_train[train_idx])) == 2
                and len(np.unique(y_train[val_idx])) == 2
            ]
            if candidate_splits:
                cv_splits = candidate_splits
                cv_mode = f"group{n_group_splits}"
                cv_n_splits = n_group_splits
                break

    cv_splits = [
        (train_idx, val_idx)
        for train_idx, val_idx in cv_splits
        if len(np.unique(y_train[train_idx])) == 2
        and len(np.unique(y_train[val_idx])) == 2
    ]

    for C in C_grid:
        scores = []
        for train_idx, val_idx in cv_splits:
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            clf = LogisticRegression(
                C=C,
                class_weight=class_weight,
                max_iter=max_iter,
                solver="lbfgs",
                random_state=random_state,
            )
            try:
                fit_kwargs = {}
                if sample_weight is not None:
                    fit_kwargs["sample_weight"] = sample_weight[train_idx]
                clf.fit(X_tr_s, y_tr, **fit_kwargs)
                proba = clf.predict_proba(X_val_s)[:, 1]
                if inner_selection_metric == "patient_auroc":
                    val_patient = pd.DataFrame(
                        {
                            "patient_id": groups_train[val_idx],
                            "y_true": y_val,
                            "y_prob": proba,
                        }
                    ).groupby("patient_id", as_index=False).agg(
                        y_true=("y_true", "max"),
                        y_prob=("y_prob", "mean"),
                    )
                    if val_patient["y_true"].nunique() < 2:
                        continue
                    scores.append(roc_auc_score(val_patient["y_true"], val_patient["y_prob"]))
                else:
                    scores.append(roc_auc_score(y_val, proba))
            except ValueError:
                continue

        if not scores:
            continue
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_clf = LogisticRegression(
                C=C,
                class_weight=class_weight,
                max_iter=max_iter,
                solver="lbfgs",
                random_state=random_state,
            )
            metadata["selected_C"] = float(C)
            metadata["inner_cv_score"] = float(mean_score)

    if best_clf is None:
        fallback_c = 1.0 if 1.0 in C_grid else C_grid[len(C_grid) // 2]
        best_clf = LogisticRegression(
            C=fallback_c,
            class_weight=class_weight,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=random_state,
        )
        metadata["selected_C"] = float(fallback_c)
        metadata["inner_cv_score"] = None
        if groups_train is None:
            metadata["fallback_reason"] = "no_groups_provided"
        elif not cv_splits:
            metadata["fallback_reason"] = "no_valid_group_inner_split"
        else:
            metadata["fallback_reason"] = "no_valid_inner_scores"
    else:
        metadata["inner_cv_mode"] = cv_mode
        metadata["inner_cv_n_splits"] = int(cv_n_splits)

    if best_clf is not None and metadata["inner_cv_mode"] == "fixed_C":
        metadata["inner_cv_n_splits"] = 0

    # Final fit on all training data
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    best_clf.fit(X_train_s, y_train, **fit_kwargs)

    if return_metadata:
        return best_clf, scaler, metadata
    return best_clf, scaler


def patient_balanced_sample_weights(groups: np.ndarray) -> np.ndarray:
    """Per-image weights that give each training patient equal total weight."""
    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError("groups must be a one-dimensional array")
    counts = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy(dtype=float)
    weights = 1.0 / counts
    mean_weight = float(weights.mean()) if len(weights) else 1.0
    if mean_weight > 0:
        weights = weights / mean_weight
    return weights


def patient_level_aggregate(
    df_test: pd.DataFrame,
    y_col: str = "y_true",
    prob_col: str = "y_prob",
    patient_col: str = "patient_id",
) -> pd.DataFrame:
    """
    Aggregate to patient level: mean probability and any-malignant target.

    The maximum binary target implements the clinical rule that a patient is
    malignant when any sampled lesion is malignant. It leaves record-level
    labels unchanged for auditability.
    """
    # Report mixed lesion labels without mutating the record-level targets.
    for pid, group in df_test.groupby(patient_col):
        if group[y_col].nunique() > 1:
            print(f"[WARN] Patient {pid} has inconsistent labels: {group[y_col].values}")

    agg = df_test.groupby(patient_col).agg(
        y_true=(y_col, "max"),
        y_prob=(prob_col, "mean"),
        n_images=(y_col, "count"),
    ).reset_index()

    return agg


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute all metrics."""
    metrics = {}

    try:
        metrics["AUROC"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["AUROC"] = np.nan

    try:
        metrics["AUPRC"] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics["AUPRC"] = np.nan

    # Balanced accuracy at default 0.5 threshold
    y_pred = (y_prob >= 0.5).astype(int)
    try:
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    except ValueError:
        metrics["balanced_accuracy"] = np.nan

    # Sensitivity = recall of positive (malignant) class
    try:
        metrics["sensitivity"] = recall_score(y_true, y_pred, pos_label=1)
    except ValueError:
        metrics["sensitivity"] = np.nan

    # Specificity = recall of negative (benign) class
    try:
        metrics["specificity"] = recall_score(y_true, y_pred, pos_label=0)
    except ValueError:
        metrics["specificity"] = np.nan

    return metrics




def run_probe_experiment(
    model_name: str,
    dataset_name: str,
    condition: str,
    seeds: list = None,
    use_split_ids: bool = True,
    patient_label_overrides: dict = None,
    result_tag: str = None,
    sample_weight_strategy: str = None,
    inner_selection_metric: str = "image_auroc",
) -> dict:
    """
    Run full probe experiment for one (model, dataset, condition).

    Steps:
    1. Load features
    2. For each fold & seed:
       a. Split train/test based on pre-assigned split_id
       b. Train probe with inner CV on training fold
       c. Evaluate on test fold
    3. Patient-level aggregation
    4. Compute patient-level metrics
    """
    if seeds is None:
        seeds = SEEDS
    output_condition = result_tag or condition

    # Load features
    feat_path = FEATURES_DIR / model_name / dataset_name / f"{condition}.npy"
    id_path = FEATURES_DIR / model_name / dataset_name / f"{condition}_ids.json"

    if not feat_path.exists():
        print(f"[SKIP] Features not found: {feat_path}")
        return {}

    features = np.load(feat_path)
    with open(id_path) as f:
        ids = json.load(f)

    print(f"[FEATURES] {features.shape} from {feat_path}")

    # Load manifest for labels and split info
    manifest = pd.read_csv(ROOT / "manifest.csv")
    manifest = manifest[manifest["dataset_name"] == dataset_name].copy()
    # Build lookup from image_path → label, patient_id, split_id
    def normalize_path(value: str) -> str:
        return str(Path(value).resolve()).casefold()

    exact_lookup = {
        normalize_path(row.image_path): row
        for row in manifest.itertuples(index=False)
    }
    stem_groups = {}
    for row in manifest.itertuples(index=False):
        stem_groups.setdefault(Path(row.image_path).stem.casefold(), []).append(row)
    stem_lookup = {stem: rows[0] for stem, rows in stem_groups.items() if len(rows) == 1}

    # Match IDs to manifest
    labels = []
    patient_ids = []
    split_ids = []
    for img_id in ids:
        # Normalize path matching
        img_path = Path(img_id)
        row = exact_lookup.get(normalize_path(str(img_path)))
        if row is None:
            row = stem_lookup.get(img_path.stem.casefold())

        if row is not None:
            if patient_label_overrides is None:
                label = row.label
            else:
                override = patient_label_overrides.get(str(row.patient_id))
                if override is None:
                    label = "unknown"
                elif isinstance(override, str):
                    label = override
                else:
                    label = "malignant" if int(override) == 1 else "benign"
            labels.append(label)
            patient_ids.append(row.patient_id)
            split_ids.append(row.split_id)
        else:
            reason = "ambiguous" if img_path.stem.casefold() in stem_groups else "missing"
            print(f"[WARN] {reason.capitalize()} manifest match for {img_path.stem}")
            labels.append("unknown")
            patient_ids.append(f"unk_{img_path.stem}")
            split_ids.append(-1)

    y = np.array([1 if l == "malignant" else 0 for l in labels])
    groups = np.array(patient_ids)
    splits = np.array(split_ids, dtype=int)

    # Filter valid
    valid = (splits >= 0) & (np.array(labels) != "unknown")
    features = features[valid]
    y = y[valid]
    groups = groups[valid]
    splits = splits[valid]
    ids = [ids[i] for i in range(len(valid)) if valid[i]]

    if len(y) == 0:
        print(f"[SKIP] No feature IDs could be aligned for {model_name}/{dataset_name}/{condition}")
        return {}
    if len(np.unique(y)) < 2:
        print(f"[SKIP] Aligned data contain only one class for {model_name}/{dataset_name}/{condition}")
        return {}

    print(f"[DATA] {len(y)} samples, {y.sum()} malignant, "
          f"{(1-y).sum()} benign, {len(np.unique(groups))} patients")

    # Per-fold, per-seed evaluation
    all_results = []

    for fold in range(N_SPLITS):
        train_mask = splits != fold
        test_mask = splits == fold

        X_train_all = features[train_mask]
        y_train_all = y[train_mask]
        groups_train_all = groups[train_mask]
        sample_weight_train = None
        if sample_weight_strategy == "patient_balanced":
            sample_weight_train = patient_balanced_sample_weights(groups_train_all)
        elif sample_weight_strategy is not None:
            raise ValueError(f"Unknown sample_weight_strategy: {sample_weight_strategy}")

        X_test = features[test_mask]
        y_test = y[test_mask]
        groups_test = groups[test_mask]
        ids_test = [ids[i] for i in range(len(ids)) if test_mask[i]]

        if not train_mask.any() or not test_mask.any():
            print(f"[SKIP] Fold {fold} has an empty train or test partition")
            continue
        if len(np.unique(y_train_all)) < 2 or len(np.unique(y_test)) < 2:
            print(f"[SKIP] Fold {fold} does not contain both classes in train and test")
            continue

        # Map to original image paths for patient-level aggregation
        test_records = pd.DataFrame({
            "image_path": ids_test,
            "y_true": y_test,
            "patient_id": groups_test,
        })

        for seed in seeds:
            # Train probe (C selection via inner CV)
            try:
                clf, scaler, probe_metadata = train_probe(
                    X_train_all, y_train_all, groups_train_all,
                    sample_weight=sample_weight_train,
                    inner_selection_metric=inner_selection_metric,
                    C_grid=PROBE_CONFIG["C_grid"],
                    inner_cv=PROBE_CONFIG["inner_cv"],
                    class_weight=PROBE_CONFIG["class_weight"],
                    max_iter=PROBE_CONFIG["max_iter"],
                    random_state=seed,
                    return_metadata=True,
                )
            except ValueError as exc:
                print(f"[SKIP] Fold {fold}, seed {seed}: {exc}")
                continue

            # Predict
            X_test_s = scaler.transform(X_test)
            y_prob = clf.predict_proba(X_test_s)[:, 1]

            test_records["y_prob"] = y_prob

            # Patient-level aggregation
            patient_df = patient_level_aggregate(test_records)

            # Metrics
            metrics = compute_metrics(
                patient_df["y_true"].values,
                patient_df["y_prob"].values,
            )

            result = {
                "model": model_name,
                "dataset": dataset_name,
                "condition": output_condition,
                "fold": fold,
                "seed": seed,
                "n_test_images": len(y_test),
                "n_test_patients": len(patient_df),
                "sample_weight_strategy": sample_weight_strategy or "none",
                "inner_selection_metric": probe_metadata["inner_selection_metric"],
                "inner_cv_mode": probe_metadata["inner_cv_mode"],
                "inner_cv_n_splits": probe_metadata["inner_cv_n_splits"],
                "selected_C": probe_metadata["selected_C"],
                "inner_cv_score": probe_metadata["inner_cv_score"],
                "inner_cv_fallback_reason": probe_metadata["fallback_reason"],
                **metrics,
            }
            all_results.append(result)

            # Save predictions
            pred_dir = PREDICTIONS_DIR / model_name / dataset_name / output_condition
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_df = test_records.copy()
            pred_df.to_csv(pred_dir / f"fold{fold}_seed{seed}.csv", index=False)

    # Aggregate across folds and seeds
    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        print(f"[SKIP] No valid folds for {model_name}/{dataset_name}/{condition}")
        return {}

    # Summary
    summary = {}
    for metric in [PRIMARY_METRIC] + SECONDARY_METRICS:
        vals = results_df[metric].dropna()
        if len(vals) > 0:
            summary[metric] = {
                "mean": vals.mean(),
                "std": vals.std(),
                "min": vals.min(),
                "max": vals.max(),
            }

    print(f"\n[RESULTS] {model_name}/{dataset_name}/{output_condition}")
    for metric, stats in summary.items():
        print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    # Save summary
    summary_dir = RESULTS_DIR / model_name / dataset_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(summary_dir / f"{output_condition}_results.csv", index=False)

    # Save cross-fold record-level predictions; statistical aggregation is
    # performed later so mixed-lesion patient targets remain auditable.
    all_preds = []
    for _, row in results_df.iterrows():
        pred_file = (PREDICTIONS_DIR / model_name / dataset_name /
                     output_condition / f"fold{row['fold']}_seed{row['seed']}.csv")
        if pred_file.exists():
            preds = pd.read_csv(pred_file)
            preds["fold"] = row["fold"]
            preds["seed"] = row["seed"]
            all_preds.append(preds)

    if all_preds:
        all_preds_df = pd.concat(all_preds, ignore_index=True)
        all_preds_df.to_csv(summary_dir / f"{output_condition}_all_predictions.csv", index=False)

    return summary




def main():
    parser = argparse.ArgumentParser(description="Linear Probe Training & Evaluation")
    parser.add_argument("--model", type=str, default="DINOv3")
    parser.add_argument("--dataset", type=str, default="BUS-BRA")
    parser.add_argument("--condition", type=str, default="C1_full")
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    args = parser.parse_args()

    run_probe_experiment(
        model_name=args.model,
        dataset_name=args.dataset,
        condition=args.condition,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
