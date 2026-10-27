"""Patient-level AUROC, paired bootstrap and prediction aggregation utilities."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Optional, Callable
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

from src.config import STAT_CONFIG, RESULTS_DIR


def _indices_for_sampled_patients(
    patient_ids: np.ndarray, sampled_patients: np.ndarray
) -> np.ndarray:
    """Expand patient draws to row indices while preserving repeated draws."""
    patient_ids = np.asarray(patient_ids)
    chunks = [np.flatnonzero(patient_ids == pid) for pid in sampled_patients]
    if not chunks:
        return np.array([], dtype=int)
    return np.concatenate(chunks)


def align_patient_predictions(
    first: pd.DataFrame,
    second: pd.DataFrame,
    first_prob_name: str = "y_prob_1",
    second_prob_name: str = "y_prob_2",
) -> pd.DataFrame:
    """Align unique patient predictions deterministically and validate labels."""
    required = {"patient_id", "y_true", "y_prob"}
    for name, frame in (("first", first), ("second", second)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} predictions missing columns: {sorted(missing)}")
        if frame["patient_id"].duplicated().any():
            raise ValueError(f"{name} predictions contain duplicate patient IDs")

    columns = ["patient_id", "y_true", "y_prob"]
    aligned = first[columns].merge(
        second[columns], on="patient_id", how="inner", suffixes=("_first", "_second")
    )
    if not aligned.empty and not np.array_equal(
        aligned["y_true_first"].to_numpy(), aligned["y_true_second"].to_numpy()
    ):
        raise ValueError("Patient labels differ between prediction sets")

    aligned = aligned.rename(columns={
        "y_true_first": "y_true",
        "y_prob_first": first_prob_name,
        "y_prob_second": second_prob_name,
    }).drop(columns=["y_true_second"])
    return aligned.sort_values("patient_id", kind="stable").reset_index(drop=True)


def _stratified_row_bootstrap_indices(
    y_true: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Sample row indices with replacement within each observed class."""
    y_true = np.asarray(y_true)
    chunks = []
    for label in np.unique(y_true):
        class_idx = np.flatnonzero(y_true == label)
        if len(class_idx) == 0:
            continue
        chunks.append(rng.choice(class_idx, size=len(class_idx), replace=True))
    if not chunks:
        return np.array([], dtype=int)
    return np.concatenate(chunks)


def _patient_class_table(
    y_true: np.ndarray,
    patient_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return patient IDs and any-positive patient labels for stratified resampling."""
    y_true = np.asarray(y_true)
    patient_ids = np.asarray(patient_ids)
    patients = np.unique(patient_ids)
    patient_labels = np.array(
        [int(np.max(y_true[patient_ids == pid])) for pid in patients],
        dtype=int,
    )
    return patients, patient_labels


def _stratified_patient_bootstrap_indices(
    y_true: np.ndarray,
    patient_ids: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Sample patients with replacement within patient-label strata and expand rows."""
    patients, patient_labels = _patient_class_table(y_true, patient_ids)
    sampled_patients = []
    for label in np.unique(patient_labels):
        class_patients = patients[patient_labels == label]
        if len(class_patients) == 0:
            continue
        sampled_patients.append(
            rng.choice(class_patients, size=len(class_patients), replace=True)
        )
    if not sampled_patients:
        return np.array([], dtype=int)
    return _indices_for_sampled_patients(patient_ids, np.concatenate(sampled_patients))


def _patient_bootstrap_plan(
    y_true: np.ndarray,
    patient_ids: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Precompute row indices and patient-label strata for fast patient bootstrap."""
    y_true = np.asarray(y_true)
    patient_ids = np.asarray(patient_ids)
    patients = np.unique(patient_ids)
    patient_row_indices = [np.flatnonzero(patient_ids == pid) for pid in patients]
    patient_labels = np.array(
        [int(np.max(y_true[idx])) for idx in patient_row_indices],
        dtype=int,
    )
    strata = [
        np.flatnonzero(patient_labels == label)
        for label in np.unique(patient_labels)
    ]
    return patient_row_indices, strata


def _draw_from_patient_bootstrap_plan(
    patient_row_indices: list[np.ndarray],
    strata: list[np.ndarray],
    rng: np.random.RandomState,
) -> np.ndarray:
    sampled_patient_indices = [
        rng.choice(stratum, size=len(stratum), replace=True)
        for stratum in strata
        if len(stratum) > 0
    ]
    if not sampled_patient_indices:
        return np.array([], dtype=int)
    return np.concatenate([
        patient_row_indices[int(patient_idx)]
        for patient_idx in np.concatenate(sampled_patient_indices)
    ])


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    patient_ids: np.ndarray = None,
    n_iter: int = None,
    ci_level: float = None,
    seed: int = 2026,
) -> dict:
    """
    Stratified bootstrap for AUROC 95% CI.

    If patient_ids are provided, resample patients with replacement within
    benign/malignant patient-label strata and keep all rows for each selected
    patient. If patient_ids are absent, resample rows within class strata. This
    avoids undefined one-class AUROC bootstrap samples.

    Returns: {auroc_mean, ci_lower, ci_upper, bootstrap_aurocs}
    """
    if n_iter is None:
        n_iter = STAT_CONFIG["bootstrap_iter"]
    if ci_level is None:
        ci_level = STAT_CONFIG["ci_level"]

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return {
            "observed_auroc": float("nan"),
            "auroc_mean": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "bootstrap_aurocs": np.array([], dtype=float),
        }

    rng = np.random.RandomState(seed)
    observed = float(roc_auc_score(y_true, y_prob))
    boot_aurocs = []

    if patient_ids is not None:
        patient_ids = np.asarray(patient_ids)
        patient_rows, patient_strata = _patient_bootstrap_plan(y_true, patient_ids)
        for _ in range(n_iter):
            idx = _draw_from_patient_bootstrap_plan(patient_rows, patient_strata, rng)
            yt = y_true[idx]
            yp = y_prob[idx]
            boot_aurocs.append(roc_auc_score(yt, yp))
    else:
        for _ in range(n_iter):
            idx = _stratified_row_bootstrap_indices(y_true, rng)
            yt = y_true[idx]
            yp = y_prob[idx]
            boot_aurocs.append(roc_auc_score(yt, yp))

    boot_aurocs = np.array(boot_aurocs)
    alpha = 1 - ci_level
    ci_lower = np.percentile(boot_aurocs, 100 * alpha / 2)
    ci_upper = np.percentile(boot_aurocs, 100 * (1 - alpha / 2))

    return {
        "observed_auroc": observed,
        "auroc_mean": boot_aurocs.mean(),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_aurocs": boot_aurocs,
    }


def permutation_test(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    patient_ids: np.ndarray = None,
    n_iter: int = None,
    seed: int = 2026,
) -> dict:
    """
    Permutation test: H0 = labels are random (AUROC = 0.5).

    Shuffles y_true within patients if patient_ids provided,
    then recomputes AUROC. p-value = fraction of permutations
    with AUROC >= observed.

    Returns: {observed_auroc, p_value, perm_aurocs}
    """
    if n_iter is None:
        n_iter = STAT_CONFIG["permutation_iter"]

    rng = np.random.RandomState(seed)
    observed_auroc = roc_auc_score(y_true, y_prob)
    n = len(y_true)
    perm_aurocs = []

    if patient_ids is not None:
        unique_patients = np.unique(patient_ids)
        # Pre-compute patient-level label
        patient_label = {}
        for pid in unique_patients:
            mask = patient_ids == pid
            patient_label[pid] = y_true[mask][0]  # assume consistent

        for _ in range(n_iter):
            shuffled = dict(zip(unique_patients,
                                rng.permutation(list(patient_label.values()))))
            y_shuffled = np.array([shuffled[pid] for pid in patient_ids])
            if len(np.unique(y_shuffled)) < 2:
                perm_aurocs.append(0.5)
            else:
                perm_aurocs.append(roc_auc_score(y_shuffled, y_prob))
    else:
        for _ in range(n_iter):
            y_shuffled = rng.permutation(y_true)
            if len(np.unique(y_shuffled)) < 2:
                perm_aurocs.append(0.5)
            else:
                perm_aurocs.append(roc_auc_score(y_shuffled, y_prob))

    perm_aurocs = np.array(perm_aurocs)
    p_value = (1 + np.sum(perm_aurocs >= observed_auroc)) / (n_iter + 1)

    return {
        "observed_auroc": observed_auroc,
        "p_value": p_value,
        "perm_aurocs": perm_aurocs,
        "perm_mean": perm_aurocs.mean(),
        "perm_std": perm_aurocs.std(),
    }


def paired_bootstrap_difference(
    y_true: np.ndarray,
    y_prob_1: np.ndarray,
    y_prob_2: np.ndarray,
    patient_ids: np.ndarray = None,
    n_iter: int = None,
    seed: int = 2026,
) -> dict:
    """
    Stratified paired bootstrap test for difference between two conditions/systems.

    H0: AUROC_1 = AUROC_2
    Returns: {delta_auroc, ci_lower, ci_upper, p_value}
    p_value from percentile bootstrap: fraction where delta <= 0
    """
    if n_iter is None:
        n_iter = STAT_CONFIG["bootstrap_iter"]

    y_true = np.asarray(y_true)
    y_prob_1 = np.asarray(y_prob_1)
    y_prob_2 = np.asarray(y_prob_2)
    if len(np.unique(y_true)) < 2:
        return {
            "delta_auroc": float("nan"),
            "observed_delta_auroc": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "p_value": float("nan"),
            "bootstrap_deltas": np.array([], dtype=float),
        }

    rng = np.random.RandomState(seed)
    observed_delta = float(
        roc_auc_score(y_true, y_prob_1) - roc_auc_score(y_true, y_prob_2)
    )
    deltas = []

    if patient_ids is not None:
        patient_ids = np.asarray(patient_ids)
        patient_rows, patient_strata = _patient_bootstrap_plan(y_true, patient_ids)
        for _ in range(n_iter):
            idx = _draw_from_patient_bootstrap_plan(patient_rows, patient_strata, rng)
            yt = y_true[idx]
            a1 = roc_auc_score(yt, y_prob_1[idx])
            a2 = roc_auc_score(yt, y_prob_2[idx])
            deltas.append(a1 - a2)
    else:
        for _ in range(n_iter):
            idx = _stratified_row_bootstrap_indices(y_true, rng)
            yt = y_true[idx]
            a1 = roc_auc_score(yt, y_prob_1[idx])
            a2 = roc_auc_score(yt, y_prob_2[idx])
            deltas.append(a1 - a2)

    deltas = np.array(deltas)
    alpha = 1 - STAT_CONFIG["ci_level"]
    ci_lower = np.percentile(deltas, 100 * alpha / 2)
    ci_upper = np.percentile(deltas, 100 * (1 - alpha / 2))
    p_value = (1 + np.sum(deltas <= 0)) / (n_iter + 1)

    return {
        "delta_auroc": deltas.mean(),
        "observed_delta_auroc": observed_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "bootstrap_deltas": deltas,
    }








def load_predictions(model_name: str, dataset_name: str, condition: str) -> pd.DataFrame:
    """Load all predictions for a (model, dataset, condition) combo."""
    pred_dir = RESULTS_DIR / model_name / dataset_name
    pred_file = pred_dir / f"{condition}_all_predictions.csv"
    if pred_file.exists():
        return pd.read_csv(pred_file)
    return pd.DataFrame()


def aggregate_patient_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Average seed/image predictions and apply the any-malignant patient target."""
    image_columns = ["patient_id", "y_true"]
    if "image_path" in df.columns:
        image_columns.insert(0, "image_path")
    image_df = df.groupby(image_columns, as_index=False)["y_prob"].mean()
    return image_df.groupby("patient_id", as_index=False).agg(
        y_true=("y_true", "max"),
        y_prob=("y_prob", "mean"),
    )






