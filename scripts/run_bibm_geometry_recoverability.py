"""Recover dilated-mask geometry from frozen DINOv3 representations.

The label-decodability audit shows that dilated intervention-mask geometry is
predictive in isolation. This script tests the adjacent reviewer-visible
question: whether that geometry is recoverable from the edited C4
representation itself. It trains patient-disjoint Ridge regressors from frozen
features to the stored explicit dilated-mask geometry targets. The manuscript
reports eight non-redundant targets after excluding the raw lesion-to-image
area ratio, which is redundant with the square-root area-ratio target.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_bibm_geometry_feature_baseline import FEATURE_NAMES, mask_geometry_features  # noqa: E402
from src.config import FEATURES_DIR, N_SPLITS, RESULTS_DIR  # noqa: E402


MODEL = "DINOv3"
DATASET = "BUS-BRA"
CONDITIONS = [
    "C1_full",
    "C4_lesion-removed-inpaint",
    "C5d_dilated-mask-only",
    "C6d_dilated-geometry-rendering",
    "C10_fixed-circle-inpaint",
    "C11_fixed-square-inpaint",
    "C12_fixed-ellipse-inpaint",
]
CONDITION_LABELS = {
    "C1_full": "Full image",
    "C4_lesion-removed-inpaint": "Lesion-removed C4",
    "C5d_dilated-mask-only": "Dilated mask-only C5d",
    "C6d_dilated-geometry-rendering": "Dilated geometry C6d",
    "C10_fixed-circle-inpaint": "Fixed circle",
    "C11_fixed-square-inpaint": "Fixed square",
    "C12_fixed-ellipse-inpaint": "Fixed ellipse",
}
ALPHA_GRID = np.logspace(-3, 3, 7).tolist()
CORE_TARGETS = [
    "sqrt_normalized_area",
    "normalized_perimeter",
    "centroid_x",
    "centroid_y",
]
MANUSCRIPT_TARGETS = [
    target for target in FEATURE_NAMES if target != "lesion_to_image_area_ratio"
]
CORE_TARGET_RATIONALE = (
    "Core targets summarize quantities that directly encode mask "
    "size/location and the intervention footprint: area, perimeter, centroid x, "
    "and centroid y."
)


def _normalize_path(value: str | Path) -> str:
    return str(Path(value).resolve()).casefold()


def load_feature_table(condition: str) -> tuple[np.ndarray, pd.DataFrame]:
    feat_path = FEATURES_DIR / MODEL / DATASET / f"{condition}.npy"
    id_path = FEATURES_DIR / MODEL / DATASET / f"{condition}_ids.json"
    if not feat_path.exists() or not id_path.exists():
        raise RuntimeError(f"Missing cached features for {MODEL}/{DATASET}/{condition}")

    features = np.load(feat_path)
    ids = json.loads(id_path.read_text(encoding="utf-8"))
    manifest = pd.read_csv(ROOT / "manifest.csv")
    manifest = manifest[manifest["dataset_name"].eq(DATASET)].copy()

    exact = {_normalize_path(row.image_path): row for row in manifest.itertuples(index=False)}
    stem_groups: dict[str, list[Any]] = {}
    for row in manifest.itertuples(index=False):
        stem_groups.setdefault(Path(str(row.image_path)).stem.casefold(), []).append(row)
    stem_lookup = {stem: rows[0] for stem, rows in stem_groups.items() if len(rows) == 1}

    rows: list[dict[str, Any]] = []
    valid_indices: list[int] = []
    for idx, image_id in enumerate(ids):
        path = Path(image_id)
        row = exact.get(_normalize_path(path))
        if row is None:
            stem = path.stem.casefold().split("__", 1)[0]
            row = stem_lookup.get(stem)
        if row is None:
            continue
        target = mask_geometry_features(row.dilated_mask_path)
        out = {
            "image_path": str(row.image_path),
            "feature_id": str(image_id),
            "patient_id": str(row.patient_id),
            "split_id": int(row.split_id),
            "condition": condition,
        }
        out.update(dict(zip(FEATURE_NAMES, target)))
        rows.append(out)
        valid_indices.append(idx)

    if not rows:
        raise RuntimeError(f"No feature IDs aligned to manifest for {condition}")
    return features[np.asarray(valid_indices, dtype=int)], pd.DataFrame(rows)


def choose_alpha(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
) -> float:
    X_train = X[train_indices]
    y_train = y[train_indices]
    group_train = groups[train_indices]
    unique_groups = np.unique(group_train)
    if len(unique_groups) < 3:
        return 1.0

    n_splits = min(3, len(unique_groups))
    cv = GroupKFold(n_splits=n_splits)
    best_alpha = 1.0
    best_score = -np.inf

    for alpha in ALPHA_GRID:
        scores: list[float] = []
        for inner_train, inner_val in cv.split(X_train, y_train, group_train):
            y_inner = y_train[inner_train]
            if float(np.std(y_inner)) <= 1e-12:
                continue
            x_scaler = StandardScaler()
            y_mean = float(y_inner.mean())
            y_std = float(y_inner.std())
            X_tr = x_scaler.fit_transform(X_train[inner_train])
            X_val = x_scaler.transform(X_train[inner_val])
            model = Ridge(alpha=float(alpha))
            model.fit(X_tr, (y_inner - y_mean) / y_std)
            pred = model.predict(X_val) * y_std + y_mean
            if float(np.var(y_train[inner_val])) <= 1e-12:
                continue
            scores.append(float(r2_score(y_train[inner_val], pred)))
        if scores and float(np.mean(scores)) > best_score:
            best_score = float(np.mean(scores))
            best_alpha = float(alpha)
    return best_alpha


def fit_predict_target(
    X: np.ndarray,
    table: pd.DataFrame,
    target_name: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    y = table[target_name].to_numpy(dtype=float)
    groups = table["patient_id"].astype(str).to_numpy()
    splits = table["split_id"].to_numpy(dtype=int)
    predictions: list[pd.DataFrame] = []
    fold_meta: list[dict[str, Any]] = []

    for fold in range(N_SPLITS):
        train_idx = np.flatnonzero(splits != fold)
        test_idx = np.flatnonzero(splits == fold)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        alpha = choose_alpha(X, y, groups, train_idx)
        x_scaler = StandardScaler()
        y_train = y[train_idx]
        y_mean = float(y_train.mean())
        y_std = float(y_train.std())
        if y_std <= 1e-12:
            continue
        X_train = x_scaler.fit_transform(X[train_idx])
        X_test = x_scaler.transform(X[test_idx])
        model = Ridge(alpha=alpha)
        model.fit(X_train, (y_train - y_mean) / y_std)
        y_pred = model.predict(X_test) * y_std + y_mean
        pred = table.loc[test_idx, ["image_path", "patient_id", "split_id", "condition"]].copy()
        pred["target"] = target_name
        pred["y_true"] = y[test_idx]
        pred["y_pred"] = y_pred
        pred["fold"] = fold
        pred["alpha"] = alpha
        predictions.append(pred)
        fold_meta.append({"fold": int(fold), "target": target_name, "alpha": float(alpha)})

    if not predictions:
        raise RuntimeError(f"No recoverability predictions produced for target {target_name}")
    return pd.concat(predictions, ignore_index=True), fold_meta


def metric_values_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) < 2 or float(np.var(y_true)) <= 1e-12:
        r2 = float("nan")
        rho = float("nan")
    else:
        r2 = float(r2_score(y_true, y_pred))
        rho = float(spearmanr(y_true, y_pred).correlation)
    target_std = float(np.std(y_true))
    smae = float(np.mean(np.abs(y_pred - y_true)) / target_std) if target_std > 0 else float("nan")
    return {"r2": r2, "spearman": rho, "standardized_mae": smae}


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    return metric_values_from_arrays(
        frame["y_true"].to_numpy(dtype=float),
        frame["y_pred"].to_numpy(dtype=float),
    )


def patient_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    n_iter: int = 1000,
    seed: int = 2026,
) -> dict[str, Any]:
    rng = np.random.RandomState(seed)
    patient_ids = frame["patient_id"].astype(str).to_numpy()
    patients = np.unique(patient_ids)
    patient_index = [np.flatnonzero(patient_ids == pid) for pid in patients]
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    observed = metric_values_from_arrays(y_true, y_pred)
    boot: dict[str, list[float]] = {key: [] for key in observed}
    for _ in range(n_iter):
        sampled = rng.randint(0, len(patients), size=len(patients))
        idx = np.concatenate([patient_index[i] for i in sampled])
        vals = metric_values_from_arrays(y_true[idx], y_pred[idx])
        for key, value in vals.items():
            if math.isfinite(value):
                boot[key].append(float(value))

    out: dict[str, Any] = {}
    for key, value in observed.items():
        values = np.asarray(boot[key], dtype=float)
        out[key] = {
            "value": float(value),
            "ci_lower": float(np.percentile(values, 2.5)) if len(values) else float("nan"),
            "ci_upper": float(np.percentile(values, 97.5)) if len(values) else float("nan"),
        }
    return out


def spearman_delta_values(y_true: np.ndarray, pred_first: np.ndarray, pred_second: np.ndarray) -> float:
    if len(y_true) < 2 or float(np.var(y_true)) <= 1e-12:
        return float("nan")
    first = float(spearmanr(y_true, pred_first).correlation)
    second = float(spearmanr(y_true, pred_second).correlation)
    return first - second


def patient_cluster_spearman_delta(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    first_name: str,
    second_name: str,
    n_iter: int = 1000,
    seed: int = 2026,
) -> dict[str, Any]:
    left = first[["image_path", "patient_id", "target", "y_true", "y_pred"]].rename(
        columns={"y_pred": f"y_pred_{first_name}", "y_true": "y_true_first"}
    )
    right = second[["image_path", "patient_id", "target", "y_true", "y_pred"]].rename(
        columns={"y_pred": f"y_pred_{second_name}", "y_true": "y_true_second"}
    )
    aligned = left.merge(right, on=["image_path", "patient_id", "target"], validate="one_to_one")
    if not np.allclose(aligned["y_true_first"], aligned["y_true_second"]):
        raise RuntimeError("Recoverability target values differ between aligned conditions")

    y_true = aligned["y_true_first"].to_numpy(dtype=float)
    pred_first = aligned[f"y_pred_{first_name}"].to_numpy(dtype=float)
    pred_second = aligned[f"y_pred_{second_name}"].to_numpy(dtype=float)
    observed = spearman_delta_values(y_true, pred_first, pred_second)
    rng = np.random.RandomState(seed)
    patient_ids = aligned["patient_id"].astype(str).to_numpy()
    patients = np.unique(patient_ids)
    patient_index = [np.flatnonzero(patient_ids == pid) for pid in patients]
    boot: list[float] = []
    for _ in range(n_iter):
        sampled = rng.randint(0, len(patients), size=len(patients))
        idx = np.concatenate([patient_index[i] for i in sampled])
        value = spearman_delta_values(y_true[idx], pred_first[idx], pred_second[idx])
        if math.isfinite(value):
            boot.append(float(value))
    values = np.asarray(boot, dtype=float)
    return {
        "convention": f"{first_name}_minus_{second_name}",
        "delta_spearman": float(observed),
        "ci_lower": float(np.percentile(values, 2.5)) if len(values) else float("nan"),
        "ci_upper": float(np.percentile(values, 97.5)) if len(values) else float("nan"),
        "n_images": int(aligned["image_path"].nunique()),
        "n_patients": int(aligned["patient_id"].nunique()),
    }


def evaluate_condition(condition: str) -> tuple[dict[str, Any], pd.DataFrame]:
    X, table = load_feature_table(condition)
    target_predictions: list[pd.DataFrame] = []
    fold_metadata: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for target in FEATURE_NAMES:
        predictions, metadata = fit_predict_target(X, table, target)
        target_predictions.append(predictions)
        fold_metadata.extend(metadata)
        summaries[target] = patient_cluster_bootstrap(predictions)

    predictions = pd.concat(target_predictions, ignore_index=True)
    core = {target: summaries[target] for target in CORE_TARGETS}
    manuscript_targets = {target: summaries[target] for target in MANUSCRIPT_TARGETS}
    summary = {
        "condition": condition,
        "display_name": CONDITION_LABELS.get(condition, condition),
        "n_images": int(len(table)),
        "n_patients": int(table["patient_id"].nunique()),
        "targets": summaries,
        "core_targets": core,
        "manuscript_targets": manuscript_targets,
        "median_core_spearman": float(np.median([
            stats["spearman"]["value"] for stats in core.values()
        ])),
        "median_core_r2": float(np.median([
            stats["r2"]["value"] for stats in core.values()
        ])),
        "median_nonredundant_target_spearman": float(np.nanmedian([
            stats["spearman"]["value"] for stats in manuscript_targets.values()
        ])),
        "median_nonredundant_target_r2": float(np.nanmedian([
            stats["r2"]["value"] for stats in manuscript_targets.values()
        ])),
        "median_all_target_spearman": float(np.nanmedian([
            stats["spearman"]["value"] for stats in summaries.values()
        ])),
        "median_all_target_r2": float(np.nanmedian([
            stats["r2"]["value"] for stats in summaries.values()
        ])),
        "fold_metadata": fold_metadata,
    }
    return summary, predictions


def _fmt_ci(metric: dict[str, float]) -> str:
    return (
        f"{metric['value']:.3f} "
        f"[{metric['ci_lower']:.3f}, {metric['ci_upper']:.3f}]"
    )


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Geometry recoverability from DINOv3 representations",
        "",
        "Ridge regressors predict dilated-mask geometry from frozen image representations using patient-disjoint folds. Intervals are patient-cluster bootstrap 95% CIs.",
        "",
        "The manuscript-facing summary uses eight non-redundant dilated-mask targets, excluding the raw mask-to-image area ratio because it is redundant with the square-root area-ratio target. The full nine-target audit trace is preserved separately.",
        "",
        CORE_TARGET_RATIONALE,
        "",
        "Fixed-shape edits approximately match dilated-mask area and center before rasterization; their recoverability values are procedural checks rather than new evidence of learned recovery.",
        "",
        "| Representation | Median 8-target Spearman | Median core Spearman | Median core R2 | Core target details |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for condition, stats in report["conditions"].items():
        details = []
        for target, target_stats in stats["core_targets"].items():
            rho = target_stats["spearman"]["value"]
            r2 = target_stats["r2"]["value"]
            details.append(f"{target}: rho={rho:.3f}, R2={r2:.3f}")
        lines.append(
            f"| {stats['display_name']} | {stats['median_nonredundant_target_spearman']:.3f} | "
            f"{stats['median_core_spearman']:.3f} | "
            f"{stats['median_core_r2']:.3f} | {'; '.join(details)} |"
        )
    lines.extend(["", "## Claim gate", "", report["claim_gate"]])
    lines.extend(
        [
            "",
            "## C4 versus full-image recoverability contrast",
            "",
            "| Target | Delta Spearman (C4 - Full) | 95% CI |",
            "| --- | ---: | ---: |",
        ]
    )
    for target, stats in report["c4_vs_full_delta_spearman"].items():
        lines.append(
            f"| {target} | {stats['delta_spearman']:+.3f} | "
            f"[{stats['ci_lower']:+.3f}, {stats['ci_upper']:+.3f}] |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")




def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    summaries: dict[str, Any] = {}
    prediction_frames: list[pd.DataFrame] = []
    for condition in CONDITIONS:
        summary, predictions = evaluate_condition(condition)
        summaries[condition] = summary
        prediction_frames.append(predictions)

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    c4_vs_full: dict[str, Any] = {}
    for target in CORE_TARGETS:
        first = all_predictions[
            all_predictions["condition"].eq("C4_lesion-removed-inpaint")
            & all_predictions["target"].eq(target)
        ]
        second = all_predictions[
            all_predictions["condition"].eq("C1_full")
            & all_predictions["target"].eq(target)
        ]
        c4_vs_full[target] = patient_cluster_spearman_delta(
            first,
            second,
            first_name="C4",
            second_name="Full",
        )

    c4 = summaries["C4_lesion-removed-inpaint"]
    c4_core = c4["core_targets"]
    recoverable = sum(
        1
        for stats in c4_core.values()
        if stats["spearman"]["value"] >= 0.50 or stats["r2"]["value"] >= 0.25
    )
    if recoverable >= 3:
        claim_gate = (
            "C4 representations recover multiple core dilated-mask geometry features; "
            "the manuscript may state that geometry information is recoverable from "
            "C4 representations while preserving source-ambiguity wording."
        )
    else:
        claim_gate = (
            "C4 recoverability is weak for core geometry features; the manuscript "
            "should retain source-ambiguity wording without claiming direct "
            "geometry recoverability from C4."
        )

    report = {
        "source": "BIBM geometry recoverability audit",
        "model": MODEL,
        "dataset": DATASET,
        "feature_source": "frozen DINOv3 condition representations",
        "target_source": "dilated intervention-mask geometry used by C4",
        "targets": FEATURE_NAMES,
        "manuscript_targets": MANUSCRIPT_TARGETS,
        "core_targets": CORE_TARGETS,
        "core_target_rationale": CORE_TARGET_RATIONALE,
        "regressor": "Ridge with StandardScaler on features and targets inside the training fold; alpha in 10^-3..10^3 selected by maximizing validation R2 across group-aware inner CV independently for each target/fold",
        "metrics": "patient-cluster bootstrap R2, Spearman correlation, standardized MAE",
        "fixed_shape_note": "Fixed-shape edits approximately match dilated-mask area and center before rasterization; those recoverability values are procedural checks.",
        "conditions": summaries,
        "c4_vs_full_delta_spearman": c4_vs_full,
        "claim_gate": claim_gate,
    }
    json_path = RESULTS_DIR / "bibm_geometry_recoverability.json"
    md_path = RESULTS_DIR / "bibm_geometry_recoverability.md"
    csv_path = RESULTS_DIR / "bibm_geometry_recoverability_predictions.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    all_predictions.to_csv(csv_path, index=False)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
