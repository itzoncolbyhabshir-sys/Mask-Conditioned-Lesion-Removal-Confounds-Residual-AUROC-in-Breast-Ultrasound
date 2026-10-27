"""Intervention-aligned dilated-mask controls for the BIBM manuscript.

The lesion-removed endpoint (C4) is generated from a dilated intervention
mask. This script keeps the control evidence aligned to that operator by:

1. generating C5d/C6d from ``dilated_mask_path``;
2. running the lightweight model/dataset pairs requested for the manuscript;
3. training an explicit nine-feature geometry baseline from the same dilated
   masks; and
4. reporting all deltas as ``control - C4``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_bibm_geometry_feature_baseline import (  # noqa: E402
    FEATURE_NAMES,
    mask_geometry_features,
)
from src.config import FEATURES_DIR, IMAGES_DIR, N_SPLITS, PROBE_CONFIG, RESULTS_DIR, SEEDS  # noqa: E402
from src.conditions import (  # noqa: E402
    condition_output_name,
    generate_C5_mask_only,
    generate_C6_shape_only,
    load_image,
    load_mask,
)
from src.features import run_extraction  # noqa: E402
from src.probe import run_probe_experiment, train_probe  # noqa: E402
from src.statistics import (  # noqa: E402
    aggregate_patient_predictions,
    align_patient_predictions,
    bootstrap_auroc_ci,
    load_predictions,
    paired_bootstrap_difference,
    permutation_test,
)


C4_CONDITION = "C4_lesion-removed-inpaint"
DILATED_IMAGE_CONDITIONS = {
    "C5d_dilated-mask-only": "dilated_mask_only",
    "C6d_dilated-geometry-rendering": "dilated_geometry_rendering",
}
DEFAULT_SETTINGS = [
    {"model": "DINOv3", "dataset": "BUS-BRA", "role": "primary_intervention_aligned"},
    {"model": "MedSAM2", "dataset": "BUS-BRA", "role": "encoder_sensitivity"},
    {"model": "DINOv3", "dataset": "BUS-UCLM", "role": "dataset_sensitivity"},
]


def setting_key(model: str, dataset: str) -> str:
    return f"{model}/{dataset}"


def manifest_for_dataset(dataset: str) -> pd.DataFrame:
    manifest = pd.read_csv(ROOT / "manifest.csv")
    manifest = manifest[manifest["dataset_name"].eq(dataset)].copy()
    if manifest.empty:
        raise RuntimeError(f"No manifest rows found for dataset {dataset!r}")
    missing = manifest["dilated_mask_path"].isna() | manifest["dilated_mask_path"].eq("")
    if missing.any():
        raise RuntimeError(
            f"{int(missing.sum())} {dataset} rows are missing dilated_mask_path; "
            "dilated intervention-mask controls cannot be generated safely."
        )
    return manifest


def duplicate_stems(manifest: pd.DataFrame) -> set[str]:
    if manifest.empty:
        return set()
    tmp = manifest.assign(__stem=manifest["image_path"].map(lambda p: Path(str(p)).stem))
    duplicated = tmp.duplicated(["dataset_name", "__stem"], keep=False)
    return set(tmp.loc[duplicated, "__stem"])


def generate_dilated_condition_images(dataset: str) -> dict[str, int]:
    manifest = manifest_for_dataset(dataset)
    dupes = duplicate_stems(manifest)
    counts = {condition: 0 for condition in DILATED_IMAGE_CONDITIONS}
    for condition in DILATED_IMAGE_CONDITIONS:
        (IMAGES_DIR / dataset / condition).mkdir(parents=True, exist_ok=True)

    for _, row in manifest.iterrows():
        image = load_image(Path(str(row["image_path"])), grayscale=True)
        dilated_mask = load_mask(Path(str(row["dilated_mask_path"])))
        output_name = condition_output_name(
            row,
            disambiguate=Path(str(row["image_path"])).stem in dupes,
        )
        generated = {
            "C5d_dilated-mask-only": generate_C5_mask_only(dilated_mask),
            "C6d_dilated-geometry-rendering": generate_C6_shape_only(dilated_mask, image.shape),
        }
        for condition, array in generated.items():
            out_path = IMAGES_DIR / dataset / condition / output_name
            Image.fromarray(array).save(out_path)
            counts[condition] += 1
    return counts


def ensure_reference_outputs(
    model: str,
    dataset: str,
    *,
    batch_size: int,
    device: str,
    offline: bool,
) -> None:
    """Ensure C4 predictions exist for the setting used as the delta reference."""
    if load_predictions(model, dataset, C4_CONDITION).empty:
        feat_path = FEATURES_DIR / model / dataset / f"{C4_CONDITION}.npy"
        if not feat_path.exists():
            run_extraction(
                model_name=model,
                dataset_name=dataset,
                condition=C4_CONDITION,
                batch_size=batch_size,
                device_str=device,
                num_workers=0,
                force=False,
                offline=offline,
            )
        run_probe_experiment(model, dataset, C4_CONDITION, seeds=SEEDS)
    if load_predictions(model, dataset, C4_CONDITION).empty:
        raise RuntimeError(f"Missing C4 reference predictions for {model}/{dataset}")


def ensure_image_probe_outputs(
    model: str,
    dataset: str,
    *,
    batch_size: int,
    device: str,
    offline: bool,
) -> None:
    for condition in DILATED_IMAGE_CONDITIONS:
        run_extraction(
            model_name=model,
            dataset_name=dataset,
            condition=condition,
            batch_size=batch_size,
            device_str=device,
            num_workers=0,
            force=False,
            offline=offline,
        )
        feat_path = FEATURES_DIR / model / dataset / f"{condition}.npy"
        if not feat_path.exists():
            raise RuntimeError(f"Feature extraction did not create {feat_path}")
        run_probe_experiment(model, dataset, condition, seeds=SEEDS)


def summarize_patient_predictions(patient_predictions: pd.DataFrame) -> dict[str, Any]:
    y_true = patient_predictions["y_true"].to_numpy(dtype=int)
    y_prob = patient_predictions["y_prob"].to_numpy(dtype=float)
    patient_ids = patient_predictions["patient_id"].astype(str).to_numpy()
    boot = bootstrap_auroc_ci(y_true, y_prob, patient_ids)
    perm = permutation_test(y_true, y_prob, patient_ids)
    return {
        "n_patients": int(len(patient_predictions)),
        "n_benign_patients": int((patient_predictions["y_true"] == 0).sum()),
        "n_malignant_patients": int((patient_predictions["y_true"] == 1).sum()),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "ci_lower": float(boot["ci_lower"]),
        "ci_upper": float(boot["ci_upper"]),
        "p_value": float(perm["p_value"]),
    }


def delta_against_c4(patient_predictions: pd.DataFrame, model: str, dataset: str) -> dict[str, Any]:
    reference = aggregate_patient_predictions(load_predictions(model, dataset, C4_CONDITION))
    aligned = align_patient_predictions(
        patient_predictions,
        reference,
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
        "convention": "control_minus_C4",
        "delta_auroc": float(paired["delta_auroc"]),
        "ci_lower": float(paired["ci_lower"]),
        "ci_upper": float(paired["ci_upper"]),
        "p_value": float(paired["p_value"]),
    }


def image_condition_summary(model: str, dataset: str, condition: str) -> dict[str, Any]:
    predictions = load_predictions(model, dataset, condition)
    if predictions.empty:
        raise RuntimeError(f"Missing predictions for {model}/{dataset}/{condition}")
    patient_predictions = aggregate_patient_predictions(predictions)
    summary = summarize_patient_predictions(patient_predictions)
    summary["delta_vs_c4"] = delta_against_c4(patient_predictions, model, dataset)
    return summary


def build_dilated_geometry_feature_table(dataset: str) -> pd.DataFrame:
    manifest = manifest_for_dataset(dataset)
    rows: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        features = mask_geometry_features(row.dilated_mask_path)
        out = {
            "image_path": str(row.image_path),
            "patient_id": str(row.patient_id),
            "label": str(row.label),
            "split_id": int(row.split_id),
        }
        out.update(dict(zip(FEATURE_NAMES, features)))
        rows.append(out)
    return pd.DataFrame(rows)


def fit_dilated_geometry_features(dataset: str) -> tuple[dict[str, Any], pd.DataFrame]:
    table = build_dilated_geometry_feature_table(dataset)
    X = table[FEATURE_NAMES].to_numpy(dtype=float)
    y = table["label"].eq("malignant").astype(int).to_numpy()
    groups = table["patient_id"].astype(str).to_numpy()
    splits = table["split_id"].astype(int).to_numpy()
    all_predictions: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []

    for fold in range(N_SPLITS):
        train_mask = splits != fold
        test_mask = splits == fold
        if not train_mask.any() or not test_mask.any():
            continue
        if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[test_mask])) < 2:
            continue
        for seed in SEEDS:
            clf, scaler = train_probe(
                X[train_mask],
                y[train_mask],
                groups[train_mask],
                C_grid=PROBE_CONFIG["C_grid"],
                inner_cv=PROBE_CONFIG["inner_cv"],
                class_weight=PROBE_CONFIG["class_weight"],
                max_iter=PROBE_CONFIG["max_iter"],
                random_state=seed,
            )
            y_prob = clf.predict_proba(scaler.transform(X[test_mask]))[:, 1]
            pred = table.loc[test_mask, ["image_path", "patient_id"]].copy()
            pred["y_true"] = y[test_mask]
            pred["y_prob"] = y_prob
            pred["fold"] = fold
            pred["seed"] = seed
            pred["condition"] = "dilated_mask_geometry_features"
            pred["dataset"] = dataset
            all_predictions.append(pred)
            patient = aggregate_patient_predictions(pred)
            if len(np.unique(patient["y_true"])) == 2:
                fold_summaries.append(
                    {
                        "fold": int(fold),
                        "seed": int(seed),
                        "n_test_patients": int(len(patient)),
                        "auroc": float(roc_auc_score(patient["y_true"], patient["y_prob"])),
                    }
                )

    if not all_predictions:
        raise RuntimeError(f"No valid predictions were produced for {dataset} dilated geometry features")

    predictions = pd.concat(all_predictions, ignore_index=True)
    patient_predictions = aggregate_patient_predictions(predictions)
    base_summary = summarize_patient_predictions(patient_predictions)
    base_summary.update(
        {
            "model": "dilated_intervention_mask_geometry_logistic_probe",
            "feature_names": FEATURE_NAMES,
            "fold_seed_aurocs": fold_summaries,
        }
    )
    return base_summary, predictions


def geometry_summary_for_setting(
    base_summary: dict[str, Any],
    predictions: pd.DataFrame,
    model: str,
    dataset: str,
) -> dict[str, Any]:
    patient_predictions = aggregate_patient_predictions(predictions)
    summary = dict(base_summary)
    summary["delta_vs_c4"] = delta_against_c4(patient_predictions, model, dataset)
    # Backward-compatible key used by previous figure/manuscript code.
    summary["delta_vs_dinov3_c4"] = summary["delta_vs_c4"]
    return summary


def original_mask_reference(dataset: str, model: str) -> dict[str, Any]:
    reference: dict[str, Any] = {}
    for condition in ("C5_mask-only", "C6_shape-only"):
        predictions = load_predictions(model, dataset, condition)
        if predictions.empty:
            continue
        reference[condition] = summarize_patient_predictions(aggregate_patient_predictions(predictions))
    geometry_path = RESULTS_DIR / "bibm_geometry_feature_baseline.json"
    if geometry_path.exists():
        report = json.loads(geometry_path.read_text(encoding="utf-8"))
        if dataset in report.get("datasets", {}):
            reference["original_mask_geometry_features"] = report["datasets"][dataset]
    return reference


def interpretation_gate(primary: dict[str, Any]) -> str:
    all_dilated_aurocs = [
        stats["auroc"] for stats in primary["image_controls"].values()
    ] + [primary["explicit_dilated_geometry"]["auroc"]]
    if min(all_dilated_aurocs) >= 0.85:
        return (
            "Dilated intervention-mask controls remain highly label-decodable in the "
            "primary BUS-BRA/DINOv3 setting, supporting the bounded claim that "
            "intervention-mask geometry is predictive in isolation."
        )
    return (
        "At least one dilated intervention-mask control is substantially weaker, "
        "so the manuscript should frame direct intervention-mask geometry evidence "
        "as limited while retaining the source-ambiguity conclusion."
    )


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Dilated intervention-mask controls",
        "",
        "These controls use the same dilated intervention-mask family used to generate C4, rather than relying only on the original lesion mask.",
        "",
        "| Setting | Control | AUROC [95% CI] | Delta vs C4 |",
        "| --- | --- | --- | --- |",
    ]
    for key, setting in report["settings"].items():
        for condition, stats in setting["image_controls"].items():
            delta = stats["delta_vs_c4"]
            lines.append(
                f"| {key} | {condition} | {stats['auroc']:.3f} "
                f"[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}] | "
                f"{delta['delta_auroc']:+.3f} "
                f"[{delta['ci_lower']:+.3f}, {delta['ci_upper']:+.3f}] |"
            )
        geom = setting["explicit_dilated_geometry"]
        delta = geom["delta_vs_c4"]
        lines.append(
            f"| {key} | dilated geometry features | {geom['auroc']:.3f} "
            f"[{geom['ci_lower']:.3f}, {geom['ci_upper']:.3f}] | "
            f"{delta['delta_auroc']:+.3f} "
            f"[{delta['ci_lower']:+.3f}, {delta['ci_upper']:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation gate",
            "",
            report["interpretation_gate"],
            "",
            "## Features",
            "",
            ", ".join(FEATURE_NAMES),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_settings(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.settings:
        settings = []
        for item in args.settings:
            if "/" not in item:
                raise ValueError("--settings entries must use MODEL/DATASET")
            model, dataset = item.split("/", 1)
            settings.append({"model": model, "dataset": dataset, "role": "custom"})
        return settings
    if args.model or args.dataset:
        return [
            {
                "model": args.model or "DINOv3",
                "dataset": args.dataset or "BUS-BRA",
                "role": "custom_single_setting",
            }
        ]
    return [dict(item) for item in DEFAULT_SETTINGS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BIBM dilated intervention-mask controls")
    parser.add_argument("--settings", nargs="*", help="Optional MODEL/DATASET setting list")
    parser.add_argument("--dataset", default=None, help="Backward-compatible single dataset override")
    parser.add_argument("--model", default=None, help="Backward-compatible single model override")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offline", action="store_true", help="Use local model caches only")
    args = parser.parse_args()

    settings = parse_settings(args)
    generated_counts: dict[str, dict[str, int]] = {}
    geometry_by_dataset: dict[str, tuple[dict[str, Any], pd.DataFrame]] = {}
    prediction_frames: list[pd.DataFrame] = []

    for dataset in sorted({item["dataset"] for item in settings}):
        generated_counts[dataset] = generate_dilated_condition_images(dataset)
        geometry_by_dataset[dataset] = fit_dilated_geometry_features(dataset)
        geometry_predictions = geometry_by_dataset[dataset][1].copy()
        prediction_frames.append(geometry_predictions)

    report_settings: dict[str, Any] = {}
    for item in settings:
        model = item["model"]
        dataset = item["dataset"]
        key = setting_key(model, dataset)
        ensure_reference_outputs(
            model,
            dataset,
            batch_size=args.batch_size,
            device=args.device,
            offline=args.offline,
        )
        ensure_image_probe_outputs(
            model,
            dataset,
            batch_size=args.batch_size,
            device=args.device,
            offline=args.offline,
        )
        base_geometry_summary, geometry_predictions = geometry_by_dataset[dataset]
        image_controls = {
            condition: image_condition_summary(model, dataset, condition)
            for condition in DILATED_IMAGE_CONDITIONS
        }
        explicit_geometry = geometry_summary_for_setting(
            base_geometry_summary,
            geometry_predictions,
            model,
            dataset,
        )
        c4_stats = summarize_patient_predictions(
            aggregate_patient_predictions(load_predictions(model, dataset, C4_CONDITION))
        )
        report_settings[key] = {
            "model": model,
            "dataset": dataset,
            "role": item.get("role", "custom"),
            "c4_reference": c4_stats,
            "image_controls": image_controls,
            "explicit_dilated_geometry": explicit_geometry,
            "original_mask_reference": original_mask_reference(dataset, model),
        }

    primary_key = "DINOv3/BUS-BRA" if "DINOv3/BUS-BRA" in report_settings else next(iter(report_settings))
    primary = report_settings[primary_key]
    report = {
        "source": "BIBM dilated intervention-mask control audit",
        "settings": report_settings,
        "generated_counts": generated_counts,
        "mask_source": "dilated intervention mask used by C4 lesion removal",
        "normalization": (
            "area is normalized by image area, perimeter by sqrt(image area), "
            "and centroid coordinates by image width/height; standardization is "
            "fitted inside each training fold"
        ),
        "interpretation_gate": interpretation_gate(primary),
        # Backward-compatible primary fields.
        "dataset": primary["dataset"],
        "model": primary["model"],
        "c4_reference": primary["c4_reference"],
        "image_controls": primary["image_controls"],
        "explicit_dilated_geometry": primary["explicit_dilated_geometry"],
        "original_mask_reference": primary["original_mask_reference"],
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / "bibm_dilated_mask_controls.json"
    md_path = RESULTS_DIR / "bibm_dilated_mask_controls.md"
    pred_path = RESULTS_DIR / "bibm_dilated_geometry_feature_predictions.csv"
    combined_predictions = []
    for dataset, (_, preds) in geometry_by_dataset.items():
        frame = preds.copy()
        frame["geometry_dataset"] = dataset
        combined_predictions.append(frame)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    pd.concat(combined_predictions, ignore_index=True).to_csv(pred_path, index=False)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {pred_path}")


if __name__ == "__main__":
    main()
