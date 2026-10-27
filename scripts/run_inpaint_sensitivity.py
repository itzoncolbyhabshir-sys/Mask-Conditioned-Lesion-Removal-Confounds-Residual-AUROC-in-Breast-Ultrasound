"""Run the registered Telea/Navier-Stokes C4 sensitivity analysis only."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import SEEDS, IMAGES_DIR
from src.conditions import load_image, load_mask, inpaint_telea, inpaint_navier_stokes
from src.features import run_extraction
from src.probe import run_probe_experiment
from src.statistics import load_predictions, paired_bootstrap_difference


def align_patient_predictions(
    predictions_full: pd.DataFrame,
    predictions_variant: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate and deterministically align patient predictions one-to-one."""
    required = {"patient_id", "y_true", "y_prob"}
    for name, frame in (("full", predictions_full), ("variant", predictions_variant)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} predictions missing columns: {sorted(missing)}")

    def aggregate(frame: pd.DataFrame, probability_name: str) -> pd.DataFrame:
        work = frame.copy()
        work["patient_key"] = work["patient_id"].astype(str)
        labels_per_patient = work.groupby("patient_key", sort=False)["y_true"].nunique()
        if (labels_per_patient > 1).any():
            raise ValueError("A patient has conflicting labels across predictions")
        return (
            work.groupby("patient_key", sort=False)
            .agg(patient_id=("patient_id", "first"), y_true=("y_true", "first"),
                 **{probability_name: ("y_prob", "mean")})
            .reset_index()
        )

    avg_full = aggregate(predictions_full, "y_prob_full")
    avg_variant = aggregate(predictions_variant, "y_prob_variant")
    paired = avg_full.merge(
        avg_variant[["patient_key", "y_true", "y_prob_variant"]],
        on="patient_key",
        how="inner",
        suffixes=("_full", "_variant_label"),
        validate="one_to_one",
        sort=True,
    )
    if "y_true_variant_label" in paired and not np.array_equal(
        paired["y_true_full"].to_numpy(), paired["y_true_variant_label"].to_numpy()
    ):
        raise ValueError("Full and variant predictions disagree on patient labels")
    if "y_true_full" in paired:
        paired = paired.rename(columns={"y_true_full": "y_true"})
        paired = paired.drop(columns=["y_true_variant_label"])
    return paired.sort_values("patient_key", kind="stable").reset_index(drop=True)


def generate_c4_inpaint_variants(
    manifest: pd.DataFrame,
    dataset_name: str,
    methods: list[str],
) -> list[str]:
    """Generate the two validated classical C4 sensitivity conditions."""
    method_map = {
        "telea": ("C4_Telea", inpaint_telea),
        "navier-stokes": ("C4_Navier-Stokes", inpaint_navier_stokes),
    }
    selected = []
    normalized_methods = [str(method).lower().replace("_", "-") for method in methods]
    ds_df = manifest[manifest["dataset_name"] == dataset_name]

    for normalized in normalized_methods:
        if normalized not in method_map:
            print(
                f"[SKIP] {normalized}: unsupported operator; "
                "no output will be created."
            )
            continue
        condition, inpaint_fn = method_map[normalized]
        output_dir = IMAGES_DIR / dataset_name / condition
        output_dir.mkdir(parents=True, exist_ok=True)
        used_names = set()
        for _, row in tqdm(
            ds_df.iterrows(), total=len(ds_df), desc=f"{condition} gen [{dataset_name}]"
        ):
            image_path = Path(row["image_path"])
            output_name = image_path.name
            if output_name in used_names:
                raise ValueError(f"Duplicate output stem for {dataset_name}: {output_name}")
            used_names.add(output_name)
            image = load_image(image_path, grayscale=True)
            mask_value = row.get("dilated_mask_path", row.get("mask_path", ""))
            mask_path = Path(str(mask_value)) if pd.notna(mask_value) and mask_value else None
            if mask_path is None or not mask_path.exists():
                raise FileNotFoundError(f"Missing lesion mask for {image_path}")
            mask = load_mask(mask_path)
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(
                    mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
                )
            Image.fromarray(inpaint_fn(image, mask)).save(output_dir / output_name)
        selected.append(condition)
    return selected


def main() -> dict:
    dataset = "BUS-BRA"
    model = "DINOv3"
    primary = "C4_lesion-removed-inpaint"
    manifest = pd.read_csv(ROOT / "manifest.csv")
    conditions = generate_c4_inpaint_variants(
        manifest, dataset, ["Telea", "Navier-Stokes"]
    )
    results = {}
    primary_predictions = load_predictions(model, dataset, primary)
    if primary_predictions.empty:
        raise RuntimeError("Primary C4 predictions are missing")

    for condition in conditions:
        run_extraction(model, dataset, condition, batch_size=4, num_workers=0)
        run_probe_experiment(model, dataset, condition, seeds=SEEDS)
        variant_predictions = load_predictions(model, dataset, condition)
        paired = align_patient_predictions(primary_predictions, variant_predictions)
        stats = paired_bootstrap_difference(
            y_true=paired["y_true"].to_numpy(),
            y_prob_1=paired["y_prob_full"].to_numpy(),
            y_prob_2=paired["y_prob_variant"].to_numpy(),
            patient_ids=paired["patient_key"].to_numpy(),
        )
        results[condition] = {
            "delta_primary_minus_variant": stats["delta_auroc"],
            "ci_lower": stats["ci_lower"],
            "ci_upper": stats["ci_upper"],
            "p_value": stats["p_value"],
        }

    output = ROOT / "results" / "inpaint_operator_sensitivity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"[SAVE] {output}")
    return results


if __name__ == "__main__":
    main()
