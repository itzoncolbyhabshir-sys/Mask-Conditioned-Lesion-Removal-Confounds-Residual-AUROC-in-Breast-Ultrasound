"""Generate the seven main-text image conditions; fixed shapes use a separate script.

Outputs: images/{dataset}/{condition}/. Run: python -m src.conditions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm
import random
import hashlib
import re

from src.config import (
    ROOT, IMAGES_DIR, MASK_CONFIG, INPAINT_CONFIG, CONDITIONS,
)

PROCESSED_MASK_DIR = ROOT / "data" / "processed_masks"


def load_image(path: Path, grayscale: bool = True) -> np.ndarray:
    img = Image.open(path)
    if grayscale:
        img = img.convert("L")
    return np.array(img)


def load_mask(path: Path) -> np.ndarray:
    """Load binary mask."""
    img = np.array(Image.open(path))
    if img.ndim == 3:
        img = img.max(axis=2)
    return (img > 127).astype(np.uint8)


def condition_output_name(row: pd.Series, disambiguate: bool = False) -> str:
    """Preserve normal stems and add a stable suffix only on collisions."""
    source = str(Path(str(row["image_path"])).resolve())
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(source).stem).strip("_") or "image"
    if disambiguate:
        digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
        stem = f"{stem}__{digest}"
    return f"{stem}.png"


def inpaint_telea(image: np.ndarray, mask: np.ndarray, radius: int = None) -> np.ndarray:
    """
    Inpaint using Alexandru Telea's method.
    image: uint8 grayscale or BGR
    mask: uint8 binary, non-zero = region to inpaint
    """
    if radius is None:
        radius = INPAINT_CONFIG["telea_radius"]
    return cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)


def inpaint_navier_stokes(image: np.ndarray, mask: np.ndarray, radius: int = None) -> np.ndarray:
    """Inpaint using Navier-Stokes method."""
    if radius is None:
        radius = INPAINT_CONFIG["ns_radius"]
    return cv2.inpaint(image, mask, radius, cv2.INPAINT_NS)


def generate_C1_full(image: np.ndarray) -> np.ndarray:
    """C1: Original grayscale image."""
    return image.copy()


def generate_C2_lesion_only(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """C2: Keep only lesion region, zero out background."""
    result = np.zeros_like(image)
    result[mask > 0] = image[mask > 0]
    return result




def generate_C4_lesion_removed_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    method: str = "telea",
    mask_is_dilated: bool = False,
) -> np.ndarray:
    """
    C4: Inpaint lesion region — PRIMARY ENDPOINT.
    Use dilated mask to ensure full lesion coverage.
    """
    if mask_is_dilated:
        dilated_mask = mask
    else:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MASK_CONFIG["dilation_kernel"], MASK_CONFIG["dilation_kernel"])
        )
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    if method == "telea":
        return inpaint_telea(image, dilated_mask)
    elif method == "ns":
        return inpaint_navier_stokes(image, dilated_mask)
    else:
        return inpaint_telea(image, dilated_mask)


def generate_C5_mask_only(mask: np.ndarray) -> np.ndarray:
    """C5: Binary mask as image (control: geometric information only)."""
    return (mask * 255).astype(np.uint8)


def generate_C6_shape_only(mask: np.ndarray, image_shape: tuple) -> np.ndarray:
    """
    C6: Shape descriptors as a feature vector visualization (control).
    Actually returns the shape features as a fixed-length vector representation;
    for image output, we create a blank canvas encoding shape properties.
    """
    if mask.sum() == 0:
        return np.zeros(image_shape, dtype=np.uint8)

    # Create shape visualization: filled lesion on blank canvas
    result = np.zeros(image_shape, dtype=np.uint8)
    result[mask > 0] = 128  # mid-gray lesion

    # Add bounding box
    ys, xs = np.where(mask > 0)
    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max(), ys.max()
    cv2.rectangle(result, (x1, y1), (x2, y2), 255, 2)

    # Draw centroid
    cx = int(xs.mean())
    cy = int(ys.mean())
    cv2.circle(result, (cx, cy), 3, 255, -1)

    return result










def generate_conditions_for_record(
    row: pd.Series,
    output_dirs: dict,
    dilated_mask_path: str = None,
    output_name: str = None,
) -> dict:
    """
    Generate all condition images for one manifest record.
    Returns dict of output paths per condition.
    """
    img_path = Path(row["image_path"])
    if not img_path.exists():
        return {}

    image = load_image(img_path, grayscale=True)

    # Load mask
    mask_path = Path(row["mask_path"]) if pd.notna(row["mask_path"]) and row["mask_path"] else None
    if mask_path and mask_path.exists():
        mask = load_mask(mask_path)
    else:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Load dilated mask
    dilated_mask = mask
    if dilated_mask_path and Path(dilated_mask_path).exists():
        dilated_mask = load_mask(Path(dilated_mask_path))

    img_name = output_name or condition_output_name(row)
    outputs = {}

    condition_generators = {
        "C1_full": lambda: generate_C1_full(image),
        "C2_lesion-only": lambda: generate_C2_lesion_only(image, mask),
        "C4_lesion-removed-inpaint": lambda: generate_C4_lesion_removed_inpaint(
            image, dilated_mask, mask_is_dilated=dilated_mask_path is not None and Path(dilated_mask_path).exists()
        ),
        "C5_mask-only": lambda: generate_C5_mask_only(mask),
        "C6_shape-only": lambda: generate_C6_shape_only(mask, image.shape),
        "C5d_dilated-mask-only": lambda: generate_C5_mask_only(dilated_mask),
        "C6d_dilated-geometry-rendering": lambda: generate_C6_shape_only(dilated_mask, image.shape),
    }

    unknown = sorted(set(output_dirs) - set(condition_generators))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")

    for cond_name in output_dirs:
        generator = condition_generators[cond_name]
        try:
            result = generator()
            out_path = output_dirs[cond_name] / img_name
            Image.fromarray(result).save(out_path)
            outputs[cond_name] = str(out_path)
        except Exception as e:
            print(f"[WARN] {cond_name} generation failed for {img_path.name}: {e}")
            outputs[cond_name] = ""

    return outputs


def main(datasets: list = None, conditions: list = None, n_spotcheck: int = 20):
    """
    Generate all conditions for all records in manifest.
    """
    manifest_path = ROOT / "manifest.csv"
    if not manifest_path.exists():
        print("[ERROR] manifest.csv not found. Run 00_manifest.py first.")
        return

    print("="*60)
    print("02_conditions.py — Condition Image Generation")
    print("="*60)

    manifest = pd.read_csv(manifest_path)
    if datasets:
        manifest = manifest[manifest["dataset_name"].isin(datasets)]

    if conditions is None:
        conditions = list(CONDITIONS.keys())
    else:
        conditions = list(dict.fromkeys(conditions))
    unknown = sorted(set(conditions) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    print(f"[COND] Generating: {conditions}")
    print(f"[DATA] {len(manifest)} records across {manifest['dataset_name'].nunique()} datasets")

    # Create output directories
    for ds in manifest["dataset_name"].unique():
        for cond in conditions:
            out_dir = IMAGES_DIR / ds / cond
            out_dir.mkdir(parents=True, exist_ok=True)

    # Flatten: output_dirs[condition][dataset] = path
    flat_output_dirs = {}
    for cond in conditions:
        flat_output_dirs[cond] = {
            ds: IMAGES_DIR / ds / cond
            for ds in manifest["dataset_name"].unique()
        }

    # Generate
    all_outputs = []
    duplicate_stems = set(
        manifest.loc[
            manifest.assign(__stem=manifest["image_path"].map(lambda p: Path(str(p)).stem))
            .duplicated(["dataset_name", "__stem"], keep=False),
            "image_path",
        ].map(lambda p: Path(str(p)).stem)
    ) if not manifest.empty else set()
    for idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Generating conditions"):
        ds = row["dataset_name"]
        ds_dirs = {c: flat_output_dirs[c][ds] for c in conditions}
        dilated_path = row.get("dilated_mask_path", "")
        if pd.isna(dilated_path):
            dilated_path = ""
        outputs = generate_conditions_for_record(
            row,
            ds_dirs,
            dilated_mask_path=str(dilated_path) or None,
            output_name=condition_output_name(
                row, disambiguate=Path(str(row["image_path"])).stem in duplicate_stems
            ),
        )
        outputs["manifest_idx"] = idx
        outputs["dataset"] = ds
        all_outputs.append(outputs)

    # Spot-check
    print(f"\n[QC] Spot-checking {n_spotcheck} random samples...")
    for _ in range(min(n_spotcheck, len(all_outputs))):
        rec = random.choice(all_outputs)
        idx = rec["manifest_idx"]
        row = manifest.loc[idx]
        for cond in conditions:
            if rec.get(cond):
                p = Path(rec[cond])
                if p.exists():
                    img = Image.open(p)
                    print(f"  ✓ {row['dataset_name']}/{cond}/{p.name} ({img.size})")
                else:
                    print(f"  ✗ MISSING: {row['dataset_name']}/{cond}/{p.name}")

    # Save condition manifest
    cond_df = pd.DataFrame(all_outputs)
    cond_path = ROOT / "condition_paths.csv"
    cond_df.to_csv(cond_path, index=False)
    print(f"\n[SAVE] Condition paths → {cond_path}")

    print("\n[DONE] Condition generation complete.")
    return cond_df


if __name__ == "__main__":
    main()
