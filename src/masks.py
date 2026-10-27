"""
01_masks.py — 掩膜处理 SOP
=============================
- 二值化：RGB 掩膜任一通道 >0 → 病灶=1
- 尺寸对齐：最近邻到原图尺寸
- 多病灶：union mask（敏感分析用最大病灶）
- 轻度膨胀：~5px 防止 lesion-removed 残留边界
- 统计 lesion_area_ratio 分布

Usage: python -m src.01_masks
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm
import json
import hashlib
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    ROOT, MASK_CONFIG, LOGS_DIR, IMAGES_DIR,
)

PROCESSED_MASK_DIR = ROOT / "data" / "processed_masks"


def collision_safe_record_id(row: pd.Series) -> str:
    """Create a stable filesystem ID without assuming globally unique stems."""
    source = str(Path(str(row["image_path"])).resolve())
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
    dataset = re.sub(r"[^A-Za-z0-9._-]+", "_", str(row["dataset_name"])).strip("_")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(source).stem).strip("_")
    return f"{dataset}_{stem}_{digest}"


def load_image(path: Path, grayscale: bool = True) -> np.ndarray:
    """Load image as numpy array."""
    img = Image.open(path)
    if grayscale:
        img = img.convert("L")
    return np.array(img)


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Binarize mask: any channel > 0 → lesion=1.
    Handles both grayscale and RGB masks.
    """
    if mask.ndim == 3:
        # RGB: any channel > 0
        binary = (mask.max(axis=2) > MASK_CONFIG["binary_threshold"]).astype(np.uint8)
    else:
        binary = (mask > MASK_CONFIG["binary_threshold"]).astype(np.uint8)
    return binary


def resize_to_original(mask: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Resize mask to original image dimensions using nearest-neighbor."""
    h, w = original_shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def dilate_mask(mask: np.ndarray, kernel_size: int = None) -> np.ndarray:
    """
    Mild dilation to ensure lesion edge coverage.
    Used for lesion-removed condition generation to prevent boundary leaks.
    """
    if kernel_size is None:
        kernel_size = MASK_CONFIG["dilation_kernel"]
    if kernel_size < 1:
        raise ValueError("kernel_size must be positive")
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    dilated = cv2.dilate(mask, kernel, iterations=1)
    return dilated


def union_masks(masks: list[np.ndarray]) -> np.ndarray:
    """Union of multiple lesion masks (multi-lesion case)."""
    if not masks:
        return np.zeros((1, 1), dtype=np.uint8)
    result = masks[0].copy()
    for m in masks[1:]:
        result = np.maximum(result, m)
    return result


def largest_lesion_only(masks: list[np.ndarray]) -> np.ndarray:
    """Keep only the largest lesion (sensitivity analysis)."""
    if not masks:
        return np.zeros((1, 1), dtype=np.uint8)
    areas = [m.sum() for m in masks]
    return masks[np.argmax(areas)]


def compute_lesion_area_ratio(mask: np.ndarray) -> float:
    """Lesion area as fraction of total image."""
    total = mask.size
    lesion = (mask > 0).sum()
    return lesion / total if total > 0 else 0.0


def compute_mask_bbox(mask: np.ndarray) -> dict:
    """Compute bounding box of lesion."""
    if mask.sum() == 0:
        return {"x": 0, "y": 0, "w": 0, "h": 0}
    ys, xs = np.where(mask > 0)
    return {
        "x": int(xs.min()),
        "y": int(ys.min()),
        "w": int(xs.max() - xs.min()),
        "h": int(ys.max() - ys.min()),
    }


def plot_area_distribution(df: pd.DataFrame, out_path: Path = None):
    """Plot and save lesion_area_ratio distribution per dataset."""
    if df.empty or "dataset_name" not in df or "lesion_area_ratio" not in df:
        print("[PLOT] Empty manifest; lesion area plot skipped.")
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    for ds in df["dataset_name"].unique():
        subset = df[df["dataset_name"] == ds]
        axes[0].hist(subset["lesion_area_ratio"], bins=50, alpha=0.5, label=ds)
    axes[0].set_xlabel("Lesion Area Ratio")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Lesion Area Ratio Distribution by Dataset")
    axes[0].legend()
    axes[0].axvline(x=0.01, color="red", linestyle="--", alpha=0.5, label="1% cutoff")

    # Box plot
    data_by_ds = [df[df["dataset_name"] == ds]["lesion_area_ratio"].values
                  for ds in df["dataset_name"].unique()]
    axes[1].boxplot(data_by_ds, labels=df["dataset_name"].unique())
    axes[1].set_ylabel("Lesion Area Ratio")
    axes[1].set_title("Lesion Area Ratio Box Plot")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if out_path is None:
        out_path = LOGS_DIR / "lesion_area_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[PLOT] Lesion area distribution -> {out_path}")


def process_all_masks(manifest_path: Path = None):
    """
    Process all masks from manifest:
    1. Binarize
    2. Align to original image
    3. Dilate (save dilated version separately)
    4. Update manifest with computed fields
    """
    if manifest_path is None:
        manifest_path = ROOT / "manifest.csv"

    if not manifest_path.exists():
        print(f"[ERROR] manifest.csv not found at {manifest_path}")
        print("Run 00_manifest.py first.")
        return

    print("="*60)
    print("01_masks.py — Mask Processing SOP")
    print("="*60)

    manifest = pd.read_csv(manifest_path)
    print(f"[LOAD] {len(manifest)} records from manifest.csv")

    for column, default in (
        ("mask_path", ""),
        ("lesion_area_ratio", 0.0),
        ("mask_bbox", json.dumps({"x": 0, "y": 0, "w": 0, "h": 0})),
        ("dilated_mask_path", ""),
    ):
        if column not in manifest:
            manifest[column] = default

    if manifest.empty:
        manifest.to_csv(manifest_path, index=False)
        print("[DONE] Empty manifest; no masks to process.")
        return manifest

    PROCESSED_MASK_DIR.mkdir(parents=True, exist_ok=True)

    area_ratios = []
    bboxes = []
    processed_paths = []
    dilated_paths = []

    for idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Processing masks"):
        img_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"]) if pd.notna(row["mask_path"]) and row["mask_path"] else None

        if mask_path is None or not mask_path.exists():
            # No mask — use empty
            area_ratios.append(0.0)
            bboxes.append(json.dumps({"x": 0, "y": 0, "w": 0, "h": 0}))
            processed_paths.append("")
            dilated_paths.append("")
            continue

        # Load & binarize
        mask_arr = load_image(mask_path, grayscale=False)
        binary = binarize_mask(mask_arr)

        # Align to original image size
        if img_path.exists():
            original = load_image(img_path, grayscale=True)
            binary = resize_to_original(binary, original.shape)
        else:
            original_shape = binary.shape
        # Save processed (undilated) mask
        record_id = collision_safe_record_id(row)
        out_name = f"{record_id}_mask.png"
        out_path = PROCESSED_MASK_DIR / out_name
        Image.fromarray(binary * 255).save(out_path)

        # Save dilated mask (for lesion-removed conditions)
        dilated = dilate_mask(binary)
        dilated_name = f"{record_id}_mask_dilated.png"
        dilated_path = PROCESSED_MASK_DIR / dilated_name
        Image.fromarray(dilated * 255).save(dilated_path)

        # Compute metrics
        area_ratios.append(compute_lesion_area_ratio(binary))
        bboxes.append(json.dumps(compute_mask_bbox(binary)))
        processed_paths.append(str(out_path))
        dilated_paths.append(str(dilated_path))

    # Update manifest
    manifest["mask_path"] = processed_paths
    manifest["lesion_area_ratio"] = area_ratios
    manifest["mask_bbox"] = bboxes
    manifest["dilated_mask_path"] = dilated_paths

    # Save updated manifest
    manifest.to_csv(manifest_path, index=False)
    print(f"[SAVE] Updated manifest -> {manifest_path}")

    # Statistics
    ratios = np.array([r for r in area_ratios if r > 0])
    print(f"\n[MASK STATS] Lesion area ratio (non-zero masks: {len(ratios)}):")
    if len(ratios):
        print(f"  Mean: {ratios.mean():.4f}")
        print(f"  Median: {np.median(ratios):.4f}")
        print(f"  Min: {ratios.min():.4f}")
        print(f"  Max: {ratios.max():.4f}")
        print(f"  <1%: {(ratios < 0.01).sum()} cases")
    else:
        print("  No non-zero masks available.")

    # Plot
    plot_area_distribution(manifest)

    # Report very small lesions (potential QC issue)
    tiny = manifest[manifest["lesion_area_ratio"] < 0.005]
    if len(tiny) > 0:
        print(f"\n[QC] WARNING: {len(tiny)} images with lesion < 0.5% of image - review recommended")
        for _, r in tiny.head(10).iterrows():
            print(f"  {r['dataset_name']} | {r['image_path']} | ratio={r['lesion_area_ratio']:.5f}")

    print("\n[DONE] Mask processing complete.")
    return manifest


if __name__ == "__main__":
    process_all_masks()
