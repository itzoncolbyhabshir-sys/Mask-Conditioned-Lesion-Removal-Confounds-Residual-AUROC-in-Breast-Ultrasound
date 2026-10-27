"""
00_manifest.py — 统一清单构建
================================
构建 manifest.csv，包含所有数据集的图像路径、掩膜路径、标签、患者ID、
数据集名称、超声设备厂商、病灶面积比等字段。

Usage: python -m src.manifest
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import hashlib
from datetime import datetime

from src.config import (
    DATA_DIR, DATASETS, MANIFEST_COLUMNS, LOG_CONFIG, ROOT
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def empty_manifest() -> pd.DataFrame:
    """Return an empty manifest with the canonical schema."""
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def normalize_label(value) -> str | None:
    """Normalize common binary pathology encodings; reject normal/unknown."""
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"benign", "b", "0", "false"}:
        return "benign"
    if text in {"malignant", "m", "1", "true"}:
        return "malignant"
    return None


def namespaced_patient_id(dataset_name: str, patient_id) -> str:
    """Keep patient identifiers collision-safe when datasets are combined."""
    raw = str(patient_id).strip()
    prefix = f"{dataset_name}:"
    return raw if raw.startswith(prefix) else f"{prefix}{raw}"


def _case_insensitive_child(directory: Path, name: str) -> Path | None:
    if not directory.is_dir():
        return None
    target = name.casefold()
    return next((p for p in directory.iterdir() if p.name.casefold() == target), None)


def _find_bus_bra_layout(raw_dir: Path) -> tuple[Path, Path, Path] | None:
    """Find the official BUSBRA/{Images,Masks,bus_data.csv} layout.

    The Zenodo archive adds a ``BUSBRA`` wrapper directory. Users sometimes
    extract that wrapper into ``data/BUS-BRA`` and sometimes directly beside
    it, so discovery is deliberately tolerant while remaining dataset-scoped.
    """
    candidates = [raw_dir, raw_dir / "BUSBRA", raw_dir.parent / "BUSBRA"]
    if raw_dir.exists():
        candidates.extend(p.parent for p in raw_dir.rglob("bus_data.csv"))

    seen = set()
    for root in candidates:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        images = _case_insensitive_child(root, "Images")
        masks = _case_insensitive_child(root, "Masks")
        metadata = _case_insensitive_child(root, "bus_data.csv")
        if images and images.is_dir() and masks and masks.is_dir() and metadata and metadata.is_file():
            return images, masks, metadata
    return None


def _index_files_by_stem(directory: Path) -> dict[str, Path]:
    return {
        path.stem.casefold(): path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    }


def md5(file_path: Path) -> str:
    """Compute MD5 checksum of a file."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_bbox(mask: np.ndarray) -> dict:
    """Compute bounding box of binary mask as JSON dict."""
    if mask.sum() == 0:
        return {"x": 0, "y": 0, "w": 0, "h": 0}
    ys, xs = np.where(mask > 0)
    return {
        "x": int(xs.min()),
        "y": int(ys.min()),
        "w": int(xs.max() - xs.min()),
        "h": int(ys.max() - ys.min()),
    }


def compute_lesion_area_ratio(mask: np.ndarray) -> float:
    """Lesion area as fraction of total image area."""
    total = mask.size
    lesion = (mask > 0).sum()
    return lesion / total if total > 0 else 0.0


def load_mask_binary(path: Path) -> np.ndarray:
    """Load mask as binary numpy array."""
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        # RGB mask: any channel > 0 = lesion
        arr = (arr.max(axis=2) > 0).astype(np.uint8)
    else:
        arr = (arr > 0).astype(np.uint8)
    return arr




def build_bus_bra_manifest() -> pd.DataFrame:
    """Build manifest for BUS-BRA dataset."""
    ds = DATASETS["BUS-BRA"]
    raw_dir = ds["raw_dir"]
    records = []

    layout = _find_bus_bra_layout(raw_dir)
    if layout is None:
        print(f"[WARN] Official BUS-BRA layout not found below {raw_dir}. Skipping.")
        return empty_manifest()

    img_dir, mask_dir, csv_path = layout
    df_labels = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df_labels.columns = [str(column).strip() for column in df_labels.columns]
    column_lookup = {column.casefold(): column for column in df_labels.columns}
    required = {"id", "case", "pathology"}
    missing = sorted(required - set(column_lookup))
    if missing:
        raise ValueError(f"BUS-BRA metadata missing required columns: {missing}")

    id_column = column_lookup["id"]
    case_column = column_lookup["case"]
    pathology_column = column_lookup["pathology"]
    device_column = column_lookup.get("device")
    df_labels["__id_key"] = df_labels[id_column].str.strip().str.casefold()
    if df_labels["__id_key"].duplicated().any():
        duplicates = df_labels.loc[df_labels["__id_key"].duplicated(), id_column].tolist()
        raise ValueError(f"BUS-BRA metadata contains duplicate image IDs: {duplicates[:5]}")
    metadata_by_id = df_labels.set_index("__id_key")

    mask_by_stem = _index_files_by_stem(mask_dir)
    img_files = sorted(
        path for path in img_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    )
    missing_metadata = []
    missing_masks = []
    for img_path in tqdm(img_files, desc="BUS-BRA"):
        img_id = img_path.stem
        key = img_id.casefold()
        if key not in metadata_by_id.index:
            missing_metadata.append(img_id)
            continue
        metadata_row = metadata_by_id.loc[key]
        label = normalize_label(metadata_row[pathology_column])
        if label is None:
            continue

        # Official names are bus_0001-l.png -> mask_0001-l.png.
        mask_key = ("mask_" + img_id[4:] if key.startswith("bus_") else img_id).casefold()
        mask_path = mask_by_stem.get(mask_key) or mask_by_stem.get(key)
        if mask_path is None:
            missing_masks.append(img_id)
            mask_arr = np.zeros((1, 1), dtype=np.uint8)
        else:
            mask_arr = load_mask_binary(mask_path)

        records.append({
            "image_path": str(img_path),
            "mask_path": str(mask_path) if mask_path else "",
            "label": label,
            "patient_id": namespaced_patient_id("BUS-BRA", metadata_row[case_column]),
            "dataset_name": "BUS-BRA",
            "scanner_vendor": str(metadata_row[device_column]).strip() if device_column else "unknown",
            "lesion_area_ratio": compute_lesion_area_ratio(mask_arr),
            "mask_bbox": json.dumps(compute_bbox(mask_arr)),
            "split_id": -1,  # assign later
        })

    if missing_metadata:
        print(f"[WARN] BUS-BRA: {len(missing_metadata)} images absent from bus_data.csv")
    if missing_masks:
        print(f"[WARN] BUS-BRA: {len(missing_masks)} images have no matching mask")
    df = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    print(f"[BUS-BRA] {len(df)} records (B:{sum(df.label=='benign')}, M:{sum(df.label=='malignant')})")
    return df


def infer_bus_uclm_label(mask_rgb: np.ndarray) -> str:
    """Infer lesion class from the documented RGB mask colors; exclude normals."""
    if mask_rgb.ndim != 3 or mask_rgb.shape[2] < 3:
        return "unknown"
    red = int(mask_rgb[:, :, 0].sum())
    green = int(mask_rgb[:, :, 1].sum())
    if red == 0 and green == 0:
        return "normal"
    return "malignant" if red > green else "benign"


def build_bus_uclm_manifest() -> pd.DataFrame:
    """Build manifest for BUS-UCLM dataset."""
    ds = DATASETS["BUS-UCLM"]
    raw_dir = ds["raw_dir"]
    records = []
    size_mismatch_files = []

    # BUS-UCLM structure: images with RGB masks
    # Patient ID from 4-letter filename prefix
    img_dir = raw_dir / "images"
    mask_dir = raw_dir / "masks"

    if not img_dir.exists():
        print(f"[WARN] BUS-UCLM image dir not found: {img_dir}. Skipping.")
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    img_files = sorted(img_dir.glob("*.png"))
    for img_path in tqdm(img_files, desc="BUS-UCLM"):
        img_id = img_path.stem
        # Patient ID: first 4 letters of filename
        patient_id = img_id[:4] if len(img_id) >= 4 else img_id

        mask_path = mask_dir / f"{img_id}.png"
        if not mask_path.exists():
            mask_path = mask_dir / f"{img_id}_mask.png"

        if mask_path.exists():
            with Image.open(img_path) as image_handle, Image.open(mask_path) as mask_handle:
                if image_handle.size != mask_handle.size:
                    size_mismatch_files.append(img_path.name)
                    continue
            mask_arr = load_mask_binary(mask_path)
            # Infer label from RGB mask: green=benign, red=malignant
            mask_rgb = np.array(Image.open(mask_path))
            label = infer_bus_uclm_label(mask_rgb)
        else:
            mask_arr = np.zeros((1, 1))
            label = "unknown"

        if label not in {"benign", "malignant"}:
            continue

        records.append({
            "image_path": str(img_path),
            "mask_path": str(mask_path) if mask_path.exists() else "",
            "label": label,
            "patient_id": patient_id,
            "dataset_name": "BUS-UCLM",
            "scanner_vendor": "unknown",
            "lesion_area_ratio": compute_lesion_area_ratio(mask_arr),
            "mask_bbox": json.dumps(compute_bbox(mask_arr)),
            "split_id": -1,
        })

    df = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    print(f"[BUS-UCLM] {len(df)} records (B:{sum(df.label=='benign')}, M:{sum(df.label=='malignant')})")
    if size_mismatch_files:
        print(
            "[BUS-UCLM] Excluded image/mask size mismatches: "
            + ", ".join(size_mismatch_files)
        )
    return df








def quality_check(df: pd.DataFrame, n_sample: int = 30) -> pd.DataFrame:
    """Random spot-check: image–mask–label–patient alignment."""
    print("\n" + "="*60)
    print("[QC] Quality Check — Image/Mask/Label/Patient")
    print("="*60)

    problems = []

    for _, row in df.iterrows():
        img_path = Path(row["image_path"])
        mask_value = row.get("mask_path", "")
        mask_path = (
            Path(str(mask_value))
            if pd.notna(mask_value) and str(mask_value).strip()
            else None
        )

        if not img_path.exists():
            problems.append(f"MISSING_IMAGE: {row['image_path']}")
        if mask_path is None:
            problems.append(f"MISSING_MASK_PATH: {row['image_path']}")
        elif not mask_path.exists():
            problems.append(f"MISSING_MASK: {mask_path}")

        # Check mask-image size match
        if img_path.exists() and mask_path is not None and mask_path.is_file():
            img_size = Image.open(img_path).size
            mask_size = Image.open(mask_path).size
            if img_size != mask_size:
                problems.append(f"SIZE_MISMATCH: {row['image_path']} img{img_size} vs mask{mask_size}")

    # Sample check
    sample = df.sample(min(n_sample, len(df)), random_state=2026)
    print(f"[QC] Random {len(sample)} sample check:")
    for _, row in sample.iterrows():
        img_path = Path(row["image_path"])
        print(f"  {row['dataset_name']:12s} | {row['patient_id']:12s} | "
              f"{row['label']:10s} | area={row['lesion_area_ratio']:.4f} | {img_path.name}")

    if problems:
        print(f"\n[QC] WARNING: {len(problems)} problems found:")
        for p in problems[:20]:
            print(f"  - {p}")
    else:
        print(f"\n[QC] All checks passed.")

    print(f"[QC] Total: {len(df)} records")
    print(f"[QC] Label dist: benign={sum(df.label=='benign')}, malignant={sum(df.label=='malignant')}")
    print(f"[QC] Datasets: {df.dataset_name.value_counts().to_dict()}")

    return df


def main():
    print("="*60)
    print("00_manifest.py — Building Unified Manifest")
    print("="*60)

    frames = []
    builders = [
        ("BUS-BRA", build_bus_bra_manifest),
        ("BUS-UCLM", build_bus_uclm_manifest),
    ]

    for name, builder in builders:
        print(f"\n--- {name} ---")
        df = builder()
        if len(df) > 0:
            frames.append(df)

    if not frames:
        raise RuntimeError("No analysis datasets found; prepare BUS-BRA or BUS-UCLM first.")

    manifest = pd.concat(frames, ignore_index=True)
    manifest = quality_check(manifest)

    # Save
    out_path = ROOT / "manifest.csv"
    manifest.to_csv(out_path, index=False)
    print(f"\n[SAVE] manifest.csv -> {out_path} ({len(manifest)} records)")

    # Data provenance log
    with open(LOG_CONFIG["data_provenance"], "w", encoding="utf-8") as f:
        f.write(f"# Data Provenance Log\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
        for ds_name, ds_info in DATASETS.items():
            archive_path = DATA_DIR / ("BUSBRA.zip" if ds_name == "BUS-BRA" else f"{ds_name}.zip")
            available = archive_path.exists() or ds_info["raw_dir"].exists()
            download_date = (
                datetime.fromtimestamp(archive_path.stat().st_mtime).date().isoformat()
                if archive_path.exists() else "local directory present" if available else "not downloaded"
            )
            checksum = f"md5:{md5(archive_path)}" if archive_path.exists() else "not available"
            f.write(f"## {ds_name}\n")
            f.write(f"  Source: {ds_info['source']}\n")
            f.write(f"  Paper: {ds_info['paper']}\n")
            f.write(f"  Download Date: {download_date}\n")
            f.write(f"  Checksum: {checksum}\n")
            f.write(f"  Version/DOI: {ds_info.get('source','')}\n\n")

    print(f"[LOG] data_provenance.txt -> {LOG_CONFIG['data_provenance']}")


if __name__ == "__main__":
    main()
