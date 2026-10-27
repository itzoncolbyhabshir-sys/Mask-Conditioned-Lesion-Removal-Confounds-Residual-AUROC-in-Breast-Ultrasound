"""
03_splits.py — 患者级划分 (StratifiedGroupKFold)
====================================================
- Per-dataset independent StratifiedGroupKFold (n_splits=5)
- y=label, groups=patient_id
- Assert no patient crosses folds
- Print per-fold benign/malignant distribution
- BUS-UCLM (38 patients): extra uncertainty caveat
- Seeds fixed {2026..2030} for downstream use only (NOT re-splitting)
- Split written once to manifest.csv split_id column

Usage: python -m src.03_splits
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from collections import Counter

from src.config import ROOT, N_SPLITS


def stratified_group_kfold_split(
    df: pd.DataFrame,
    n_splits: int = N_SPLITS,
    random_state: int = 2026,
) -> pd.DataFrame:
    """
    Assign split_id (0..n_splits-1) to each record.
    Ensures no patient crosses folds and label balance maintained.
    """
    df = df.copy()
    df["split_id"] = -1

    if df.empty:
        return df

    required = {"label", "patient_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cannot split without columns: {sorted(missing)}")

    if df["patient_id"].isna().any() or (df["patient_id"].astype(str).str.strip() == "").any():
        raise ValueError("patient_id must be present for every record")

    # Map labels to integers
    label_map = {"benign": 0, "malignant": 1}
    mapped = df["label"].map(label_map)
    if mapped.isna().any():
        bad = sorted(df.loc[mapped.isna(), "label"].astype(str).unique())
        raise ValueError(f"Unsupported labels for binary split: {bad}")
    y = mapped.values
    groups = df["patient_id"].values

    patient_labels = df[["patient_id", "label"]].drop_duplicates()
    conflicted = patient_labels["patient_id"].value_counts()
    conflicted = conflicted[conflicted > 1]
    if not conflicted.empty:
        # Some external datasets label individual lesions rather than patients.
        # Keep every patient's lesions in one fold and stratify patients by an
        # "any malignant lesion" summary without altering record-level targets.
        patient_table = (
            df.assign(_patient_target=(mapped == 1).astype(int))
            .groupby("patient_id", as_index=False)["_patient_target"]
            .max()
        )
        class_group_counts = patient_table["_patient_target"].value_counts()
        effective_splits = min(
            n_splits,
            int(class_group_counts.min()),
            len(patient_table),
        )
        if effective_splits < 2:
            print("[WARN] Split skipped: fewer than two patient groups per class.")
            return df
        splitter = StratifiedKFold(
            n_splits=effective_splits,
            shuffle=True,
            random_state=random_state,
        )
        for fold_idx, (_, test_idx) in enumerate(
            splitter.split(patient_table, patient_table["_patient_target"])
        ):
            test_patients = set(patient_table.iloc[test_idx]["patient_id"])
            df.loc[df["patient_id"].isin(test_patients), "split_id"] = fold_idx
        return df

    class_group_counts = patient_labels.groupby("label")["patient_id"].nunique()
    if len(class_group_counts) < 2:
        print("[WARN] Split skipped: both benign and malignant groups are required.")
        return df
    effective_splits = min(n_splits, int(class_group_counts.min()), df["patient_id"].nunique())
    if effective_splits < 2:
        print("[WARN] Split skipped: fewer than two patient groups per class.")
        return df
    if effective_splits != n_splits:
        print(f"[WARN] Reducing folds from {n_splits} to {effective_splits} for available patient groups.")

    sgkf = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(np.zeros(len(df)), y, groups)):
        df.loc[df.index[test_idx], "split_id"] = fold_idx

        # Verify
        test_patients = set(groups[test_idx])
        train_patients = set(groups[train_idx])
        overlap = test_patients & train_patients
        assert len(overlap) == 0, f"Fold {fold_idx}: patient overlap! {overlap}"

    return df


def verify_splits(df: pd.DataFrame) -> dict:
    """Verify split integrity and report statistics."""
    print("\n" + "="*60)
    print("[VERIFY] Split Integrity Check")
    print("="*60)

    report = {}

    for ds_name in df["dataset_name"].unique():
        ds_df = df[df["dataset_name"] == ds_name]
        n_patients = ds_df["patient_id"].nunique()
        n_images = len(ds_df)

        print(f"\n--- {ds_name} ---")
        print(f"  Images: {n_images}, Patients: {n_patients}")

        fold_info = []
        assigned_folds = sorted(
            int(fold)
            for fold in ds_df["split_id"].unique()
            if pd.notna(fold) and int(fold) >= 0
        )
        for fold in assigned_folds:
            fold_df = ds_df[ds_df["split_id"] == fold]
            b_count = (fold_df["label"] == "benign").sum()
            m_count = (fold_df["label"] == "malignant").sum()
            n_pat = fold_df["patient_id"].nunique()
            fold_info.append({
                "fold": fold,
                "n_images": len(fold_df),
                "n_patients": n_pat,
                "benign": b_count,
                "malignant": m_count,
                "b_ratio": b_count / len(fold_df) if len(fold_df) > 0 else 0,
            })
            benign_percent = b_count / len(fold_df) * 100 if len(fold_df) else 0.0
            print(f"  Fold {fold}: {len(fold_df):4d} imgs | {n_pat:4d} patients | "
                  f"B:{b_count:4d} M:{m_count:4d} | B%:{benign_percent:.1f}%")

        unassigned = int((ds_df["split_id"] < 0).sum())
        if unassigned:
            print(f"  WARNING: {unassigned} records unassigned: insufficient data for stratified group CV.")

        # Check no patient in multiple folds
        for patient in ds_df["patient_id"].unique():
            patient_folds = ds_df[ds_df["patient_id"] == patient]["split_id"].unique()
            if len(patient_folds) > 1:
                print(f"  CRITICAL: Patient {patient} in folds {patient_folds}!")

        report[ds_name] = {
            "n_images": n_images,
            "n_patients": n_patients,
            "fold_info": fold_info,
        }

        # BUS-UCLM caveat
        if ds_name == "BUS-UCLM" and n_patients < 50:
            denominator = max(len(assigned_folds), 1)
            print(f"  WARNING: BUS-UCLM: only {n_patients} patients -> ~{n_patients//denominator}/fold.")
            print(f"     Results have higher uncertainty. Report explicitly.")

    return report


def main():
    manifest_path = ROOT / "manifest.csv"
    if not manifest_path.exists():
        print("[ERROR] manifest.csv not found. Run 00_manifest.py first.")
        return

    print("="*60)
    print("03_splits.py — Patient-Level StratifiedGroupKFold Split")
    print("="*60)

    manifest = pd.read_csv(manifest_path)
    print(f"[LOAD] {len(manifest)} records from manifest.csv")

    if "split_id" not in manifest:
        manifest["split_id"] = -1

    if manifest.empty:
        manifest.to_csv(manifest_path, index=False)
        split_path = ROOT / "splits.csv"
        manifest.reindex(columns=["image_path", "patient_id", "dataset_name", "label", "split_id"]).to_csv(
            split_path, index=False
        )
        print("[DONE] Empty manifest; wrote empty split archive.")
        return manifest

    # Per-dataset splitting
    all_frames = []
    for ds_name in manifest["dataset_name"].unique():
        ds_df = manifest[manifest["dataset_name"] == ds_name].copy()
        print(f"\n[Splitting] {ds_name}: {len(ds_df)} images, "
              f"{ds_df['patient_id'].nunique()} patients")

        ds_df = stratified_group_kfold_split(ds_df)

        all_frames.append(ds_df)

    manifest = pd.concat(all_frames, ignore_index=True)

    # Verify
    verify_splits(manifest)

    # Save
    manifest.to_csv(manifest_path, index=False)
    print(f"\n[SAVE] manifest.csv with split_id -> {manifest_path}")

    # Also save a standalone split file for archiving
    split_df = manifest[["image_path", "patient_id", "dataset_name", "label", "split_id"]]
    split_path = ROOT / "splits.csv"
    split_df.to_csv(split_path, index=False)
    print(f"[ARCHIVE] Split file -> {split_path}")

    return manifest


if __name__ == "__main__":
    main()
