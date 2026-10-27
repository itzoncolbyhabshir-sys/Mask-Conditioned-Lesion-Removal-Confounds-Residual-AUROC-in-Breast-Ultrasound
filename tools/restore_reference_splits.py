"""Check/apply archived outer folds after rebuilding a machine-local manifest.

Uses only the Python standard library. Run before feature extraction/training.
"""
import argparse
import csv
import io
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def key(row):
    name = row.get("image_name") or row["image_path"].replace("\\", "/").rsplit("/", 1)[-1]
    return row["dataset_name"], name


def index_rows(rows):
    indexed = {}
    for row in rows:
        item = key(row)
        if item in indexed:
            raise ValueError(f"Duplicate image key: {item}")
        indexed[item] = row
    return indexed


def align(manifest, reference):
    if not manifest:
        raise ValueError("Empty manifest")
    current = index_rows(manifest)
    archived = index_rows(reference)
    datasets = {row["dataset_name"] for row in manifest}
    expected = {item for item in archived if item[0] in datasets}
    if set(current) != expected:
        raise ValueError("Image set differs from reference for installed datasets; check exclusions and duplicate names")
    patient_folds = {}
    updated = []
    changed = 0
    for row in manifest:
        ref = archived[key(row)]
        for field in ("patient_id", "label"):
            if row[field] != ref[field] or not row[field].strip():
                raise ValueError(f"{field} differs from reference: {key(row)}")
        fold = ref["split_id"]
        if fold not in {"0", "1", "2", "3", "4"}:
            raise ValueError(f"Invalid reference fold: {fold}")
        group = row["dataset_name"], row["patient_id"]
        if group in patient_folds and patient_folds[group] != fold:
            raise ValueError(f"Patient crosses reference folds: {group}")
        patient_folds[group] = fold
        changed += row.get("split_id") != fold
        updated.append(dict(row, split_id=fold))
    for dataset in datasets:
        for fold in range(5):
            labels = {r["label"] for r in updated if r["dataset_name"] == dataset and r["split_id"] == str(fold)}
            if labels != {"benign", "malignant"}:
                raise ValueError(f"Reference {dataset} fold {fold} does not contain both classes")
    return updated, changed


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, reader.fieldnames


def csv_text(rows, fields):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Back up and update local manifest.csv and splits.csv")
    args = parser.parse_args()
    manifest_path = ROOT / "manifest.csv"
    rows, fields = read_csv(manifest_path)
    reference, _ = read_csv(ROOT / "reproducibility" / "splits_reference.csv")
    updated, changed = align(rows, reference)
    if not args.apply:
        split_rows, _ = read_csv(ROOT / "splits.csv")
        _, split_changes = align(split_rows, reference)
        if set(index_rows(rows)) != set(index_rows(split_rows)):
            raise ValueError("manifest.csv and splits.csv contain different samples")
        print(f"Matched {len(rows)} images; manifest fold differences={changed}; splits differences={split_changes}")
        return 1 if changed or split_changes else 0
    if "split_id" not in fields:
        fields.append("split_id")
    writes = {
        manifest_path: csv_text(updated, fields),
        ROOT / "splits.csv": csv_text(updated, ["image_path", "patient_id", "dataset_name", "label", "split_id"]),
    }
    # Validate everything before any write, retain existing local files for rollback.
    for path in writes:
        if path.with_name(path.name + ".before-reference-splits").exists():
            raise ValueError(f"Backup already exists for {path.name}; archive it before applying again")
    for path in writes:
        if path.exists():
            shutil.copy2(path, path.with_name(path.name + ".before-reference-splits"))
    for path, contents in writes.items():
        temporary = path.with_name(path.name + ".reference-splits.tmp")
        temporary.write_text(contents, encoding="utf-8", newline="")
        temporary.replace(path)
    print(f"Applied reference to {len(updated)} images; changed {changed} folds. Re-run without --apply to verify.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
