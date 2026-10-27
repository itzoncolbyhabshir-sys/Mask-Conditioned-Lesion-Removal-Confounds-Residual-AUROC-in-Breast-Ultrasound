from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src import conditions, manifest, masks, splits


def test_bus_uclm_rgb_label_inference_excludes_normal_masks():
    normal = np.zeros((4, 4, 3), dtype=np.uint8)
    benign = normal.copy()
    benign[1:3, 1:3, 1] = 255
    malignant = normal.copy()
    malignant[1:3, 1:3, 0] = 255

    assert manifest.infer_bus_uclm_label(normal) == "normal"
    assert manifest.infer_bus_uclm_label(benign) == "benign"
    assert manifest.infer_bus_uclm_label(malignant) == "malignant"


def _save_gray(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def test_bus_bra_official_archive_layout(monkeypatch, tmp_path):
    dataset_root = tmp_path / "BUS-BRA"
    official_root = dataset_root / "BUSBRA"
    images_dir = official_root / "Images"
    masks_dir = official_root / "Masks"
    image = np.full((8, 8), 80, dtype=np.uint8)
    lesion = np.zeros((8, 8), dtype=np.uint8)
    lesion[2:5, 3:6] = 255

    _save_gray(images_dir / "bus_0001-l.png", image)
    _save_gray(images_dir / "bus_0001-r.png", image)
    _save_gray(masks_dir / "mask_0001-l.png", lesion)
    _save_gray(masks_dir / "mask_0001-r.png", lesion)
    pd.DataFrame(
        [
            {"ID": "bus_0001-l", "Case": "1", "Pathology": "malignant", "Device": "GE Logiq 7"},
            {"ID": "bus_0001-r", "Case": "1", "Pathology": "malignant", "Device": "GE Logiq 7"},
        ]
    ).to_csv(official_root / "bus_data.csv", index=False)

    config = dict(manifest.DATASETS["BUS-BRA"])
    config["raw_dir"] = dataset_root
    monkeypatch.setitem(manifest.DATASETS, "BUS-BRA", config)

    result = manifest.build_bus_bra_manifest()

    assert len(result) == 2
    assert set(result["label"]) == {"malignant"}
    assert set(result["patient_id"]) == {"BUS-BRA:1"}
    assert set(result["scanner_vendor"]) == {"GE Logiq 7"}
    assert all(Path(path).name.startswith("mask_") for path in result["mask_path"])
    assert all(ratio > 0 for ratio in result["lesion_area_ratio"])


def test_quality_check_handles_blank_mask_path(tmp_path):
    image_path = tmp_path / "image.png"
    _save_gray(image_path, np.zeros((4, 4), dtype=np.uint8))
    frame = pd.DataFrame(
        [
            {
                "image_path": str(image_path),
                "mask_path": "",
                "label": "benign",
                "patient_id": "TEST:1",
                "dataset_name": "TEST",
                "lesion_area_ratio": 0.0,
            }
        ]
    )

    assert manifest.quality_check(frame, n_sample=1) is frame


def test_mask_processing_uses_collision_safe_paths_and_records_dilation(monkeypatch, tmp_path):
    image = np.full((9, 9), 100, dtype=np.uint8)
    lesion = np.zeros((9, 9), dtype=np.uint8)
    lesion[4, 4] = 255
    rows = []
    for directory in (tmp_path / "a", tmp_path / "b"):
        image_path = directory / "same.png"
        mask_path = directory / "same_mask.png"
        _save_gray(image_path, image)
        _save_gray(mask_path, lesion)
        rows.append(
            {
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "label": "benign",
                "patient_id": str(directory.name),
                "dataset_name": "TEST",
                "split_id": -1,
            }
        )
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    monkeypatch.setattr(masks, "PROCESSED_MASK_DIR", tmp_path / "processed")
    monkeypatch.setattr(masks, "plot_area_distribution", lambda *_args, **_kwargs: None)

    result = masks.process_all_masks(manifest_path)

    assert result["mask_path"].nunique() == 2
    assert result["dilated_mask_path"].nunique() == 2
    for base_path, dilated_path in zip(result["mask_path"], result["dilated_mask_path"]):
        base = conditions.load_mask(Path(base_path))
        dilated = conditions.load_mask(Path(dilated_path))
        assert dilated.sum() > base.sum()


def test_condition_subset_uses_saved_dilated_mask_once(monkeypatch, tmp_path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    dilated_path = tmp_path / "mask_dilated.png"
    _save_gray(image_path, np.full((9, 9), 100, dtype=np.uint8))
    base = np.zeros((9, 9), dtype=np.uint8)
    base[4, 4] = 255
    dilated = np.zeros((9, 9), dtype=np.uint8)
    dilated[3:6, 3:6] = 255
    _save_gray(mask_path, base)
    _save_gray(dilated_path, dilated)
    output_dirs = {name: tmp_path / name for name in ("C1_full", "C4_lesion-removed-inpaint")}
    for directory in output_dirs.values():
        directory.mkdir()

    observed = {}

    def capture_inpaint(image, mask, radius=None):
        observed["mask_sum"] = int(mask.sum())
        return image.copy()

    monkeypatch.setattr(conditions, "inpaint_telea", capture_inpaint)
    row = pd.Series(
        {"image_path": str(image_path), "mask_path": str(mask_path), "dataset_name": "TEST"}
    )

    outputs = conditions.generate_conditions_for_record(
        row, output_dirs, dilated_mask_path=str(dilated_path)
    )

    assert set(outputs) == set(output_dirs)
    assert observed["mask_sum"] == 9
    assert all(Path(path).exists() for path in outputs.values())


def test_dilated_mask_controls_use_precomputed_dilated_mask(tmp_path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    dilated_path = tmp_path / "mask_dilated.png"
    _save_gray(image_path, np.full((9, 9), 100, dtype=np.uint8))
    base = np.zeros((9, 9), dtype=np.uint8)
    base[4, 4] = 255
    dilated = np.zeros((9, 9), dtype=np.uint8)
    dilated[3:6, 3:6] = 255
    _save_gray(mask_path, base)
    _save_gray(dilated_path, dilated)

    output_dirs = {
        name: tmp_path / name
        for name in ("C5d_dilated-mask-only", "C6d_dilated-geometry-rendering")
    }
    for directory in output_dirs.values():
        directory.mkdir()

    row = pd.Series(
        {"image_path": str(image_path), "mask_path": str(mask_path), "dataset_name": "TEST"}
    )
    outputs = conditions.generate_conditions_for_record(
        row, output_dirs, dilated_mask_path=str(dilated_path)
    )

    c5d = np.array(Image.open(outputs["C5d_dilated-mask-only"]))
    c6d = np.array(Image.open(outputs["C6d_dilated-geometry-rendering"]))
    assert int((c5d > 0).sum()) == 9
    assert c6d[3, 3] > 0






def test_condition_collision_names_are_stable_and_distinct(tmp_path):
    first = pd.Series({"image_path": str(tmp_path / "a" / "same.png")})
    second = pd.Series({"image_path": str(tmp_path / "b" / "same.png")})
    assert conditions.condition_output_name(first, True) != conditions.condition_output_name(second, True)
    assert conditions.condition_output_name(first, False) == "same.png"


def test_condition_main_generates_only_requested_subset_without_overwrite(monkeypatch, tmp_path):
    rows = []
    for directory in (tmp_path / "source_a", tmp_path / "source_b"):
        image_path = directory / "same.png"
        mask_path = directory / "same_mask.png"
        _save_gray(image_path, np.full((7, 7), 100, dtype=np.uint8))
        _save_gray(mask_path, np.zeros((7, 7), dtype=np.uint8))
        rows.append(
            {
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "dataset_name": "TEST",
                "label": "benign",
                "patient_id": directory.name,
            }
        )
    pd.DataFrame(rows).to_csv(tmp_path / "manifest.csv", index=False)
    monkeypatch.setattr(conditions, "ROOT", tmp_path)
    monkeypatch.setattr(conditions, "IMAGES_DIR", tmp_path / "images")

    result = conditions.main(conditions=["C1_full"], n_spotcheck=0)

    generated = list((tmp_path / "images" / "TEST" / "C1_full").glob("*.png"))
    assert len(generated) == 2
    assert len({path.name for path in generated}) == 2
    assert "C1_full" in result
    assert "C2_lesion-only" not in result


def test_small_group_split_reduces_fold_count_without_patient_leakage():
    frame = pd.DataFrame(
        {
            "label": ["benign", "benign", "malignant", "malignant"],
            "patient_id": ["b1", "b2", "m1", "m2"],
            "dataset_name": ["TEST"] * 4,
        }
    )

    result = splits.stratified_group_kfold_split(frame, n_splits=5)

    assert set(result["split_id"]) == {0, 1}
    assert result.groupby("patient_id")["split_id"].nunique().max() == 1


def test_mixed_lesion_labels_keep_patient_in_one_fold():
    frame = pd.DataFrame(
        {
            "label": [
                "benign", "malignant", "benign", "benign",
                "malignant", "malignant", "benign", "malignant",
            ],
            "patient_id": ["mixed", "mixed", "b1", "b2", "m1", "m2", "b3", "m3"],
            "dataset_name": ["TEST"] * 8,
        }
    )

    result = splits.stratified_group_kfold_split(frame, n_splits=2)

    assert result.groupby("patient_id")["split_id"].nunique().max() == 1
    assert result.loc[result["patient_id"] == "mixed", "split_id"].nunique() == 1


def test_unsplittable_and_empty_frames_are_returned_unassigned():
    tiny = pd.DataFrame(
        {"label": ["benign", "malignant"], "patient_id": ["b1", "m1"]}
    )
    tiny_result = splits.stratified_group_kfold_split(tiny)
    empty_result = splits.stratified_group_kfold_split(
        pd.DataFrame(columns=["label", "patient_id"])
    )

    assert set(tiny_result["split_id"]) == {-1}
    assert empty_result.empty
    assert "split_id" in empty_result
