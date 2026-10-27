import numpy as np
import pandas as pd
import pytest

from src.probe import patient_level_aggregate, train_probe
from src.statistics import (
    _indices_for_sampled_patients,
    align_patient_predictions,
    paired_bootstrap_difference,
    permutation_test,
)


def test_patient_sampling_preserves_duplicate_draws():
    patient_ids = np.array(["a", "a", "b"])
    sampled = np.array(["a", "a", "b"])

    indices = _indices_for_sampled_patients(patient_ids, sampled)

    assert indices.tolist() == [0, 1, 0, 1, 2]


def test_patient_aggregation_uses_any_malignant_without_mutating_records():
    records = pd.DataFrame(
        {
            "patient_id": ["mixed", "mixed", "benign"],
            "y_true": [0, 1, 0],
            "y_prob": [0.2, 0.8, 0.1],
        }
    )
    original = records.copy(deep=True)

    aggregated = patient_level_aggregate(records)

    mixed = aggregated.loc[aggregated["patient_id"] == "mixed"].iloc[0]
    assert mixed["y_true"] == 1
    assert mixed["y_prob"] == pytest.approx(0.5)
    pd.testing.assert_frame_equal(records, original)


def test_permutation_and_bootstrap_p_values_use_plus_one_correction():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    patient_ids = np.array(["a", "b", "c", "d"])

    permutation = permutation_test(
        y_true, y_prob, patient_ids=patient_ids, n_iter=9, seed=3
    )
    paired = paired_bootstrap_difference(
        y_true,
        y_prob,
        1 - y_prob,
        patient_ids=patient_ids,
        n_iter=9,
        seed=3,
    )

    assert permutation["p_value"] >= 0.1
    assert paired["p_value"] >= 0.1


def test_prediction_alignment_is_ordered_and_label_safe():
    first = pd.DataFrame({
        "patient_id": ["p2", "p1"],
        "y_true": [1, 0],
        "y_prob": [0.8, 0.2],
    })
    second = pd.DataFrame({
        "patient_id": ["p1", "p2"],
        "y_true": [0, 1],
        "y_prob": [0.3, 0.7],
    })

    aligned = align_patient_predictions(first, second)

    assert aligned["patient_id"].tolist() == ["p1", "p2"]
    assert aligned["y_prob_1"].tolist() == [0.2, 0.8]
    assert aligned["y_prob_2"].tolist() == [0.3, 0.7]

    second.loc[second["patient_id"] == "p2", "y_true"] = 0
    with pytest.raises(ValueError, match="labels differ"):
        align_patient_predictions(first, second)


def test_probe_handles_requested_cv_larger_than_class_count():
    X = np.array([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    groups = np.array(["a", "b", "c", "d", "e", "f"])

    classifier, scaler, metadata = train_probe(
        X, y, groups_train=groups, C_grid=[0.1, 1.0], inner_cv=5,
        return_metadata=True,
    )

    assert classifier.classes_.tolist() == [0, 1]
    assert scaler.transform(X).shape == X.shape
    assert metadata["inner_cv_mode"] == "group3"


def test_probe_falls_back_from_group3_to_group2_without_image_level_cv():
    X = np.array([[0.0], [0.1], [0.8], [0.9]])
    y = np.array([0, 0, 1, 1])
    groups = np.array(["a", "b", "c", "d"])

    classifier, _, metadata = train_probe(
        X, y, groups_train=groups, C_grid=[0.1, 1.0], inner_cv=3,
        return_metadata=True,
    )

    assert classifier.classes_.tolist() == [0, 1]
    assert metadata["inner_cv_mode"] == "group2"
    assert metadata["inner_cv_n_splits"] == 2


def test_probe_uses_fixed_c_when_group_inner_cv_is_infeasible():
    X = np.array([[0.0], [0.1], [0.8], [0.9]])
    y = np.array([0, 0, 1, 1])
    groups = np.array(["benign_patient", "benign_patient", "malignant_patient", "malignant_patient"])

    classifier, _, metadata = train_probe(
        X, y, groups_train=groups, C_grid=[0.1, 1.0], inner_cv=3,
        return_metadata=True,
    )

    assert classifier.classes_.tolist() == [0, 1]
    assert metadata["inner_cv_mode"] == "fixed_C"
    assert metadata["selected_C"] == 1.0
    assert metadata["fallback_reason"] == "no_valid_group_inner_split"


def test_probe_accepts_patient_balanced_sample_weights():
    X = np.array([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    groups = np.array(["a", "a", "b", "c", "d", "d"])
    weights = np.array([0.75, 0.75, 1.5, 1.5, 0.75, 0.75])

    classifier, _, metadata = train_probe(
        X, y, groups_train=groups, sample_weight=weights,
        C_grid=[0.1, 1.0], inner_cv=2, return_metadata=True,
    )

    assert classifier.classes_.tolist() == [0, 1]
    assert metadata["inner_cv_mode"] == "group2"


def test_probe_can_select_c_by_patient_level_inner_auroc():
    X = np.array([[0.0], [0.1], [0.2], [0.3], [0.8], [0.9], [1.0], [1.1]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    groups = np.array(["b1", "b1", "b2", "b3", "m1", "m1", "m2", "m3"])

    classifier, _, metadata = train_probe(
        X,
        y,
        groups_train=groups,
        C_grid=[0.1, 1.0],
        inner_cv=2,
        inner_selection_metric="patient_auroc",
        return_metadata=True,
    )

    assert classifier.classes_.tolist() == [0, 1]
    assert metadata["inner_selection_metric"] == "patient_auroc"
    assert metadata["inner_cv_mode"] == "group2"


def test_probe_rejects_misaligned_sample_weights():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    groups = np.array(["a", "b", "c", "d"])

    with pytest.raises(ValueError, match="sample_weight"):
        train_probe(X, y, groups_train=groups, sample_weight=np.array([1.0, 1.0]))


def test_probe_rejects_one_class_training_data():
    with pytest.raises(ValueError, match="requires both"):
        train_probe(np.array([[0.0], [1.0]]), np.array([0, 0]))
