import run_pipeline
from src.features import _map_ultrasam_backbone_key



def test_run_stage_reports_failure_status(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pipeline, "LOGS_DIR", tmp_path)

    def fail():
        raise RuntimeError("expected")

    result = run_pipeline.run_stage("Synthetic", fail)
    assert result["status"] == "FAILED"
    assert result["result"]["error"] == "expected"
    assert "Status: FAILED" in (tmp_path / "synthetic_pipeline.log").read_text()


def test_stage_log_path_sanitizes_windows_reserved_separator(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pipeline, "LOGS_DIR", tmp_path)
    assert run_pipeline._stage_log_path("Experiment 1: Primary Audit") == (
        tmp_path / "experiment_1_primary_audit_pipeline.log"
    )


def test_ultrasam_checkpoint_key_mapping_matches_timm_samvit_names():
    assert _map_ultrasam_backbone_key("bbox_head.weight") is None
    assert _map_ultrasam_backbone_key("backbone.patch_embed.projection.weight") == (
        "patch_embed.proj.weight"
    )
    assert _map_ultrasam_backbone_key("backbone.layers.3.ln1.weight") == (
        "blocks.3.norm1.weight"
    )
    assert _map_ultrasam_backbone_key(
        "backbone.layers.3.ffn.layers.0.0.weight"
    ) == "blocks.3.mlp.fc1.weight"
    assert _map_ultrasam_backbone_key("backbone.channel_reduction.3.bias") == (
        "neck.3.bias"
    )


def test_stage_binding_passes_cli_filters(monkeypatch):
    captured = {}
    def stage(models=None, datasets=None):
        captured.update(models=models, datasets=datasets)
    monkeypatch.setitem(run_pipeline.STAGE_MAP, "exp1", ("primary", stage))
    monkeypatch.setattr(run_pipeline, "run_stage", lambda name, fn: {"status": "OK", "result": fn()})
    assert run_pipeline.main(["--stage", "exp1", "--models", "ResNet-50", "--datasets", "BUS-BRA"]) == 0
    assert captured == {"models": ["ResNet-50"], "datasets": ["BUS-BRA"]}


def test_cli_returns_nonzero_for_failed_stage(monkeypatch):
    monkeypatch.setattr(run_pipeline, "run_stage", lambda *_args: {"status": "FAILED"})
    assert run_pipeline.main(["--stage", "exp1"]) == 1


def test_primary_report_is_descriptive_and_missing_probe_fails(tmp_path, monkeypatch):
    import pandas as pd
    import pytest
    import src.exp1_primary as experiment
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    monkeypatch.setitem(experiment.LOG_CONFIG, "exp1_runlog", tmp_path / "runlog.md")
    monkeypatch.setattr(experiment, "run_extraction", lambda **kwargs: None)
    monkeypatch.setattr(experiment, "run_probe_experiment", lambda **kwargs: {"AUROC": {"mean": 0.8}})
    patient = pd.DataFrame({"patient_id": ["a", "b"], "y_true": [0, 1], "y_prob": [0.2, 0.8]})
    monkeypatch.setattr(experiment, "load_predictions", lambda *args: patient)
    monkeypatch.setattr(experiment, "aggregate_patient_predictions", lambda frame: frame)
    monkeypatch.setattr(experiment, "bootstrap_auroc_ci", lambda *args: {"observed_auroc": 1.0, "ci_lower": 0.8, "ci_upper": 1.0})
    report = experiment.run_exp1(models=["DINOv3"], datasets=["BUS-BRA"])
    assert set(report["settings"]["DINOv3/BUS-BRA"]) == {"C1_full", "C2_lesion-only", "C4_lesion-removed-inpaint"}
    assert report["settings"]["DINOv3/BUS-BRA"]["C4_lesion-removed-inpaint"]["auroc"] == 1.0
    monkeypatch.setattr(experiment, "run_probe_experiment", lambda **kwargs: {})
    with pytest.raises(RuntimeError, match="no results"):
        experiment.run_exp1(models=["DINOv3"], datasets=["BUS-BRA"])


def test_morphology_report_uses_control_minus_reference(tmp_path, monkeypatch):
    import pandas as pd
    import src.exp2_controls as experiment
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    monkeypatch.setitem(experiment.LOG_CONFIG, "exp2_runlog", tmp_path / "runlog.md")
    monkeypatch.setattr(experiment, "run_extraction", lambda **kwargs: None)
    monkeypatch.setattr(experiment, "run_probe_experiment", lambda **kwargs: {"AUROC": {"mean": 0.8}})
    def predictions(model, dataset, condition):
        value = 0.9 if condition != "C4_lesion-removed-inpaint" else 0.7
        return pd.DataFrame({"patient_id": ["a", "b"], "y_true": [0, 1], "y_prob": [0.1, value]})
    monkeypatch.setattr(experiment, "load_predictions", predictions)
    monkeypatch.setattr(experiment, "aggregate_patient_predictions", lambda frame: frame)
    seen = []
    def compare(**kwargs):
        seen.append((kwargs["y_prob_1"].tolist(), kwargs["y_prob_2"].tolist()))
        return {"delta_auroc": 0.02, "ci_lower": 0.01, "ci_upper": 0.03}
    monkeypatch.setattr(experiment, "paired_bootstrap_difference", compare)
    report = experiment.run_exp2(models=["DINOv3"], datasets=["BUS-BRA"])
    assert report["contrast"] == "control minus C4"
    assert set(report["settings"]["DINOv3/BUS-BRA"]) == {"C5_mask-only", "C6_shape-only"}
    assert seen == [([0.1, 0.9], [0.1, 0.7])] * 2



def _repeated_oof_predictions(patient_probabilities):
    """Two image predictions per patient, each evaluated under two seeds."""
    import pandas as pd
    rows = []
    for index, probability in enumerate(patient_probabilities):
        for image_index, image_offset in enumerate((-0.04, 0.04)):
            for seed, seed_offset in ((2026, -0.01), (2027, 0.01)):
                rows.append({
                    "patient_id": f"patient-{index}",
                    "image_path": f"patient-{index}/image-{image_index}.png",
                    "seed": seed,
                    "y_true": int(index >= 2),
                    "y_prob": probability + image_offset + seed_offset,
                })
    return pd.DataFrame(rows)


def test_primary_report_runs_real_patient_aggregation_and_bootstrap(tmp_path, monkeypatch):
    import json
    import pytest
    import src.exp1_primary as experiment
    from src.config import STAT_CONFIG

    (tmp_path / "results").mkdir()
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    monkeypatch.setitem(experiment.LOG_CONFIG, "exp1_runlog", tmp_path / "runlog.md")
    monkeypatch.setitem(STAT_CONFIG, "bootstrap_iter", 25)
    monkeypatch.setattr(experiment, "run_extraction", lambda **kwargs: None)
    monkeypatch.setattr(experiment, "run_probe_experiment", lambda **kwargs: {"completed": True})
    frame = _repeated_oof_predictions([0.2, 0.65, 0.55, 0.8])
    monkeypatch.setattr(experiment, "load_predictions", lambda *args: frame.copy())

    report = experiment.run_exp1(models=["DINOv3"], datasets=["BUS-BRA"])

    assert report["endpoint"] == "patient-level pooled OOF AUROC"
    endpoints = report["settings"]["DINOv3/BUS-BRA"]
    assert set(endpoints) == {"C1_full", "C2_lesion-only", "C4_lesion-removed-inpaint"}
    for endpoint in endpoints.values():
        assert set(endpoint) == {"auroc", "ci_lower", "ci_upper", "n_patients"}
        assert endpoint["n_patients"] == 4  # 16 seed/image rows collapse to four patients.
        assert endpoint["auroc"] == pytest.approx(0.75)  # Three of four patient pairs rank correctly.
        assert 0 <= endpoint["ci_lower"] <= endpoint["auroc"] <= endpoint["ci_upper"] <= 1
    saved = json.loads((tmp_path / "results" / "exp1_summary.json").read_text(encoding="utf-8"))
    assert saved == report


def test_morphology_report_runs_real_paired_patient_bootstrap(tmp_path, monkeypatch):
    import json
    import pytest
    import src.exp2_controls as experiment
    from src.config import STAT_CONFIG

    (tmp_path / "results").mkdir()
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    monkeypatch.setitem(experiment.LOG_CONFIG, "exp2_runlog", tmp_path / "runlog.md")
    monkeypatch.setitem(STAT_CONFIG, "bootstrap_iter", 25)
    monkeypatch.setattr(experiment, "run_extraction", lambda **kwargs: None)
    monkeypatch.setattr(experiment, "run_probe_experiment", lambda **kwargs: {"completed": True})

    def predictions(model, dataset, condition):
        if condition == "C4_lesion-removed-inpaint":
            return _repeated_oof_predictions([0.8, 0.9, 0.1, 0.2])
        # Different row order must not change patient matching across conditions.
        return _repeated_oof_predictions([0.1, 0.2, 0.8, 0.9]).sample(frac=1, random_state=7)

    monkeypatch.setattr(experiment, "load_predictions", predictions)
    report = experiment.run_exp2(models=["DINOv3"], datasets=["BUS-BRA"])

    assert report["contrast"] == "control minus C4"
    assert report["interval"] == "nominal paired patient bootstrap 95% CI"
    comparisons = report["settings"]["DINOv3/BUS-BRA"]
    assert set(comparisons) == {"C5_mask-only", "C6_shape-only"}
    for comparison in comparisons.values():
        # Perfect controls versus an inverse-ranked reference give +1 for every paired draw.
        assert set(comparison) == {"delta_auroc", "ci_lower", "ci_upper"}
        assert comparison == pytest.approx({"delta_auroc": 1.0, "ci_lower": 1.0, "ci_upper": 1.0})
    saved = json.loads((tmp_path / "results" / "exp2_summary.json").read_text(encoding="utf-8"))
    assert saved == report
