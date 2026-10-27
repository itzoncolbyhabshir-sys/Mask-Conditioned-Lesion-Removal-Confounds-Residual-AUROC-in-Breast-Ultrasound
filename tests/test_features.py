import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

import src.features as features_module


class IdentityModel(torch.nn.Module):
    def forward(self, inputs):
        return inputs


def test_dinov3_uses_installed_timm_identifier_without_substitution(tmp_path):
    model = IdentityModel()
    with (
        patch("timm.list_models", return_value=[features_module.DINOV3_TIMM_MODEL]),
        patch("timm.create_model", return_value=model) as create_model,
    ):
        loaded, _, dimension = features_module.load_dinov3(
            torch.device("cpu"),
            offline=False,
            checkpoint_path=tmp_path / "missing.safetensors",
        )

    assert loaded is model
    assert dimension == 1024
    create_model.assert_called_once_with(
        features_module.DINOV3_TIMM_MODEL,
        pretrained=True,
        num_classes=0,
    )


def test_dinov3_failure_is_explicit_and_never_falls_back_to_torch_hub():
    with (
        patch("timm.list_models", return_value=[features_module.DINOV3_TIMM_MODEL]),
        patch("timm.create_model", side_effect=OSError("weights missing")),
        patch("torch.hub.load") as hub_load,
        pytest.raises(features_module.ModelLoadError, match="No DINOv2 fallback"),
    ):
        features_module.load_dinov3(torch.device("cpu"), offline=False)

    hub_load.assert_not_called()


def test_dinov3_prefers_local_safetensors_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "dinov3.safetensors"
    checkpoint.touch()
    monkeypatch.setattr(features_module, "DINOV3_EXPECTED_CHECKPOINT_BYTES", 0)
    model = IdentityModel()
    with (
        patch("timm.list_models", return_value=[features_module.DINOV3_TIMM_MODEL]),
        patch("timm.create_model", return_value=model) as create_model,
    ):
        features_module.load_dinov3(
            torch.device("cpu"), offline=True, checkpoint_path=checkpoint
        )

    create_model.assert_called_once_with(
        features_module.DINOV3_TIMM_MODEL,
        pretrained=False,
        num_classes=0,
        checkpoint_path=str(checkpoint),
    )


def test_dinov3_offline_missing_checkpoint_does_not_attempt_download(tmp_path):
    missing_checkpoint = tmp_path / "missing.safetensors"
    with (
        patch("timm.list_models", return_value=[features_module.DINOV3_TIMM_MODEL]),
        patch("timm.create_model") as create_model,
        pytest.raises(features_module.ModelLoadError, match="checkpoint is missing"),
    ):
        features_module.load_dinov3(
            torch.device("cpu"), offline=True, checkpoint_path=missing_checkpoint
        )

    create_model.assert_not_called()


def test_dinov3_rejects_partial_local_checkpoint(tmp_path):
    checkpoint = tmp_path / "partial.safetensors"
    checkpoint.write_bytes(b"partial")
    with (
        patch("timm.list_models", return_value=[features_module.DINOV3_TIMM_MODEL]),
        patch("timm.create_model") as create_model,
        pytest.raises(features_module.ModelLoadError, match="expected"),
    ):
        features_module.load_dinov3(
            torch.device("cpu"), offline=True, checkpoint_path=checkpoint
        )

    create_model.assert_not_called()


def test_medsam2_missing_checkpoint_fails_before_import(tmp_path):
    repository = tmp_path / "MedSAM2-main"
    repository.mkdir()
    with pytest.raises(features_module.ModelLoadError, match="checkpoint is missing"):
        features_module.load_medsam2_encoder(
            torch.device("cpu"),
            checkpoint_path=tmp_path / "missing.pt",
            repo_path=repository,
        )


def test_medsam2_wrapper_pools_official_vision_features():
    class FakeEncoder(torch.nn.Module):
        def forward(self, inputs):
            batch = inputs.shape[0]
            values = torch.arange(24.0).reshape(1, 3, 2, 4).repeat(batch, 1, 1, 1)
            return {"vision_features": values}

    wrapper = features_module._MedSAM2PooledEncoder(FakeEncoder())
    output = wrapper(torch.zeros(2, 3, 8, 8))

    assert output.shape == (2, 3)
    torch.testing.assert_close(output[0], torch.tensor([3.5, 11.5, 19.5]))




def test_extract_features_restores_dataset_index_order():
    batches = [
        (
            torch.tensor([[20.0], [0.0]]),
            ["two.png", "zero.png"],
            torch.tensor([2, 0]),
        ),
        (torch.tensor([[10.0]]), ["one.png"], torch.tensor([1])),
    ]

    matrix, ids = features_module.extract_features(
        IdentityModel(), batches, torch.device("cpu"), feature_dim=1
    )

    np.testing.assert_array_equal(matrix[:, 0], [0.0, 10.0, 20.0])
    assert ids == ["zero.png", "one.png", "two.png"]


def test_run_extraction_reuses_only_aligned_finite_cache(tmp_path, monkeypatch):
    feature_root = tmp_path / "features"
    cache_dir = feature_root / "ResNet-50" / "BUS-BRA"
    cache_dir.mkdir(parents=True)
    matrix = np.asarray([[1.0, 2.0]], dtype=np.float32)
    np.save(cache_dir / "C1_full.npy", matrix)
    (cache_dir / "C1_full_ids.json").write_text(
        json.dumps(["cached.png"]), encoding="utf-8"
    )
    monkeypatch.setattr(features_module, "FEATURES_DIR", feature_root)
    monkeypatch.setattr(features_module, "IMAGES_DIR", tmp_path / "no-images")
    monkeypatch.setattr(features_module, "ROOT", tmp_path)
    loader = Mock(side_effect=AssertionError("valid cache must avoid model loading"))
    monkeypatch.setitem(features_module.MODEL_LOADERS, "ResNet-50", loader)

    cached_matrix, cached_ids = features_module.run_extraction(
        "ResNet-50", "BUS-BRA", "C1_full", device_str="cpu"
    )

    np.testing.assert_array_equal(cached_matrix, matrix)
    assert cached_ids == ["cached.png"]
    loader.assert_not_called()


def test_cache_loader_rejects_misaligned_or_nonfinite_entries(tmp_path):
    feature_path = tmp_path / "features.npy"
    id_path = tmp_path / "ids.json"
    np.save(feature_path, np.asarray([[np.nan]], dtype=np.float32))
    id_path.write_text(json.dumps(["one.png"]), encoding="utf-8")
    assert features_module._load_cached_features(feature_path, id_path) is None

    np.save(feature_path, np.ones((2, 1), dtype=np.float32))
    assert features_module._load_cached_features(feature_path, id_path) is None
