"""
04_features.py — 冻结特征提取 SOP
=====================================
- All models in eval(), torch.no_grad(), parameters fully frozen
- Use official preprocessing pipeline per model
- Extract agreed pooled features
- Cache to features/{model}/{dataset}/{condition}.npy
- Store strictly aligned ID list
- Verify rows == n_samples, no NaN/Inf

Usage: python -m src.04_features --model DINOv3 --dataset BUS-BRA --condition C1_full
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import argparse
import json
import os
from typing import Optional, Tuple

from src.config import (
    ROOT, MODELS, CONDITIONS, FEATURES_DIR, IMAGES_DIR, MODELS_DIR, SEEDS,
)


# ============================================================
# Model Loaders
# ============================================================

DINOV3_TIMM_MODEL = "vit_large_patch16_dinov3.lvd1689m"
DINOV3_LOCAL_CHECKPOINT = MODELS_DIR / "dinov3_vitl16_lvd1689m.safetensors"
DINOV3_EXPECTED_CHECKPOINT_BYTES = 1_212_347_640
ULTRASAM_TIMM_MODEL = "samvit_base_patch16"
ULTRASAM_LOCAL_CHECKPOINT = MODELS_DIR / "UltraSam.pth"
ULTRASAM_EXPECTED_CHECKPOINT_BYTES = 381_156_023
ULTRASAM_EXPECTED_SHA256 = "d7c223dd03f56b0b77cd246aa3edfae651e99f0ee2dde2e8c50eda7b21fa8a0c"
MEDSAM2_REPO = MODELS_DIR / "MedSAM2-main"
MEDSAM2_LOCAL_CHECKPOINT = MODELS_DIR / "MedSAM2_latest.pt"
MEDSAM2_CONFIG = "configs/sam2.1_hiera_t512.yaml"
MEDSAM2_EXPECTED_CHECKPOINT_BYTES = 156_040_129
MEDSAM2_EXPECTED_SHA256 = "c92743b99f00d078bf32a3afcc38aaa9faf1c1692dffe3eaa7a90938c1991060"


class ModelLoadError(RuntimeError):
    """Raised when a requested pretrained encoder cannot be loaded safely."""


def _offline_requested(offline: Optional[bool] = None) -> bool:
    if offline is not None:
        return offline
    return any(
        os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
        for name in ("BUS_AUDIT_OFFLINE", "HF_HUB_OFFLINE")
    )



def get_torchvision_preprocess():
    """Standard torchvision ImageNet preprocessing."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),  # ultrasound → 3-channel
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_resnet50(device: torch.device) -> Tuple[nn.Module, callable, int]:
    """Load ResNet-50 from torchvision, return (model, preprocess_fn, feature_dim)."""
    from torchvision.models import resnet50, ResNet50_Weights

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # Remove classifier head, keep avgpool
    model.fc = nn.Identity()
    model.eval()
    model.to(device)

    def preprocess(path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        transform = get_torchvision_preprocess()
        return transform(img)

    return model, preprocess, 2048


def load_dinov3(
    device: torch.device,
    offline: Optional[bool] = None,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[nn.Module, callable, int]:
    """Load the timm DINOv3 ViT-L/16 checkpoint without model substitution."""
    offline_mode = _offline_requested(offline)
    try:
        import timm

        if not timm.is_model(DINOV3_TIMM_MODEL):
            raise ModelLoadError(
                f"Installed timm does not provide {DINOV3_TIMM_MODEL!r}; upgrade timm."
            )
        local_checkpoint = Path(checkpoint_path or DINOV3_LOCAL_CHECKPOINT)
        if local_checkpoint.is_file():
            checkpoint_bytes = local_checkpoint.stat().st_size
            if checkpoint_bytes != DINOV3_EXPECTED_CHECKPOINT_BYTES:
                raise ModelLoadError(
                    f"DINOv3 checkpoint has {checkpoint_bytes} bytes; expected "
                    f"{DINOV3_EXPECTED_CHECKPOINT_BYTES}. Remove the partial or "
                    "wrong file and fetch the official model.safetensors again."
                )
            model = timm.create_model(
                DINOV3_TIMM_MODEL,
                pretrained=False,
                num_classes=0,
                checkpoint_path=str(local_checkpoint),
            )
        elif offline_mode:
            raise ModelLoadError(
                f"Offline mode is enabled but the DINOv3 checkpoint is missing: "
                f"{local_checkpoint}. Place model.safetensors at that path first."
            )
        else:
            model = timm.create_model(
                DINOV3_TIMM_MODEL,
                pretrained=True,
                num_classes=0,
            )
    except ModelLoadError:
        raise
    except Exception as exc:
        mode_hint = (
            "Offline mode is enabled and the checkpoint is not available locally. "
            if offline_mode else
            "The checkpoint may need a one-time network download or model-license access. "
        )
        raise ModelLoadError(
            f"Could not load pretrained {DINOV3_TIMM_MODEL}. {mode_hint}"
            "No DINOv2 fallback was used. Original error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    def preprocess(path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return transform(img)

    model.eval()
    model.to(device)
    return model, preprocess, 1024


class _MedSAM2PooledEncoder(nn.Module):
    """Pool the official 256-channel MedSAM2 image embedding."""

    def __init__(self, image_encoder: nn.Module):
        super().__init__()
        self.image_encoder = image_encoder

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embedding = self.image_encoder(images)["vision_features"]
        return embedding.mean(dim=(2, 3))


def load_medsam2_encoder(
    device: torch.device,
    checkpoint_path: Optional[Path] = None,
    repo_path: Optional[Path] = None,
) -> Tuple[nn.Module, callable, int]:
    """Load the official MedSAM2 latest Hiera-Tiny 512 image encoder."""
    import hashlib
    from torchvision import transforms

    checkpoint = Path(checkpoint_path or MEDSAM2_LOCAL_CHECKPOINT)
    repository = Path(repo_path or MEDSAM2_REPO)
    if not repository.is_dir():
        raise ModelLoadError(
            f"Official MedSAM2 source is missing: {repository}. Download "
            "https://github.com/bowang-lab/MedSAM2"
        )
    if not checkpoint.is_file():
        raise ModelLoadError(
            f"Official MedSAM2 checkpoint is missing: {checkpoint}. Download "
            "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt"
        )
    if checkpoint.stat().st_size != MEDSAM2_EXPECTED_CHECKPOINT_BYTES:
        raise ModelLoadError(
            f"MedSAM2 checkpoint has {checkpoint.stat().st_size} bytes; expected "
            f"{MEDSAM2_EXPECTED_CHECKPOINT_BYTES}."
        )
    digest = hashlib.sha256()
    with open(checkpoint, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != MEDSAM2_EXPECTED_SHA256:
        raise ModelLoadError(
            f"MedSAM2 checkpoint SHA256 mismatch: {digest.hexdigest()}."
        )

    repository_text = str(repository.resolve())
    if repository_text not in sys.path:
        sys.path.insert(0, repository_text)
    try:
        from sam2.build_sam import build_sam2

        full_model = build_sam2(
            MEDSAM2_CONFIG,
            str(checkpoint.resolve()),
            device=device,
            apply_postprocessing=False,
        )
    except Exception as exc:
        raise ModelLoadError(
            "Could not construct the official MedSAM2 model. Install hydra-core "
            f"and iopath. Original error: {type(exc).__name__}: {exc}"
        ) from exc

    encoder = _MedSAM2PooledEncoder(full_model.image_encoder).eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    def preprocess(path: Path) -> torch.Tensor:
        return transform(Image.open(path).convert("RGB"))

    return encoder, preprocess, 256




def _map_ultrasam_backbone_key(key: str) -> Optional[str]:
    """Map the official MMPreTrain ViTSAM backbone names to timm names."""
    if not key.startswith("backbone."):
        return None
    mapped = key.removeprefix("backbone.")
    replacements = (
        ("patch_embed.projection.", "patch_embed.proj."),
        (".ffn.layers.0.0.", ".mlp.fc1."),
        (".ffn.layers.1.", ".mlp.fc2."),
        ("layers.", "blocks."),
        (".ln1.", ".norm1."),
        (".ln2.", ".norm2."),
        ("channel_reduction.", "neck."),
    )
    for source, target in replacements:
        mapped = mapped.replace(source, target)
    return mapped


class _UltraSamPooledEncoder(nn.Module):
    """Return the official classification-view mean of raw ViTSAM tokens."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.patch_embed(images)
        if self.backbone.pos_embed is not None:
            from timm.layers import resample_abs_pos_embed_nhwc
            tokens = tokens + resample_abs_pos_embed_nhwc(
                self.backbone.pos_embed, tokens.shape[1:3]
            )
        tokens = self.backbone.pos_drop(tokens)
        tokens = self.backbone.patch_drop(tokens)
        tokens = self.backbone.norm_pre(tokens)
        tokens = self.backbone.blocks(tokens)
        return tokens.mean(dim=(1, 2))


def load_ultrasam_encoder(
    device: torch.device,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[nn.Module, callable, int]:
    """Load the official UltraSam ViT-B encoder through timm's SAM architecture."""
    import hashlib
    import timm
    from torchvision.transforms import functional as TF
    from torchvision.transforms import InterpolationMode

    checkpoint = Path(checkpoint_path or ULTRASAM_LOCAL_CHECKPOINT)
    if not checkpoint.is_file():
        raise ModelLoadError(
            f"Official UltraSam checkpoint is missing: {checkpoint}. Download "
            "https://s3.unistra.fr/camma_public/github/ultrasam/UltraSam.pth"
        )
    if checkpoint.stat().st_size != ULTRASAM_EXPECTED_CHECKPOINT_BYTES:
        raise ModelLoadError(
            f"UltraSam checkpoint has {checkpoint.stat().st_size} bytes; expected "
            f"{ULTRASAM_EXPECTED_CHECKPOINT_BYTES}."
        )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != ULTRASAM_EXPECTED_SHA256:
        raise ModelLoadError(
            f"UltraSam checkpoint SHA256 mismatch: {digest}."
        )

    backbone = timm.create_model(
        ULTRASAM_TIMM_MODEL,
        pretrained=False,
        img_size=1024,
        num_classes=0,
    )
    try:
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as exc:
        raise ModelLoadError(
            "UltraSam checkpoint metadata requires the lightweight official "
            "dependency mmengine. Install it with: python -m pip install mmengine"
        ) from exc
    official_state = checkpoint_data.get("state_dict", checkpoint_data)
    mapped_state = {
        mapped: value
        for key, value in official_state.items()
        if (mapped := _map_ultrasam_backbone_key(key)) is not None
    }
    missing, unexpected = backbone.load_state_dict(mapped_state, strict=False)
    if missing or unexpected or len(mapped_state) != len(backbone.state_dict()):
        raise ModelLoadError(
            "UltraSam backbone mapping is incomplete: "
            f"mapped={len(mapped_state)}, expected={len(backbone.state_dict())}, "
            f"missing={missing}, unexpected={unexpected}"
        )

    encoder = _UltraSamPooledEncoder(backbone).eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    mean = [123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0]
    std = [58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0]

    def preprocess(path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        width, height = image.size
        scale = 1024.0 / max(width, height)
        resized = TF.resize(
            image,
            [max(1, round(height * scale)), max(1, round(width * scale))],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        canvas = Image.new("RGB", (1024, 1024), color=(0, 0, 0))
        canvas.paste(resized, (0, 0))
        tensor = TF.to_tensor(canvas)
        return TF.normalize(tensor, mean=mean, std=std)

    return encoder, preprocess, 768




MODEL_LOADERS = {
    "ResNet-50": load_resnet50,
    "DINOv3": load_dinov3,
    "MedSAM2": load_medsam2_encoder,
    "UltraSam": load_ultrasam_encoder,
}


# ============================================================
# Feature Extraction
# ============================================================

class ImageDataset(Dataset):
    """Simple dataset loading pre-generated condition images."""
    def __init__(self, paths: list, preprocess_fn: callable):
        self.paths = [Path(p) for p in paths]
        self.preprocess = preprocess_fn

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        tensor = self.preprocess(path)
        return tensor, str(path), idx


@torch.no_grad()
def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    feature_dim: int,
) -> Tuple[np.ndarray, list]:
    """
    Extract frozen features.
    Returns (features: [N, D], ids: list of image paths).
    """
    model.eval()
    all_features = []
    all_ids = []
    all_indices = []

    for batch in tqdm(dataloader, desc="Extracting features"):
        images, paths, indices = batch
        images = images.to(device)

        feats = model(images)

        if isinstance(feats, tuple):
            feats = feats[0]  # some models return (features, aux)

        # Pool if needed
        if feats.dim() == 3:
            # Patch tokens: mean pool
            feats = feats.mean(dim=1)
        elif feats.dim() == 4:
            # Spatial: global average pool
            feats = feats.mean(dim=[2, 3])

        all_features.append(feats.cpu().numpy())
        all_ids.extend(paths)
        all_indices.extend(int(index) for index in indices)

    if not all_features:
        return np.empty((0, feature_dim), dtype=np.float32), []
    features = np.concatenate(all_features, axis=0)
    order = np.argsort(np.asarray(all_indices), kind="stable")
    sorted_indices = np.asarray(all_indices)[order]
    expected_indices = np.arange(len(sorted_indices))
    if not np.array_equal(sorted_indices, expected_indices):
        raise RuntimeError(
            "DataLoader indices are duplicated or incomplete; refusing to save "
            "potentially misaligned features."
        )
    features = features[order]
    ids = [str(all_ids[index]) for index in order]
    return features, ids


def _load_cached_features(
    feature_path: Path,
    id_path: Path,
    expected_ids: Optional[list[str]] = None,
) -> Optional[Tuple[np.ndarray, list[str]]]:
    """Return a complete, finite, aligned cache entry or None."""
    if not feature_path.exists() or not id_path.exists():
        return None
    try:
        features = np.load(feature_path, allow_pickle=False)
        with open(id_path, encoding="utf-8") as handle:
            ids = json.load(handle)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if features.ndim != 2 or len(features) != len(ids):
        return None
    if not np.isfinite(features).all() or not all(isinstance(item, str) for item in ids):
        return None
    if expected_ids is not None and ids != expected_ids:
        return None
    return features, ids


def _atomic_save_features(
    feature_path: Path,
    id_path: Path,
    features: np.ndarray,
    ids: list[str],
) -> None:
    """Write the feature and ID cache through temporary sibling files."""
    feature_tmp = feature_path.with_suffix(feature_path.suffix + ".tmp")
    id_tmp = id_path.with_suffix(id_path.suffix + ".tmp")
    try:
        with open(feature_tmp, "wb") as handle:
            np.save(handle, features, allow_pickle=False)
        with open(id_tmp, "w", encoding="utf-8") as handle:
            json.dump(ids, handle)
        feature_tmp.replace(feature_path)
        id_tmp.replace(id_path)
    finally:
        feature_tmp.unlink(missing_ok=True)
        id_tmp.unlink(missing_ok=True)


def run_extraction(
    model_name: str,
    dataset_name: str,
    condition: str,
    batch_size: int = 32,
    device_str: str = "cuda",
    num_workers: int = 0,
    force: bool = False,
    offline: Optional[bool] = None,
) -> Tuple[np.ndarray, list]:
    """
    Extract features for a specific (model, dataset, condition) combination.
    """
    print(f"\n{'='*60}")
    print(f"[EXTRACT] {model_name} / {dataset_name} / {condition}")
    print(f"{'='*60}")

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    if model_name == "UltraSam" and batch_size > 2:
        print(f"[BATCH] UltraSam 1024px encoder: reducing batch size {batch_size} -> 2")
        batch_size = 2
    if model_name not in MODEL_LOADERS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_LOADERS.keys())}")

    # Find condition images
    cond_dir = IMAGES_DIR / dataset_name / condition
    if not cond_dir.exists():
        # Try loading from condition_paths.csv
        cond_path = ROOT / "condition_paths.csv"
        if cond_path.exists():
            cond_df = pd.read_csv(cond_path)
            paths = cond_df[cond_df["dataset"] == dataset_name].get(condition, pd.Series([]))
            paths = [p for p in paths if pd.notna(p) and p and Path(p).exists()]
        else:
            paths = []
    else:
        paths = sorted(cond_dir.glob("*.png"))

    paths = sorted((Path(path) for path in paths), key=lambda path: str(path))

    out_dir = FEATURES_DIR / model_name / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_path = out_dir / f"{condition}.npy"
    id_path = out_dir / f"{condition}_ids.json"
    expected_ids = [str(path) for path in paths] if paths else None
    if not force:
        cached = _load_cached_features(feat_path, id_path, expected_ids)
        if cached is not None:
            print(f"[CACHE] Reusing {feat_path} ({cached[0].shape})")
            return cached

    if not paths:
        print(f"[SKIP] No images found for {dataset_name}/{condition}")
        return np.array([]), []

    print(f"[IMAGES] {len(paths)} images")

    loader_fn = MODEL_LOADERS[model_name]
    if model_name == "DINOv3":
        model, preprocess_fn, feature_dim = loader_fn(device, offline=offline)
    else:
        model, preprocess_fn, feature_dim = loader_fn(device)

    # Dataset & DataLoader
    dataset = ImageDataset(paths, preprocess_fn)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    # Extract
    features, ids = extract_features(model, loader, device, feature_dim)

    # Verify
    assert len(features) == len(paths), \
        f"Mismatch: {len(features)} features vs {len(paths)} images"
    assert not np.isnan(features).any(), f"NaN in features for {model_name}/{dataset_name}/{condition}"
    assert not np.isinf(features).any(), f"Inf in features for {model_name}/{dataset_name}/{condition}"

    print(f"[FEATURES] shape: {features.shape}, NaNs: {np.isnan(features).sum()}, Infs: {np.isinf(features).sum()}")

    # Save the matrix and its aligned identifiers as one validated cache entry.
    _atomic_save_features(feat_path, id_path, features, ids)
    print(f"[SAVE] {feat_path} ({features.shape})")

    return features, ids


def main():
    parser = argparse.ArgumentParser(description="Feature Extraction")
    parser.add_argument("--model", type=str, default="DINOv3", help="Model name")
    parser.add_argument("--dataset", type=str, default="BUS-BRA", help="Dataset name")
    parser.add_argument("--condition", type=str, default="C1_full", help="Condition name")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Ignore a valid feature cache")
    parser.add_argument(
        "--offline", action="store_true",
        help="Use local model caches only; fail clearly when weights are absent",
    )
    args = parser.parse_args()

    run_extraction(
        model_name=args.model,
        dataset_name=args.dataset,
        condition=args.condition,
        batch_size=args.batch_size,
        device_str=args.device,
        num_workers=args.num_workers,
        force=args.force,
        offline=args.offline,
    )


if __name__ == "__main__":
    main()
