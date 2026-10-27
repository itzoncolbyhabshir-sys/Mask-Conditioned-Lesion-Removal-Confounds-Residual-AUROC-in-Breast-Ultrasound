"""Main-text experiment configuration for Mask-Conditioned Lesion Removal."""
from pathlib import Path
import os
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
LOGS_DIR = ROOT / 'logs'
FEATURES_DIR = ROOT / 'features'
PREDICTIONS_DIR = ROOT / 'predictions'
RESULTS_DIR = ROOT / 'results'
IMAGES_DIR = ROOT / 'images'
MODELS_DIR = ROOT / 'models'
for d in [DATA_DIR, LOGS_DIR, FEATURES_DIR, PREDICTIONS_DIR, RESULTS_DIR, IMAGES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('TORCH_HOME', str(MODELS_DIR / 'torch'))
os.environ.setdefault('HF_HOME', str(MODELS_DIR / 'huggingface'))
DATASETS = {
    'BUS-BRA': {
        'source': 'Zenodo 10.5281/zenodo.8231412',
        'paper': 'Gómez-Flores et al., Med Phys 2024',
        'images': 1875,
        'patients': 1064,
        'has_patient_id': True,
        'label_source': 'CSV biopsy',
        'mask_type': 'PNG binary',
        'role': 'primary_train',
        'raw_dir': DATA_DIR / 'BUS-BRA',
    },
    'BUS-UCLM': {
        'source': 'Mendeley 10.17632/7fvgj4jsp7 (V3)',
        'paper': 'Vallez et al., Sci Data 2025',
        'images': 683,
        'patients': 38,
        'has_patient_id': True,
        'label_source': 'RGB mask color (green=benign, red=malignant)',
        'mask_type': 'RGB color',
        'role': 'secondary_external',
        'raw_dir': DATA_DIR / 'BUS-UCLM',
    },
}
MODELS = {
    'ResNet-50': {
        'family': 'CNN',
        'source': 'torchvision IMAGENET1K_V2',
        'feature_dim': 2048,
        'pooling': 'avgpool',
        'tier': 'lower_bound',
        'weights': 'IMAGENET1K_V2',
    },
    'DINOv3': {
        'family': 'general_vision_FM',
        'source': 'facebookresearch/dinov3',
        'feature_dim': 1024,
        'pooling': 'CLS',
        'variant': 'vit_large_patch16_dinov3.lvd1689m',
        'tier': 'primary',
    },
    'MedSAM2': {
        'family': 'medical_seg_FM',
        'source': 'bowang-lab/MedSAM2',
        'feature_dim': 256,
        'pooling': 'mean_pool',
        'variant': 'MedSAM2_latest',
        'tier': 'primary',
    },
    'UltraSam': {
        'family': 'ultrasound_FM',
        'source': 'CAMMA-public/UltraSam',
        'feature_dim': 768,
        'pooling': 'mean_pool',
        'variant': 'latest',
        'tier': 'primary',
    },
}
CORE_MODELS = ['ResNet-50', 'DINOv3']
PRIMARY_MODELS = CORE_MODELS + ['MedSAM2', 'UltraSam']
CONDITIONS = {
    'C1_full': {
        'description': 'Original image (single-channel)',
        'type': 'baseline',
    },
    'C2_lesion-only': {
        'description': 'Mask retain, background zero',
        'type': 'ablation',
    },
    'C4_lesion-removed-inpaint': {
        'description': 'Mask region inpainted — primary endpoint',
        'type': 'primary',
    },
    'C5_mask-only': {
        'description': 'Binary mask only (no image)',
        'type': 'control',
    },
    'C6_shape-only': {
        'description': 'Shape features (area, perimeter, circularity, solidity, bbox)',
        'type': 'control',
    },
    'C5d_dilated-mask-only': {
        'description': 'Dilated intervention mask only (no image)',
        'type': 'control',
    },
    'C6d_dilated-geometry-rendering': {
        'description': 'Dilated intervention mask geometry rendering',
        'type': 'control',
    },
}
N_SPLITS = 5
SEEDS = [2026, 2027, 2028, 2029, 2030]
PROBE_CONFIG = {
    'class_weight': 'balanced',
    'max_iter': 5000,
    'penalty': 'l2',
    'C_grid': np.logspace(-3, 3, 7).tolist(),
    'inner_cv': 3,
}
PRIMARY_METRIC = 'AUROC'
SECONDARY_METRICS = ['AUPRC', 'balanced_accuracy', 'sensitivity', 'specificity']
STAT_CONFIG = {
    'bootstrap_iter': 1000,
    'permutation_iter': 1000,
    'ci_level': 0.95,
    'p_threshold': 0.05,
}
MANIFEST_COLUMNS = ['image_path', 'mask_path', 'label', 'patient_id', 'dataset_name', 'scanner_vendor', 'lesion_area_ratio', 'mask_bbox', 'split_id']
EXPERIMENTS = {
    'exp1_primary': {
        'conditions': ['C1_full', 'C2_lesion-only', 'C4_lesion-removed-inpaint'],
        'datasets': ['BUS-BRA', 'BUS-UCLM'],
        'models': ['DINOv3', 'ResNet-50'],
    },
    'exp2_controls': {
        'conditions': ['C4_lesion-removed-inpaint', 'C5_mask-only', 'C6_shape-only'],
        'datasets': ['BUS-BRA'],
        'models': ['DINOv3'],
    },
}
MASK_CONFIG = {
    'dilation_kernel': 5,
    'binary_threshold': 0,
}
INPAINT_CONFIG = {
    'telea_radius': 5,
    'ns_radius': 5,
}
LOG_CONFIG = {
    'data_provenance': LOGS_DIR / 'data_provenance.txt',
    'exp1_runlog': LOGS_DIR / 'exp1_runlog.md',
    'exp2_runlog': LOGS_DIR / 'exp2_runlog.md',
    'seed_hashes': LOGS_DIR / 'seed_checkpoint_hashes.txt',
}
print(f'[Config] Project root: {ROOT}')
print(f'[Config] Data dir: {DATA_DIR}')
print(f'[Config] {len(DATASETS)} datasets, {len(MODELS)} models, {len(CONDITIONS)} conditions')
print(f'[Config] {N_SPLITS}-fold CV, seeds={SEEDS}')
