# BUS-UCLM patient-label sensitivity

Probe training uses image-level lesion labels. Patient-level evaluation uses an any-malignant target.

## Cohort accounting

| Patients | Benign-only | Malignant-only | Mixed | Final benign target | Final malignant target |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 35 | 18 | 12 | 5 | 18 | 17 |

## DINOv3 BUS-UCLM sensitivity

| Condition | Pooling | Mixed patients | Patients | AUROC [95% CI] |
| --- | --- | --- | ---: | --- |
| Full image (C1) | mean | mixed included | 35 | 0.680 [0.480, 0.866] |
| Full image (C1) | mean | mixed excluded | 30 | 0.690 [0.500, 0.870] |
| Full image (C1) | max | mixed included | 35 | 0.585 [0.399, 0.784] |
| Full image (C1) | max | mixed excluded | 30 | 0.593 [0.403, 0.778] |
| Full image (C1) | noisy_or | mixed included | 35 | 0.485 [0.304, 0.683] |
| Full image (C1) | noisy_or | mixed excluded | 30 | 0.440 [0.231, 0.634] |
| Lesion-only (C2) | mean | mixed included | 35 | 0.954 [0.863, 1.000] |
| Lesion-only (C2) | mean | mixed excluded | 30 | 0.963 [0.894, 1.000] |
| Lesion-only (C2) | max | mixed included | 35 | 0.863 [0.732, 0.964] |
| Lesion-only (C2) | max | mixed excluded | 30 | 0.838 [0.676, 0.958] |
| Lesion-only (C2) | noisy_or | mixed included | 35 | 0.876 [0.748, 0.974] |
| Lesion-only (C2) | noisy_or | mixed excluded | 30 | 0.847 [0.690, 0.963] |
| Lesion-removed (C4) | mean | mixed included | 35 | 0.886 [0.755, 0.984] |
| Lesion-removed (C4) | mean | mixed excluded | 30 | 0.903 [0.778, 0.995] |
| Lesion-removed (C4) | max | mixed included | 35 | 0.801 [0.634, 0.925] |
| Lesion-removed (C4) | max | mixed excluded | 30 | 0.796 [0.606, 0.935] |
| Lesion-removed (C4) | noisy_or | mixed included | 35 | 0.775 [0.614, 0.909] |
| Lesion-removed (C4) | noisy_or | mixed excluded | 30 | 0.741 [0.560, 0.889] |
| Dilated mask-only (C5d) | mean | mixed included | 35 | 0.967 [0.892, 1.000] |
| Dilated mask-only (C5d) | mean | mixed excluded | 30 | 0.972 [0.912, 1.000] |
| Dilated mask-only (C5d) | max | mixed included | 35 | 0.908 [0.801, 0.984] |
| Dilated mask-only (C5d) | max | mixed excluded | 30 | 0.912 [0.796, 0.986] |
| Dilated mask-only (C5d) | noisy_or | mixed included | 35 | 0.840 [0.703, 0.951] |
| Dilated mask-only (C5d) | noisy_or | mixed excluded | 30 | 0.801 [0.606, 0.945] |
| Dilated geometry rendering (C6d) | mean | mixed included | 35 | 0.925 [0.824, 0.993] |
| Dilated geometry rendering (C6d) | mean | mixed excluded | 30 | 0.931 [0.829, 1.000] |
| Dilated geometry rendering (C6d) | max | mixed included | 35 | 0.810 [0.654, 0.935] |
| Dilated geometry rendering (C6d) | max | mixed excluded | 30 | 0.801 [0.606, 0.935] |
| Dilated geometry rendering (C6d) | noisy_or | mixed included | 35 | 0.817 [0.663, 0.941] |
| Dilated geometry rendering (C6d) | noisy_or | mixed excluded | 30 | 0.787 [0.602, 0.931] |
| Explicit dilated geometry | mean | mixed included | 35 | 0.889 [0.752, 0.984] |
| Explicit dilated geometry | mean | mixed excluded | 30 | 0.870 [0.722, 0.986] |
| Explicit dilated geometry | max | mixed included | 35 | 0.817 [0.654, 0.948] |
| Explicit dilated geometry | max | mixed excluded | 30 | 0.801 [0.620, 0.949] |
| Explicit dilated geometry | noisy_or | mixed included | 35 | 0.752 [0.578, 0.912] |
| Explicit dilated geometry | noisy_or | mixed excluded | 30 | 0.676 [0.458, 0.870] |
