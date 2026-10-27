# Dilated intervention-mask controls

These controls use the same dilated intervention-mask family used to generate C4, rather than relying only on the original lesion mask.

| Setting | Control | AUROC [95% CI] | Delta vs C4 |
| --- | --- | --- | --- |
| DINOv3/BUS-BRA | C5d_dilated-mask-only | 0.945 [0.929, 0.959] | +0.017 [+0.006, +0.029] |
| DINOv3/BUS-BRA | C6d_dilated-geometry-rendering | 0.935 [0.917, 0.951] | +0.007 [-0.005, +0.020] |
| DINOv3/BUS-BRA | dilated geometry features | 0.931 [0.913, 0.947] | +0.003 [-0.009, +0.016] |
| MedSAM2/BUS-BRA | C5d_dilated-mask-only | 0.935 [0.917, 0.951] | +0.048 [+0.026, +0.069] |
| MedSAM2/BUS-BRA | C6d_dilated-geometry-rendering | 0.927 [0.909, 0.945] | +0.041 [+0.017, +0.062] |
| MedSAM2/BUS-BRA | dilated geometry features | 0.931 [0.913, 0.947] | +0.044 [+0.023, +0.065] |
| DINOv3/BUS-UCLM | C5d_dilated-mask-only | 0.967 [0.892, 1.000] | +0.080 [+0.010, +0.163] |
| DINOv3/BUS-UCLM | C6d_dilated-geometry-rendering | 0.925 [0.824, 0.993] | +0.038 [-0.049, +0.124] |
| DINOv3/BUS-UCLM | dilated geometry features | 0.889 [0.752, 0.984] | +0.003 [-0.134, +0.131] |

## Interpretation gate

Dilated intervention-mask controls remain highly label-decodable in the primary BUS-BRA/DINOv3 setting, supporting the bounded claim that intervention-mask geometry is predictive in isolation.

## Features

sqrt_normalized_area, normalized_perimeter, compactness, eccentricity, solidity, bbox_aspect_ratio, centroid_x, centroid_y, lesion_to_image_area_ratio
