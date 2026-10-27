# Interpretable mask-geometry baseline

Logistic regression on explicit original-mask geometry features, using the same outer folds as the image-feature probes. The C4 dilated intervention mask is not used.

| Dataset | Patients | AUROC [95% CI] | Delta vs DINOv3 C4 |
| --- | ---: | --- | --- |
| BUS-BRA | 1064 | 0.932 [0.915, 0.948] | +0.005 [-0.008, +0.018] |
| BUS-UCLM | 35 | 0.889 [0.759, 0.980] | +0.002 [-0.140, +0.137] |

## Features

sqrt_normalized_area, normalized_perimeter, compactness, eccentricity, solidity, bbox_aspect_ratio, centroid_x, centroid_y, lesion_to_image_area_ratio
