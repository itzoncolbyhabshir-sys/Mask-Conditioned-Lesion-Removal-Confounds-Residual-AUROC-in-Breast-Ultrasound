# Fully-covered geometry sensitivity

Primary DINOv3/BUS-BRA images were filtered to dilated-mask crop coverage >= 0.95.

| Images retained | Patients retained | Below 0.95 rate | Below 0.90 rate |
| ---: | ---: | ---: | ---: |
| 1632 / 1875 | 955 / 1064 | 0.130 | 0.095 |

## Main endpoints on retained images

| Endpoint | Patients B/M | AUROC [95% CI] | Delta vs C4 |
| --- | ---: | ---: | ---: |
| Lesion-removed (C4) | 641/314 | 0.931 [0.914, 0.947] | -- |
| Dilated mask-only (C5d) | 641/314 | 0.948 [0.933, 0.962] | +0.017 [+0.005, +0.030] |
| Dilated geometry rendering (C6d) | 641/314 | 0.939 [0.923, 0.955] | +0.008 [-0.005, +0.021] |
| Explicit dilated-mask geometry | 641/314 | 0.932 [0.914, 0.949] | +0.001 [-0.012, +0.015] |

## C4 recoverability on retained images

Median core Spearman: 0.962; median core R2: 0.921.

| Core target | Spearman | R2 |
| --- | ---: | ---: |
| sqrt_normalized_area | 0.986 | 0.973 |
| normalized_perimeter | 0.978 | 0.958 |
| centroid_x | 0.684 | 0.460 |
| centroid_y | 0.946 | 0.885 |

## Claim gate

The fully-covered subset preserves the direction of the primary C4-aligned mask-control, geometry endpoint, and C4 recoverability findings.
