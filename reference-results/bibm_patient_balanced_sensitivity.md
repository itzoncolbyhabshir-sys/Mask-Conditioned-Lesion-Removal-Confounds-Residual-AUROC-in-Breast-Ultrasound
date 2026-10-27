# BIBM patient-balanced training sensitivity

Image probes were retrained with per-image weights proportional to 1 / patient image count, normalized within each training fold and combined with balanced class weighting. Controls are the formal C4-aligned dilated-mask endpoints.

| Endpoint | AUROC [95% CI] | Delta vs patient-balanced C4 |
| --- | --- | --- |
| Lesion-removed (C4) | 0.928 [0.912, 0.943] | reference |
| Dilated mask-only (C5d) | 0.945 [0.930, 0.959] | +0.017 [+0.006, +0.029] |
| Dilated geometry rendering (C6d) | 0.936 [0.918, 0.952] | +0.007 [-0.005, +0.021] |
| Explicit dilated-mask geometry | 0.932 [0.914, 0.947] | +0.003 [-0.009, +0.016] |
