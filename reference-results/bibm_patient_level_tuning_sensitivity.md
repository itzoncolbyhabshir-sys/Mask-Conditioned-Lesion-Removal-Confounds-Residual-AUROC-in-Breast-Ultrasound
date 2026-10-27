# BIBM patient-level inner-CV tuning sensitivity

The logistic penalty was selected with patient-level AUROC inside group-aware inner folds. Outer folds, labels, and patient-level reporting match the main protocol.

| Endpoint | Patients B/M | AUROC [95% CI] | Delta vs patient-tuned C4 |
| --- | ---: | ---: | ---: |
| Lesion-removed (C4) | 722/342 | 0.927 [0.912, 0.942] | reference |
| Dilated mask-only (C5d) | 722/342 | 0.947 [0.932, 0.960] | +0.020 [+0.008, +0.031] |
| Dilated geometry rendering (C6d) | 722/342 | 0.936 [0.917, 0.952] | +0.008 [-0.004, +0.020] |
| Explicit dilated-mask geometry | 722/342 | 0.931 [0.912, 0.947] | +0.003 [-0.009, +0.016] |

## Claim gate

Patient-level inner-CV tuning preserves the qualitative source-ambiguity pattern for the primary DINOv3/BUS-BRA controls.
