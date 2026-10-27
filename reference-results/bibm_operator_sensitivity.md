# BIBM operator sensitivity

DINOv3/BUS-BRA patient-level operator sensitivity for the lesion-removed C4 endpoint. This is a focused sensitivity check, not a full inpainting-operator matrix.

| Operator | C4 AUROC [95% CI] | C4-Full delta | Geometry recoverability |
| --- | ---: | ---: | ---: |
| Telea | 0.927 [0.912, 0.942] | +0.046 | 0.965 footprint rho |
| Navier-Stokes | 0.922 [0.905, 0.938] | +0.041 | 0.963 footprint rho |

Navier-Stokes minus Telea C4 AUROC delta: -0.005 (95% CI -0.012 to +0.001).
