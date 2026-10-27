# BIBM primary endpoints

Patient-level AUROC summaries generated from stored prediction CSV files. Bootstrap CIs use stratified patient resampling.

| Setting | Condition | Patients | AUROC [95% CI] |
| --- | --- | ---: | ---: |
| DINOv3/BUS-BRA | C1_full | 1064 | 0.881 [0.857, 0.901] |
| DINOv3/BUS-BRA | C2_lesion-only | 1064 | 0.942 [0.927, 0.956] |
| DINOv3/BUS-BRA | C4_lesion-removed-inpaint | 1064 | 0.927 [0.912, 0.942] |
| DINOv3/BUS-BRA | C5_mask-only | 1064 | 0.946 [0.930, 0.959] |
| DINOv3/BUS-BRA | C6_shape-only | 1064 | 0.941 [0.926, 0.955] |
| ResNet-50/BUS-BRA | C1_full | 1064 | 0.843 [0.818, 0.867] |
| ResNet-50/BUS-BRA | C2_lesion-only | 1064 | 0.936 [0.919, 0.951] |
| ResNet-50/BUS-BRA | C4_lesion-removed-inpaint | 1064 | 0.914 [0.896, 0.930] |
| MedSAM2/BUS-BRA | C1_full | 1064 | 0.849 [0.824, 0.871] |
| MedSAM2/BUS-BRA | C2_lesion-only | 1064 | 0.939 [0.922, 0.954] |
| MedSAM2/BUS-BRA | C4_lesion-removed-inpaint | 1064 | 0.887 [0.867, 0.907] |
| MedSAM2/BUS-BRA | C5_mask-only | 1064 | 0.937 [0.920, 0.953] |
| MedSAM2/BUS-BRA | C6_shape-only | 1064 | 0.928 [0.909, 0.945] |
| UltraSam/BUS-BRA | C1_full | 1064 | 0.858 [0.834, 0.880] |
| UltraSam/BUS-BRA | C2_lesion-only | 1064 | 0.919 [0.898, 0.937] |
| UltraSam/BUS-BRA | C4_lesion-removed-inpaint | 1064 | 0.841 [0.814, 0.865] |
| UltraSam/BUS-BRA | C5_mask-only | 1064 | 0.932 [0.915, 0.947] |
| UltraSam/BUS-BRA | C6_shape-only | 1064 | 0.908 [0.888, 0.926] |
| DINOv3/BUS-UCLM | C1_full | 35 | 0.680 [0.480, 0.866] |
| DINOv3/BUS-UCLM | C2_lesion-only | 35 | 0.954 [0.863, 1.000] |
| DINOv3/BUS-UCLM | C4_lesion-removed-inpaint | 35 | 0.886 [0.755, 0.984] |
| DINOv3/BUS-UCLM | C5_mask-only | 35 | 0.971 [0.902, 1.000] |
| DINOv3/BUS-UCLM | C6_shape-only | 35 | 0.928 [0.830, 0.997] |
