# BIBM preprocessing and crop audit

Encoder preprocessing was read from the project feature-extraction code and summarized for manuscript reproducibility. Mask resizing for condition generation uses nearest-neighbor interpolation when mask semantics are required.

## Encoder preprocessing

| Encoder | Input geometry | Crop/padding | Interpolation/normalization | Feature | Dim |
| --- | --- | --- | --- | --- | ---: |
| ResNet-50 | RGB conversion from grayscale-compatible source; custom Resize short side to 224 | CenterCrop 224 x 224 | bilinear image resize; ImageNet normalization | global average pooled penultimate feature | 2048 |
| DINOv3 | grayscale-to-3-channel, timm resolved eval transform, input 256 x 256 | Resize short side to 256 and CenterCrop 256 x 256 (crop_pct=1.0) | bicubic image resize with antialias; ImageNet normalization | CLS token | 1024 |
| MedSAM2 | RGB conversion and direct Resize 512 x 512 | no crop; aspect ratio is stretched to square | bilinear image resize; masks/conditions use nearest neighbor where mask semantics are required | spatial mean of `image_encoder(images)["vision_features"]` | 256 |
| UltraSam | RGB conversion, keep-ratio resize to 1024 long side | zero padding on 1024 x 1024 canvas | bilinear image resize; ultrasound checkpoint normalization | mean pooled encoder tokens | 768 |

## Crop coverage for lesion masks

| Dataset | Encoder/mask | Median | IQR | Min | <0.90 rate |
| --- | --- | ---: | ---: | ---: | ---: |
| BUS-BRA | ResNet-50_original_mask | 1.000 | 1.000--1.000 | 0.048 | 0.088 |
| BUS-BRA | ResNet-50_dilated_mask | 1.000 | 1.000--1.000 | 0.069 | 0.094 |
| BUS-BRA | DINOv3_original_mask | 1.000 | 1.000--1.000 | 0.048 | 0.087 |
| BUS-BRA | DINOv3_dilated_mask | 1.000 | 1.000--1.000 | 0.069 | 0.095 |
| BUS-BRA | MedSAM2_original_mask | 1.000 | 1.000--1.000 | 1.000 | 0.000 |
| BUS-BRA | MedSAM2_dilated_mask | 1.000 | 1.000--1.000 | 1.000 | 0.000 |
| BUS-BRA | UltraSam_original_mask | 1.000 | 1.000--1.000 | 1.000 | 0.000 |
| BUS-BRA | UltraSam_dilated_mask | 1.000 | 1.000--1.000 | 1.000 | 0.000 |
| BUS-UCLM | ResNet-50_original_mask | 1.000 | 1.000--1.000 | 0.000 | 0.092 |
| BUS-UCLM | ResNet-50_dilated_mask | 1.000 | 1.000--1.000 | 0.000 | 0.096 |
| BUS-UCLM | DINOv3_original_mask | 1.000 | 1.000--1.000 | 0.000 | 0.092 |
| BUS-UCLM | DINOv3_dilated_mask | 1.000 | 1.000--1.000 | 0.000 | 0.096 |
| BUS-UCLM | MedSAM2_original_mask | 1.000 | 1.000--1.000 | 1.000 | 0.000 |
| BUS-UCLM | MedSAM2_dilated_mask | 1.000 | 1.000--1.000 | 1.000 | 0.000 |
| BUS-UCLM | UltraSam_original_mask | 1.000 | 1.000--1.000 | 1.000 | 0.000 |
| BUS-UCLM | UltraSam_dilated_mask | 1.000 | 1.000--1.000 | 1.000 | 0.000 |
