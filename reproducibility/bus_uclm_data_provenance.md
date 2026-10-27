# BUS-UCLM data provenance and QC

- Retrieved: 2026-06-20
- Official record: Mendeley Data `7fvgj4jsp7`, version 3
- DOI: `10.17632/7fvgj4jsp7.3`
- Official files: 683 images plus 683 paired masks
- Integrity verification: 1,366/1,366 local files match the official per-file SHA256 metadata
- Local metadata: `data/bus_uclm_images_files.json`, `data/bus_uclm_masks_files.json`
- Reproducible downloader/verifier: `scripts/download_bus_uclm.py`

## Mask-derived inventory

| Mask class | Official pairs |
|---|---:|
| Normal (black) | 419 |
| Benign (green) | 174 |
| Malignant (red) | 90 |
| Total | 683 |

Normal records are outside the binary lesion endpoint. Thirteen official image/mask pairs have unequal spatial dimensions. Nine are normal; four are benign lesion records (`HESN_002.png` through `HESN_005.png`). The four lesion records were excluded because their aspect ratios differ and no authoritative registration transform is supplied. The final binary cohort is therefore 260 images: 170 benign and 90 malignant.

All included masks passed binary conversion and non-empty-mask QC. Patient-disjoint folds contain 35 patients with no train/test patient overlap.
