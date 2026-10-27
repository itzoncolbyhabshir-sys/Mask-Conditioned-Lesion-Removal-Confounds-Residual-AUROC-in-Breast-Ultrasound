# Geometry recoverability from DINOv3 representations

Ridge regressors predict dilated-mask geometry from frozen image representations using patient-disjoint folds. Intervals are patient-cluster bootstrap 95% CIs.

The manuscript-facing summary uses eight non-redundant dilated-mask targets, excluding the raw mask-to-image area ratio because it is redundant with the square-root area-ratio target. The full nine-target audit trace is preserved separately.

Core targets summarize quantities that directly encode mask size/location and the intervention footprint: area, perimeter, centroid x, and centroid y.

Fixed-shape edits approximately match dilated-mask area and center before rasterization; their recoverability values are procedural checks rather than new evidence of learned recovery.

| Representation | Median 8-target Spearman | Median core Spearman | Median core R2 | Core target details |
| --- | ---: | ---: | ---: | --- |
| Full image | 0.660 | 0.789 | 0.643 | sqrt_normalized_area: rho=0.800, R2=0.669; normalized_perimeter: rho=0.777, R2=0.617; centroid_x: rho=0.420, R2=0.183; centroid_y: rho=0.849, R2=0.701 |
| Lesion-removed C4 | 0.877 | 0.965 | 0.928 | sqrt_normalized_area: rho=0.985, R2=0.973; normalized_perimeter: rho=0.977, R2=0.957; centroid_x: rho=0.676, R2=0.442; centroid_y: rho=0.953, R2=0.899 |
| Dilated mask-only C5d | 0.975 | 0.986 | 0.971 | sqrt_normalized_area: rho=0.992, R2=0.979; normalized_perimeter: rho=0.991, R2=0.977; centroid_x: rho=0.978, R2=0.958; centroid_y: rho=0.981, R2=0.965 |
| Dilated geometry C6d | 0.982 | 0.991 | 0.981 | sqrt_normalized_area: rho=0.993, R2=0.985; normalized_perimeter: rho=0.994, R2=0.984; centroid_x: rho=0.989, R2=0.978; centroid_y: rho=0.986, R2=0.974 |
| Fixed circle | 0.719 | 0.969 | 0.931 | sqrt_normalized_area: rho=0.987, R2=0.977; normalized_perimeter: rho=0.972, R2=0.937; centroid_x: rho=0.766, R2=0.575; centroid_y: rho=0.967, R2=0.925 |
| Fixed square | 0.690 | 0.962 | 0.920 | sqrt_normalized_area: rho=0.983, R2=0.972; normalized_perimeter: rho=0.966, R2=0.931; centroid_x: rho=0.737, R2=0.533; centroid_y: rho=0.958, R2=0.908 |
| Fixed ellipse | 0.727 | 0.964 | 0.922 | sqrt_normalized_area: rho=0.986, R2=0.974; normalized_perimeter: rho=0.970, R2=0.936; centroid_x: rho=0.719, R2=0.509; centroid_y: rho=0.958, R2=0.908 |

## Claim gate

C4 representations recover multiple core dilated-mask geometry features; the manuscript may state that geometry information is recoverable from C4 representations while preserving source-ambiguity wording.

## C4 versus full-image recoverability contrast

| Target | Delta Spearman (C4 - Full) | 95% CI |
| --- | ---: | ---: |
| sqrt_normalized_area | +0.185 | [+0.165, +0.207] |
| normalized_perimeter | +0.200 | [+0.179, +0.223] |
| centroid_x | +0.256 | [+0.221, +0.292] |
| centroid_y | +0.104 | [+0.091, +0.119] |
