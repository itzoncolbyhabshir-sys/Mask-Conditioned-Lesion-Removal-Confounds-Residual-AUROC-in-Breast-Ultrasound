# 论文正文对应核对

论文：**Mask-Conditioned Lesion Removal Confounds Residual AUROC in Breast Ultrasound**。

作者：Liqi Huang and Xuan Zhang。核对依据为用户提供的八页 PDF，日期 2026-10-26。

| 正文位置 | 实验事实 | 本包对应 |
|---|---|---|
| 第 2 页 IV-A / 第 3 页 Table II | BUS-BRA 1,875 图像/1,064 患者；BUS-UCLM 260/35 | 数据说明、固定划分 |
| 第 2 页 IV-B / 第 3 页 Table III | 冻结编码器；每条件逻辑回归；四编码器；MedSAM2 256 维 | `src/features.py`、`src/probe.py` |
| 第 4 页 V-A | C1 0.881、C2 0.942、C4 0.927 | `reference-results/bibm_primary_endpoints.json` |
| 第 4–5 页 V-B / Fig. 2 | C5d 0.945、C6d 0.935、显式几何 0.931 | `reference-results/bibm_dilated_mask_controls.json` |
| 第 5–6 页 V-C / Table V / Fig. 3A | 八目标 Spearman 中位数 0.877；四足迹目标 0.965 | `reference-results/bibm_geometry_recoverability.json` |
| 第 6 页 V-D / Fig. 3B | 圆/方/椭圆改变编辑形状和覆盖率 | `reference-results/fixed_shape_controls*.json`、`fixed_shape_coverage.json` |
| 第 4–6 页方法与结果 | 算子、患者平衡、患者调参、标签聚合、替代划分、裁剪覆盖 | 对应 `run_bibm_*sensitivity.py` |

正文固定形状分析虽然被论文标为 exploratory，仍属于 Fig. 3B 和 V-D，应保留。其余不在正文报告范围内的历史分支已从代码、文档和参考结果中移除。

分类复现使用固定外层登记加五个下游种子；几何恢复使用单套五折。论文 IV-A 简写不应被误读为五套不同外层分类划分。

本次确认实验对应并裁剪发布内容，不表示重新训练或逐字复制 PDF；PDF 本身未放入上传包。
