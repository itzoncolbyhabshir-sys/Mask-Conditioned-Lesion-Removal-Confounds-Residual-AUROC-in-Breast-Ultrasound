# Mask-Conditioned Lesion Removal Confounds Residual AUROC in Breast Ultrasound

论文正文实验复现包 · Liqi Huang and Xuan Zhang

[GitHub 仓库](https://github.com/itzoncolbyhabshir-sys/Mask-Conditioned-Lesion-Removal-Confounds-Residual-AUROC-in-Breast-Ultrasound)

本仓库对应论文正文的乳腺超声审计实验：在 BUS-BRA / BUS-UCLM 上构造病灶移除及掩膜几何对照，使用冻结编码器和按条件独立拟合的逻辑回归探针，研究残余 AUROC 的信息来源能否被识别。

**核心结论：病灶移除后仍然能预测标签，但高 AUROC 本身不能区分病灶外图像、掩膜几何及编辑痕迹的贡献。**

## 正文实验范围

| 论文位置 | 实验 | 对应内容 |
|---|---|---|
| V-A / Fig. 2A | 完整图像、病灶保留、病灶移除 | C1 / C2 / C4 |
| V-B / Fig. 2 | 干预掩膜几何控制 | C5d / C6d / 显式几何特征 |
| V-C / Table V / Fig. 3A | 几何可恢复性 | 冻结特征 → Ridge → 八项几何目标 |
| V-D / Fig. 3B | 编辑形状与覆盖率 | C10 / C11 / C12 圆、方、椭圆 |
| IV–V / Table IV | 正文报告的次要分析 | C5/C6、其他编码器、BUS-UCLM、修复算子、患者平衡/调参/标签、替代划分、裁剪覆盖率 |

正文 Fig. 3 的固定形状分析属于论文报告的探索性结果，因此保留其实现与解释边界。仓库不包含正文之外的历史实验或占位模块。

## 使用文档

- [逐步复现 SOP](docs/REPRODUCIBILITY_SOP.md)
- [正文实验、命令与产物对照](docs/EXPERIMENTS.md)
- [数据与模型权重准备](docs/DATA_AND_MODELS.md)
- [论文对应表](docs/PAPER_ALIGNMENT.md)
- [验证记录](docs/VALIDATION.md)与[发布清单](docs/RELEASE_CHECKLIST.md)

## 参考结果

以下是已有实验产物中的患者级 pooled OOF AUROC，不是本次裁剪重新训练所得。

| 设置 | C1 | C2 | C4 |
|---|---:|---:|---:|
| DINOv3 / BUS-BRA | 0.881 | 0.942 | 0.927 |
| ResNet-50 / BUS-BRA | 0.843 | 0.936 | 0.914 |
| MedSAM2 / BUS-BRA | 0.849 | 0.939 | 0.887 |
| UltraSam / BUS-BRA | 0.858 | 0.919 | 0.841 |
| DINOv3 / BUS-UCLM | 0.680 | 0.954 | 0.886 |

DINOv3 / BUS-BRA 的 C5d、C6d、显式膨胀掩膜几何 AUROC 分别为 0.945、0.935、0.931。八项几何恢复的 Spearman 中位数为 0.877。完整 CI 见 [reference-results](reference-results/README.md)。

## 最短入口验证

准备数据和环境后，在仓库根目录执行：

```bash
python run_pipeline.py --stage infra
python tools/restore_reference_splits.py --apply
python -m src.features --model ResNet-50 --dataset BUS-BRA --condition C1_full --device cuda --batch_size 8
python -m src.probe --model ResNet-50 --dataset BUS-BRA --condition C1_full
```

这是一个条件的运行验证。完整正文实验顺序见 SOP。外层为一套固定患者互斥五折，探针种子为 2026–2030；不是五套外层划分。

## 内容与状态

`src/` 为共享计算和正文基础实验；`scripts/` 为正文控制、敏感性与图表；`tools/` 提供固定划分恢复和包检查；`reproducibility/` 保存 2,135 条可移植划分及来源；`reference-results/` 保存正文参考汇总，新结果写入 `results/`。

BUS-BRA：1,875 图像 / 1,064 患者。BUS-UCLM 纳排后：260 图像 / 35 患者。原始图像和模型权重按官方链接另行获取。

本版按正文裁剪了入口和模块，保留正文计算定义；测试情况见验证记录。完整历史环境尚未锁定，本次未全量重训模型。代码许可沿用原项目 MIT 意向，正式授权文本仍待维护者补齐；第三方数据和权重使用自身条款。
