# 正文实验登记表

范围依据用户提供的八页论文正文 Table IV、Fig. 2、Fig. 3、Table V 及 Section V。完整运行顺序见 [SOP](REPRODUCIBILITY_SOP.md)，环境与 checkpoint 见[数据和模型](DATA_AND_MODELS.md)。

## 实验到正文和产物的映射

| 实验 | 正文位置 | 入口 | 主要产物（新运行） |
|---|---|---|---|
| C1/C2/C4 分类 | V-A、Fig. 2A | `run_pipeline.py --stage exp1` | 每条件特征、OOF 预测、`results/exp1_summary.json` |
| C5/C6 原始掩膜 | V-B、Table IV | `run_pipeline.py --stage exp2` | 原始掩膜及几何渲染的特征/预测 |
| 显式原始几何 | IV-B、V-B | `scripts/run_bibm_geometry_feature_baseline.py` | `results/bibm_geometry_feature_baseline.json` |
| C5d/C6d 与膨胀几何 | V-B、Fig. 2 | `scripts/run_bibm_dilated_mask_controls.py` | `results/bibm_dilated_mask_controls.json`；显式几何预测 CSV |
| 固定圆/方/椭圆 | V-D、Fig. 3B | `scripts/run_fixed_shape_controls.py` | `results/fixed_shape_controls*.json`；C10–C12 特征/预测 |
| 固定形状覆盖率 | V-D | `scripts/audit_fixed_shape_coverage.py` | `results/fixed_shape_coverage.json` |
| 几何可恢复性 | V-C、Table V、Fig. 3A | `scripts/run_bibm_geometry_recoverability.py` | `results/bibm_geometry_recoverability.json` 与逐目标预测 |
| 主端点汇总 | V-A/B | `scripts/run_bibm_primary_endpoints.py` | `results/bibm_primary_endpoints.json` / `.md` |
| NS 修复及几何恢复 | V-A/C | `scripts/run_inpaint_sensitivity.py` → `scripts/run_bibm_operator_sensitivity.py` | NS 条件缓存、`results/bibm_operator_sensitivity.json` |
| 患者等权 | V-B | `scripts/run_bibm_patient_balanced_sensitivity.py` | `results/bibm_patient_balanced_sensitivity.json` |
| 患者 AUROC 内层调参 | IV-E、V-B | `scripts/run_bibm_patient_level_tuning_sensitivity.py` | `results/bibm_patient_level_tuning_sensitivity.json` |
| BUS-UCLM 混合标签/聚合 | V-A | `scripts/run_bibm_patient_label_sensitivity.py` | `results/bibm_patient_label_sensitivity.json` |
| 替代几何回归划分 | V-C | `scripts/run_bibm_recoverability_split_sensitivity.py` | `results/bibm_recoverability_split_sensitivity.json` |
| 裁剪覆盖 ≥0.95 | V-B/C | `scripts/run_bibm_fully_covered_geometry_sensitivity.py` | `results/bibm_fully_covered_geometry_sensitivity.json` |

表中入口名称省略 `python`；完整参数可直接复制 SOP。固定形状统计刷新由 `scripts/refresh_bibm_fixed_shape_statistics.py` 完成，使用相同患者分层配对 bootstrap。

## 固定设置

| 项目 | 定义 |
|---|---|
| BUS-BRA | 1,875 图像 / 1,064 患者，图像良/恶性 1,268/607 |
| BUS-UCLM | 260 图像 / 35 患者，图像良/恶性 170/90；5 位混合标签患者 |
| 外层分类划分 | 固定五折，按患者互斥；复用 `reproducibility/splits_reference.csv` |
| 分类重复种子 | 2026、2027、2028、2029、2030，不重分外层折 |
| 探针 | StandardScaler + L2 LogisticRegression；平衡类权重、lbfgs、max_iter=5000 |
| 调参 | C∈{0.001,0.01,0.1,1,10,100,1000}；训练折内患者分组 CV |
| C4 | 原图空间 5×5 椭圆核膨胀，Telea radius=5 |
| 主指标 | 患者级 pooled OOF AUROC，1,000 次分层患者 bootstrap CI |
| 几何恢复 | Ridge，训练折内输入/目标标准化；八个正文非冗余目标 |

## 模型与报告角色

| 模型 | 输出维度 | 正文角色 |
|---|---:|---|
| DINOv3 ViT-L/16 | 1,024 | BUS-BRA 主设置；BUS-UCLM 小队列分析 |
| ResNet-50 | 2,048 | C1/C2/C4 编码器比较 |
| MedSAM2 Hiera-Tiny | 256 | 编码器、足迹控制和固定形状分析 |
| UltraSam ViT-B | 768 | 可能存在数据集级训练暴露的敏感性分析 |

固定形状对照是正文自身列出的探索性分析，保留这个统计解释，不将其升级为独立因果归因。

## 标准中间产物

```text
features/{model}/{dataset}/{condition}.npy
features/{model}/{dataset}/{condition}_ids.json
predictions/{model}/{dataset}/{condition}/fold{0..4}_seed{2026..2030}.csv
results/{model}/{dataset}/{condition}_results.csv
results/{model}/{dataset}/{condition}_all_predictions.csv
```

每条件结果表应有 25 个 fold/seed 组合，完整 OOF 合并为 N×5 行。患者聚合后计数必须吻合。几何恢复的回归目标/折输出按独立 schema 验收，不把它当成五种子分类预测。

## 输出解释

C5d/C6d/显式几何对照与 C4 配对；正的 control−C4 表示对照更可解码。固定形状的图示差值同样是 fixed−C4。不能从零附近的 CI 推断统计等效，也不能把固定形状的差异解读成纯轮廓作用。BUS-UCLM 在自己的患者折内重新训练探针。

冻结编码器、分类拟合、患者统计和几何回归的计算保持现有实现。当前包删除了正文之外的入口和模块，不提供历史全流程开关。
