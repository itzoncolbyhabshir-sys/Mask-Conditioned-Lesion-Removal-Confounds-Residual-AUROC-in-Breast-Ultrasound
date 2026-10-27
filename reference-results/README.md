# 正文参考结果

本目录只保留论文正文相关的已有聚合结果。新实验输出到根目录 `results/`；这里的文件不代表本次裁剪重新训练。

| 正文内容 | 文件 |
|---|---|
| 主端点 | `bibm_primary_endpoints.{json,md}` |
| 干预掩膜控制 | `bibm_dilated_mask_controls.{json,md}` |
| 原始掩膜数值几何 | `bibm_geometry_feature_baseline.{json,md}` |
| 几何恢复 | `bibm_geometry_recoverability.{json,md}` |
| 固定形状 | `fixed_shape_controls*.json` |
| 病灶覆盖 | `fixed_shape_coverage.{json,md}` |
| 正文敏感性 | 算子、患者平衡/调参/标签、替代划分、裁剪覆盖的对应 `bibm_*` 文件 |
| 方法核查 | 预处理、膨胀尺度、患者分组内层 CV 的对应 audit 文件 |

固定形状结果从原混合报告中只提取正文所需字段，数值未改变；主端点删除了正文未采用的条件行。原工作区路径已替换为 `<PROJECT_ROOT>`。

不包含逐样本预测、特征和图像。运行代码可以生成这些产物；如果只想重算既有预测的统计而不重新训练，仍需要维护者另外提供对应 OOF 预测档案。
