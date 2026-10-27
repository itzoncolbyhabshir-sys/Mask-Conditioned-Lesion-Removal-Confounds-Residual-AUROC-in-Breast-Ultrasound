# 正文版发布清单

本包只保留论文正文实验及运行它们所需的数据、模型、划分和验证工具。代码和文档已按正文同步裁剪。

## 上传内容

将 `github-bibm` 内文件作为 GitHub 仓库根目录上传，保留 `.gitignore`。不要上传原研究工作区。ZIP 适合传递；解压后的文件便于直接浏览和运行。

- `README.md` 使用正式论文题名。
- `docs/` 提供正文实验对照、下载准备及逐步 SOP。
- `src/`、`scripts/`、`tests/`、`tools/` 只保留正文执行和验证路径。
- `reference-results/` 是正文已有聚合结果，和新运行的 `results/` 分开。
- `reproducibility/` 保存固定划分、权重哈希和数据来源；不含本机绝对路径。
- `.gitignore` 排除原始图像、权重、大型缓存、凭据、临时文件。

## 仍需补齐的发布信息

| 项目 | 状态 / 动作 |
|---|---|
| 历史训练环境 | 依赖范围尚未完全锁定；验证环境见验证记录，不能当成原实验环境 |
| MedSAM2 源码版本 | 历史 ZIP 未记录精确 commit；正式复现应锁定源码版本 |
| 完整 GPU 数值复现 | 本次裁剪验证代码与依赖，不等于全量重训论文 |
| 数据与权重 | 由使用者从官方来源取得，核对哈希 |
| 许可证 | 原项目声明 MIT 意向；维护者补充版权人与正式 LICENSE，第三方条款独立 |
| 仓库 URL | [公开仓库](https://github.com/itzoncolbyhabshir-sys/Mask-Conditioned-Lesion-Removal-Confounds-Residual-AUROC-in-Breast-Ultrasound) |
| 版本 / DOI | 尚未创建版本标签或 DOI |

固定形状覆盖不足、UltraSam 训练暴露和小队列不确定性属于论文结论边界，保留在相应实验说明中。
