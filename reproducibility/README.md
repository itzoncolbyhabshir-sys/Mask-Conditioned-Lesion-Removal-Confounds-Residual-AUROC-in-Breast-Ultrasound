# 复现来源记录

- `splits_reference.csv`：从原 `splits.csv` 导出，保留数据集、原始图像文件名、编码患者 ID、良恶性标签和固定外层折号。共 2,135 行（BUS-BRA 1,875；BUS-UCLM 260）。不含本机绝对路径。
- `seed_checkpoint_hashes.txt`：原权重与种子记录。是历史锁定值；请对新取得的权重重新计算。
- `validation_environment.json`：本次正文包导入与回归测试所用环境版本，不当作历史训练环境。
- `bus_uclm_data_provenance.md`：原下载校验及纳排记录，来源统计与本包数据说明相互核对。

在新机器先重建 manifest，再运行 `python tools/restore_reference_splits.py --apply`。本目录不提供原始医学图像，也不替代数据提供者的许可条款。
