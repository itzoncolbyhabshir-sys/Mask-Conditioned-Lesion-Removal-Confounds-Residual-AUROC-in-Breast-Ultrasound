# 正文实验复现 SOP

对应论文：**Mask-Conditioned Lesion Removal Confounds Residual AUROC in Breast Ultrasound**。版本：2026-10-26。所有命令在仓库根目录运行。

## 1. 环境和输入

按[数据与模型说明](DATA_AND_MODELS.md)准备 BUS-BRA、BUS-UCLM V3 和四编码器权重。先完成一个 ResNet-50 条件验证，再运行正文矩阵。

```bash
python -m venv .venv
```

Windows：`.venv/Scripts/Activate.ps1`；Linux/macOS：`source .venv/bin/activate`。按 [PyTorch 官方选择器](https://pytorch.org/get-started/locally/) 安装匹配驱动的 torch/torchvision，随后：

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pip check
python -c "import torch,torchvision,timm,sklearn,cv2; print(torch.__version__,torchvision.__version__,timm.__version__,sklearn.__version__,cv2.__version__); print(torch.cuda.is_available(),torch.version.cuda); assert timm.is_model('vit_large_patch16_dinov3.lvd1689m')"
python -m pip freeze > environment.freeze.txt
python -m torch.utils.collect_env > environment.torch.txt
```

根 requirements 为版本范围，不是历史环境 lock。记录 Python、CUDA/驱动、GPU、各库版本、外部 MedSAM2 commit、权重哈希、batch size、耗时。当前可用测试环境不等于已确认的论文原训练环境。

## 2. 构建输入并固定划分

```bash
python run_pipeline.py --stage infra
python tools/restore_reference_splits.py --apply
python tools/restore_reference_splits.py
```

完整双数据集 manifest 应为 2,135 行：BUS-BRA 1,875 图像 / 1,064 患者，BUS-UCLM 260 图像 / 35 患者。BUS-UCLM 排除正常图像和四个尺寸错误的良性病灶；保留图像级病灶标签。

基础设施生成本机 manifest、处理后的掩膜、七个基础条件 C1/C2/C4/C5/C6/C5d/C6d 和 splits。固定形状及 NS 条件在后续专用脚本生成。

恢复工具按数据集、图像文件名、患者、标签匹配历史折号，不依赖原机器绝对路径；拒绝样本缺失、重复、患者/标签不一致和跨折。仅安装 BUS-BRA 时核对其完整队列即可。恢复前自动备份 manifest/splits；已有备份时拒绝覆盖。

**先恢复划分，再提特征和训练。** 之后不要重新运行 infra 或 src.splits 覆盖固定折号。每名患者只属于一个外层折；2026–2030 五个种子用于下游拟合/内层 CV，外层登记固定。

## 3. 单条件验证

```bash
python -m src.features --model ResNet-50 --dataset BUS-BRA --condition C1_full --device cuda --batch_size 8
python -m src.probe --model ResNet-50 --dataset BUS-BRA --condition C1_full
```

验收：`features/ResNet-50/BUS-BRA/C1_full.npy` 为 1,875×2,048，所有值有限；`C1_full_ids.json` 与矩阵行序一致；预测目录有五折×五种子文件。单种子可排错，但不能替代正式五种子结果。CPU 入口可将 `--device cuda` 改为 `--device cpu`。

## 4. 基础端点与原始掩膜控制

```bash
python run_pipeline.py --stage exp1 --models ResNet-50 DINOv3 MedSAM2 UltraSam --datasets BUS-BRA
python run_pipeline.py --stage exp1 --models DINOv3 --datasets BUS-UCLM
python run_pipeline.py --stage exp2 --models DINOv3 MedSAM2 UltraSam --datasets BUS-BRA
python run_pipeline.py --stage exp2 --models DINOv3 --datasets BUS-UCLM
python scripts/run_bibm_geometry_feature_baseline.py
python scripts/run_bibm_primary_endpoints.py
```

exp1 对应 C1/C2/C4；exp2 对应 C4/C5/C6。C6 是几何渲染图经编码器的分类，不能与数值几何特征混为一项。数值几何脚本固定处理两个数据集，需要两者 C4 预测。

只运行 DINOv3/BUS-BRA 最小主线时，模型收窄到 DINOv3，跳过 BUS-UCLM 命令及 `run_bibm_geometry_feature_baseline.py`，随后仅运行下一节的 `--settings DINOv3/BUS-BRA`。子集运行与完整矩阵应分别保存，避免覆盖报告。

## 5. 干预掩膜几何与固定形状

```bash
python scripts/run_bibm_dilated_mask_controls.py --settings DINOv3/BUS-BRA MedSAM2/BUS-BRA DINOv3/BUS-UCLM --batch-size 16 --device cuda
python scripts/run_fixed_shape_controls.py --model DINOv3 --dataset BUS-BRA
python scripts/run_fixed_shape_controls.py --model MedSAM2 --dataset BUS-BRA
python scripts/run_fixed_shape_controls.py --model UltraSam --dataset BUS-BRA
python scripts/run_fixed_shape_controls.py --model DINOv3 --dataset BUS-UCLM
python scripts/refresh_bibm_fixed_shape_statistics.py
python scripts/audit_fixed_shape_coverage.py
python scripts/run_bibm_geometry_recoverability.py
```

膨胀控制生成 C5d/C6d 并训练显式几何。固定形状脚本仅生成圆、方、椭圆的 C10–C12；覆盖率另行审计。几何回归固定使用 DINOv3/BUS-BRA，需要 C1/C4/C5d/C6d/C10/C11/C12 特征齐备。

保留原固定形状生成规则：名义匹配膨胀掩膜面积和中心。实际覆盖并非 100%，所以 Fig. 3B 表示形状与覆盖率的联合变化，不能说是单独轮廓效应。配对图示差值为 **control − C4**，负值表示控制条件低于 C4。

## 6. 正文报告的敏感性

按本表顺序和前置条件运行：

| 命令 | 前置产物 | 正文位置 |
|---|---|---|
| `python scripts/run_inpaint_sensitivity.py` | BUS-BRA 清单/膨胀掩膜、DINOv3 C4 | V-A：Telea / Navier–Stokes |
| `python scripts/run_bibm_operator_sensitivity.py` | 上一行条件的特征与预测、C1 特征/预测 | V-C：算子下几何恢复 |
| `python scripts/run_bibm_patient_balanced_sensitivity.py` | C4/C5d/C6d 特征 | V-B：患者等权训练 |
| `python scripts/run_bibm_patient_level_tuning_sensitivity.py` | C4/C5d/C6d 特征 | IV-E / V-B：患者级内层调参 |
| `python scripts/run_bibm_patient_label_sensitivity.py` | BUS-UCLM 端点与显式几何预测 | V-A：混合标签及聚合 |
| `python scripts/run_bibm_recoverability_split_sensitivity.py` | DINOv3/BUS-BRA C4 特征与膨胀掩膜 | V-C：替代患者划分 |
| `python scripts/run_bibm_fully_covered_geometry_sensitivity.py` | 分类和几何恢复预测 | V-B/V-C：裁剪覆盖 ≥0.95 子集 |

最后可核查预处理、膨胀尺度和内层 CV：

```bash
python scripts/audit_bibm_preprocessing.py
python scripts/audit_bibm_dilation_scale.py
python scripts/audit_bibm_probe_cv.py
python scripts/make_bibm_figures.py
```

图表脚本需要实际条件图与对应结果，输出 `paper/bibm2026_label_decodability_audit/figures/`。生成后仍需对照论文检查图文，不将自动生成视为版面验证。

## 7. 统计与验收

- 分类器为训练折内 StandardScaler + L2 LogisticRegression，lbfgs、class_weight=balanced、max_iter=5000；C 网格 0.001–1000，患者分组内层 CV，默认按图像 AUROC 选 C。
- 按同图种子预测均值、同患者图像概率均值聚合；患者标签 any-malignant；不更改训练图像标签。
- 主指标为患者级 pooled OOF AUROC；95% CI 用 1,000 次分层患者 bootstrap。正文配对比较的 CI 未进行多重比较校正；含零不代表等效。
- 每条件 25 次拟合；每样本每种子一条 OOF 预测，合并 N×5 行，不能误要求 N×25 行。聚合后 BUS-BRA 1,064 患者、BUS-UCLM 35 患者。
- Ridge 几何恢复使用一套外层五折，输入/目标在训练折内标准化；按 target/fold 选择 alpha，报告八个非冗余目标。它不沿用分类的五种子均值规则。
- 核对有限特征值、完整 fold/seed、无患者泄漏、模型哈希和维度；MedSAM2 为 256 维。
- 新结果写 `results/`，已有正文参考在 `reference-results/`；不能复制参考文件充当新运行产物。

```bash
python -m pytest -q tests
python -m unittest discover -s tools -p "test_*.py"
```

改变权重、预处理、条件图或划分后，应隔离旧预测并重新提特征。模块 CLI 的覆盖参数为 `--force`，batch 参数为 `--batch_size`；脚本部分使用 `--batch-size`，以各自帮助为准。基础阶段结束后仍需核对样本与图像数量，缺失缓存可能导致某些汇总仅含子集。

## 8. 流程参考

本 SOP 参考 [NeurIPS 复现清单](https://neurips.cc/public/guides/PaperChecklist)、[ACM artifact 文档要求](https://conferences.sigcomm.org/sigcomm/2026/artifacts/)和 [PyTorch 可复现性说明](https://docs.pytorch.org/docs/stable/notes/randomness.html)，组织输入、命令、环境与验收。它不是外部机构对本项目的认证。跨软件/硬件版本不承诺逐位一致。
