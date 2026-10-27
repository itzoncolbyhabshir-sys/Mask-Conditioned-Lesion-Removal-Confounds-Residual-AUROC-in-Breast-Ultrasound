# 数据、权重与环境准备

本文对应 BIBM 乳腺超声干预审计代码。所有命令从仓库根目录执行；路径均相对仓库根目录。核对日期：2026-10-26。原始数据和预训练权重由使用者从官方来源取得，不包含在代码包中。

## 1. 数据来源与实际分析队列

| 数据集 | 官方来源 | 原始数据 | 本项目二分类分析队列 |
|---|---|---|---|
| BUS-BRA | [Zenodo 8231412](https://zenodo.org/records/8231412)，DOI `10.5281/zenodo.8231412` | 1,875 图像及配对掩膜 | 1,875 图像，1,064 患者；1,268 良性、607 恶性图像 |
| BUS-UCLM V3 | [Mendeley Data V3](https://data.mendeley.com/datasets/7fvgj4jsp7/3)，DOI `10.17632/7fvgj4jsp7.3` | 683 图像及配对 RGB 掩膜 | 260 图像，35 患者；170 良性、90 恶性图像 |

以上分析队列计数来自现有 `manifest.csv` 的实际逐行统计。配置文件中的 BUS-UCLM `683 images / 38 patients` 是原始数据规模，不能作为最终二分类样本量。

BUS-UCLM 官方掩膜颜色对应：黑色为正常、绿色为良性、红色为恶性。原始分布为 419 正常、174 良性、90 恶性。二分类分析排除正常图像；另有 13 对图像/掩膜空间尺寸不一致，其中 9 对正常、4 对良性。后 4 对为 `HESN_002.png` 至 `HESN_005.png`，由于无权威配准变换，予以排除。最终 `683 - 419 - 4 = 260`。患者标识取图像文件名前 4 个字符。患者可能有不同病灶标签，不能随意将所有图像标签改为患者最大标签。

BUS-BRA 从官方 `bus_data.csv` 的 `Pathology` 读取良恶性标签，从 `Case` 读取患者编号并加 `BUS-BRA:` 前缀；`ID` 对应图像，`Device` 对应设备信息。官方命名示例为 `bus_0001-l.png` 对应 `mask_0001-l.png`。代码要求 CSV 至少有 `ID`、`Case`、`Pathology` 列。

本包数据入口只包含正文使用的 BUS-BRA 与 BUS-UCLM。

## 2. 下载与目录

建议按下面的大小写布局保存文件，尤其在 Linux 中保持一致：

```text
data/
├── BUS-BRA/
│   └── BUSBRA/
│       ├── Images/
│       │   └── bus_0001-l.png
│       ├── Masks/
│       │   └── mask_0001-l.png
│       └── bus_data.csv
└── BUS-UCLM/
    ├── images/
    │   └── ALWI_000.png
    └── masks/
        └── ALWI_000.png
```

从 BUS-BRA 官方记录下载 `BUSBRA.zip`，解压到 `data/BUS-BRA/`，保留归档中的 `BUSBRA/` 层。历史下载记录给出的归档 MD5 为 `1f8b2be6476d58fc97bfb5e5a1ea9bab`；下载后核对官方记录和本地归档，不要将归档 MD5 当作单张图片校验值。

BUS-UCLM 可以从 V3 官方页面下载后整理到上述布局。仓库还提供下载脚本，依赖以下两份官方文件清单：

```text
data/bus_uclm_images_files.json
data/bus_uclm_masks_files.json
```

清单每条记录包含文件名、官方公开下载地址和 SHA256。Windows 下运行：

```powershell
python scripts/download_bus_uclm.py
```

脚本对已有文件先验 SHA256，再下载缺失或不符文件；成功结束应验证 1,366 个文件。脚本当前显式调用 `curl.exe`，Linux/macOS 不可直接照抄此命令完成下载，应从官方页面下载并按目录放置，或先将调用适配为当地的 `curl`。历史记录报告 1,366/1,366 文件通过 SHA256；这不代表新下载文件已经自动通过验证。

准备后运行基础设施阶段会生成 `manifest.csv`、`splits.csv`、处理后的掩膜和条件图像。具体阶段命令见仓库主复现 SOP。不要把旧电脑生成的绝对路径 CSV 直接覆盖到新电脑。

## 3. 模型身份、权重与校验

实现依据为 [src/features.py](../src/features.py)，模型表中的实际加载器优先于旧 README 或配置注释。

| 模型 | 实际加载身份 | 预期本地文件 | 特征维度 |
|---|---|---|---:|
| ResNet-50 | torchvision `IMAGENET1K_V2` | `models/torch/hub/checkpoints/resnet50-11ad3fa6.pth` | 2,048 |
| DINOv3 | timm `vit_large_patch16_dinov3.lvd1689m` | `models/dinov3_vitl16_lvd1689m.safetensors` | 1,024 |
| MedSAM2 | 官方 latest，`configs/sam2.1_hiera_t512.yaml` | `models/MedSAM2_latest.pt` | 256 |
| UltraSam | 官方 UltraSam ViT-B 权重，经 timm `samvit_base_patch16` 映射 | `models/UltraSam.pth` | 768 |

SHA256 历史锁定值：

```text
resnet50-11ad3fa6.pth
11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca

dinov3_vitl16_lvd1689m.safetensors
45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941

MedSAM2_latest.pt
c92743b99f00d078bf32a3afcc38aaa9faf1c1692dffe3eaa7a90938c1991060

UltraSam.pth
d7c223dd03f56b0b77cd246aa3edfae651e99f0ee2dde2e8c50eda7b21fa8a0c
```

代码还要求 DINOv3 为 1,212,347,640 字节、MedSAM2 为 156,040,129 字节、UltraSam 为 381,156,023 字节。MedSAM2/UltraSam 加载器同时验证长度与 SHA256；DINOv3 加载器只验证长度，应另外核 SHA256。

Windows 校验示例：

```powershell
Get-FileHash models/dinov3_vitl16_lvd1689m.safetensors -Algorithm SHA256
Get-FileHash models/MedSAM2_latest.pt -Algorithm SHA256
Get-FileHash models/UltraSam.pth -Algorithm SHA256
Get-FileHash models/torch/hub/checkpoints/resnet50-11ad3fa6.pth -Algorithm SHA256
```

下载途径：

- ResNet-50：首次运行由 torchvision 下载，项目设置 `TORCH_HOME=models/torch`。官方权重地址为 [resnet50-11ad3fa6.pth](https://download.pytorch.org/models/resnet50-11ad3fa6.pth)。
- DINOv3：取 [timm/vit_large_patch16_dinov3.lvd1689m](https://huggingface.co/timm/vit_large_patch16_dinov3.lvd1689m) 的 `model.safetensors`，重命名放到上表路径。原模型来源为 [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)。模型授权要求以官方页面为准。没有本地权重时加载器会请求 timm 的在线预训练权重；建议用本地文件加哈希锁定本次复现。
- MedSAM2：下载 [官方 MedSAM2_latest.pt](https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt)，另准备下节的官方源码。
- UltraSam：从 [CAMMA 官方权重地址](https://s3.unistra.fr/camma_public/github/ultrasam/UltraSam.pth) 下载；模型说明见 [CAMMA-public/UltraSam](https://github.com/CAMMA-public/UltraSam)。本项目通过 timm 加载映射后的编码器，不要求安装整个 MMDetection 训练工程；读取官方 checkpoint 元数据仍需 `mmengine`。

不要将 DINOv3 替换为 DINOv2。实际实现没有降级替代；旧配置中的 `fallback` 字段并不代表运行时会执行该替代。本包编码器入口只保留正文的四个模型。

## 4. 环境与 MedSAM2 源码

根目录 [requirements.txt](../requirements.txt) 是依赖范围清单，尚不是已锁定的历史训练环境。它的部分最低版本不足以单独保证四模型可用，例如 `timm>=0.9.0` 不保证存在 DINOv3 架构。MedSAM2 随附源码声明 Python >=3.10、torch >=2.5.1、torchvision >=0.20.1、numpy >=2.0.1，要求高于根清单的部分下限。

建立独立环境，按照 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装与本机驱动匹配的 torch/torchvision 后，再安装根依赖。下面是 Windows PowerShell 示例，Python 3.12 是新复现环境的选择示例，不是已确认的原训练版本：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# 此处先执行 PyTorch 官方选择器给出的、适合本机 CUDA/CPU 的安装命令。
python -m pip install -r requirements.txt
python -c "import timm; assert timm.is_model('vit_large_patch16_dinov3.lvd1689m'), 'Installed timm lacks DINOv3'; assert timm.is_model('samvit_base_patch16')"
```

如架构检查失败，需要安装包含该架构的 timm 版本并记录准确版本；不要跳过检查而换用其他模型。

MedSAM2 加载器要求 `models/MedSAM2-main/sam2/` 存在。可从官方仓库取得：

```powershell
git clone https://github.com/bowang-lab/MedSAM2.git models/MedSAM2-main
git -C models/MedSAM2-main rev-parse HEAD
```

本项目通过 `sys.path` 导入该目录中的 `sam2`，并调用 `build_sam2(..., apply_postprocessing=False)`。先确保 `hydra-core`、`iopath` 已由根依赖安装。若使用官方安装流程，需注意它可能更新 torch、torchvision 等依赖；应在独立环境进行并重做版本记录。源仓库和正式 `sam2` 包使用同名 Python 模块，不要在一次进程中先导入其他 SAM2 实现再加载 MedSAM2。

历史工作目录使用 `MedSAM2-main` 解压源码，未找到对应 Git commit；重新 clone 得到的是当前上游版本，不能声称与历史源码完全一致。建议公开发布时补充当时源码快照的校验和或精确 commit；未补齐前保留此缺口。

## 5. 运行前检查与记录

以下操作不会运行完整训练：

```powershell
python -m pip check
python -c "import sys,torch,torchvision,timm,sklearn,cv2,numpy; print(sys.version); print('torch',torch.__version__,'torchvision',torchvision.__version__,'timm',timm.__version__,'sklearn',sklearn.__version__,'opencv',cv2.__version__,'numpy',numpy.__version__); print('CUDA build',torch.version.cuda,'available',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
New-Item -ItemType Directory -Force logs | Out-Null
python -m pip freeze > logs/environment-freeze.txt
```

同时保存操作系统、GPU 型号和显存、驱动版本、PyTorch CUDA 构建版本、源码 commit、运行命令、开始/结束时间和以上权重哈希。GPU 显存/运行时长的最低需求尚未经过本整理包的新机器测试；不能从历史结果文件推算准确保证值。UltraSam 1024 像素输入在代码中将超过 2 的特征提取 batch size 限制为 2。

当前已找到可运行回归测试的 Python 3.13.1 环境，验证记录列出本次检查情况。其当前安装版本不自动等同于论文历史训练环境；仍需记录复现机器上的完整版本。

## 6. 可移植性

- 在目标机器重新生成 manifest、处理后掩膜、条件图与特征索引；这些运行产物会使用本机路径。
- 使用 `tools/restore_reference_splits.py --apply` 恢复历史患者外层折，路径通过数据集和图像名映射。
- 外层分类划分固定；2026–2030 用于下游重复拟合/内层 CV。
- 处理后的掩膜文件名包含路径哈希，跨机器名称可能不同，应按 manifest 对齐，而不是强行复用旧绝对路径。
- 软件、硬件、checkpoint 与预处理均需记录；新环境不保证逐位一致。
