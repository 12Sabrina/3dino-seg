# 3dino-seg 

## 1. 关键文件与路径清单 (File Locations)
| 项目 | 绝对路径 |
| :--- | :--- |
| **项目根目录** | `/gpfs/share/home/2401111663/syy/3DINO-main-1` |
| **Python 环境** | `/gpfs/share/home/2401111663/anaconda3/envs/syy1/bin/python` |
| **预训练权重（外部）** | `/gpfs/share/home/2401111663/syy/3DINO-main/3dino_vit_weights.pth` |
| **JSON 生成脚本** | `prepare_brats_json.py` & `braTS_5folds/prepare_jsons_v3.py` |
| **数据 Datalist** | `/gpfs/share/home/2401111663/syy/braTS_5folds/` |
| **Slurm 脚本** | `sbatch/3dino_v3_ped.sbatch` & `3dino_v3_ssa.sbatch` |
| **训练日志** | `sbatch/output/v3_ped/` 或 `sbatch/output/v3_ssa/` |
| **模型保存** | `training_runs/v3_3dino/` |

## 权重文件说明

- **外部预训练权重**: 如果使用官方或外部预训练模型，请放置或链接到路径 `/gpfs/share/home/2401111663/syy/3DINO-main/3dino_vit_weights.pth`（示例路径）。
- **本地训练输出（不纳入仓库）**: 本项目的训练输出位于 `training_runs/`，例如 `training_runs/v3_3dino/.../best_model.pth`。这些为训练产生的模型文件，属于用户输出，不应直接提交到 Git 仓库（已在 `.gitignore` 中排除）。

> 若需在仓库中管理大型权重文件，请考虑使用 Git LFS 或把权重上传到共享存储并在 README 中给出下载链接。

## 2. 环境配置
- **Python**: 3.10+ (使用 conda 环境 `syy1`)
- **关键依赖**: `torch`, `monai`, `einops`, `timm`
- **环境变量**: 运行前需确保 `PYTHONPATH` 包含根目录。
  ```bash
  export PYTHONPATH=$PWD:$PYTHONPATH
  ```

## 3. 数据列表生成 (JSON Generation)
在微调训练前，需将分折原始 JSON 转换为 3DINO 专用格式。V3 实验主要使用项目特定的处理脚本。

### A. V3 数据处理脚本 (prepare_jsons_v3.py)
该脚本专门用于 V3 实验的数据准备，集成了自动切分和合成数据注入功能。
- **脚本位置**: `/gpfs/share/home/2401111663/syy/braTS_5folds/prepare_jsons_v3.py`
- **主要逻辑与模态筛选**:
  1. **T1c 模态提取**: 脚本会从原始数据的 `modalities` 中自动筛选 `T1-weighted Contrast Enhanced` (或 `T1-weighted Contrast CE`)。由于 V3 实验统一使用单模态，因此格式化后的 `image` 字段仅包含该 T1c 路径的列表。
  2. **训练/验证二次切分**: 
     - 脚本加载原始的 `train.json` 后，将其随机打乱。
     - 划分出固定的 **10 例真实数据** 作为验证集 (`validation`)。
     - 剩余的所有真实数据被划分为正式训练集 (`training`)。
  3. **测试集确定**: 原始的 `val.json` 中的所有样本被格式化为 `test` 作为最终评估的测试集。
  4. **合成数据注入 (Mixed 类型专用)**:
     - **匹配方案**: 脚本扫描合成目录（如 `.../inference/`），通过后缀匹配 `_generated.nii.gz` (合成的 T1c 图像) 和 `_mask.nii.gz` (对应的标签)。
     - **合并**: 仅将这些合成样本追加到上述剩下的training数据集中，生成 `mixed` 版本的 JSON。

### B. 生成json命令 (prepare_brats_json.py)
该脚本是通用的格式转换脚本，若需要手动转换或处理多模态，可使用此脚本。
- **命令示例**:
  ```bash
  python prepare_brats_json.py \
      --train-json /path/to/mixed_train.json \
      --val-json /path/to/real_val.json \
      --output-json /path/to/ped_mixed_fold1_v3.json
  ```

## 4. 实验流程
1. **预处理 JSON**: 运行 `prepare_brats_json.py` 为（Real/Mixed）生成对应的 3DINO datalist。
3. **配置 sbatch**: 修改 `sbatch/3dino_v3_ped.sbatch` 中的 `JSON_REAL` 和 `JSON_MIXED` 路径，确保指向1 中生成的 JSON。其中权重文件在 `/gpfs/share/home/2401111663/syy/3DINO-main/3dino_vit_weights.pth`
4. **提交任务**: 一个sbatch文件中有两个实验，一个实验是真实的数据，一个是加入合成数据的混合数据，`/gpfs/share/home/2401111663/syy/3DINO-main-1/sbatch/3dino_ped_v3_f2to5.sbatch`
5. **结果**: 通过 `sbatch/output/` 下的日志查看 Dice 和 HD95 变化。

## 5. 数据集结构说明
数据列表存储在 `/gpfs/share/home/2401111663/syy/braTS_5folds/` 下：
- **Pediatric (PED)**: 5 折交叉验证。
*注：V3 实验中，JSON 路径已在 sbatch 脚本中针对单通道数据进行了硬编码或条件判断，请确保文件路径存在。*


## 6. 训练参数说明
在 `segmentation3d.py` 中，关键参数包括：
- `--datalist-path`: 指定任务的 JSON 路径。
- `--pretrained-weights`: 3DINO 预训练模型路径。
- `--epochs`: 默认 150 轮。
- `--batch-size`: 默认 2。
- `--image-size`: 96 x 96 x 96 (保持 DINO 特征块对齐)。



