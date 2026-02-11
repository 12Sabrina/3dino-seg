#!/bin/bash
#SBATCH --job-name=3d-ped-v3-f2
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=72:00:00
#SBATCH --output=/gpfs/share/home/2401111663/syy/3DINO-main-1/sbatch/output/v3_ped/3dino-v3-f2to5-%j.out
#SBATCH --error=/gpfs/share/home/2401111663/syy/3DINO-main-1/sbatch/output/v3_ped/3dino-v3-f2to5-%j.err

set -euo pipefail

BASE_DIR=/gpfs/share/home/2401111663/syy/3DINO-main-1
PYTHON_BIN=/gpfs/share/home/2401111663/anaconda3/envs/syy1/bin/python
PRETRAINED=/gpfs/share/home/2401111663/syy/3DINO-main/3dino_vit_weights.pth

mkdir -p $BASE_DIR/sbatch/output/v3_ped
export PYTHONPATH=$BASE_DIR:${PYTHONPATH:-}


for FOLD in {2..5}; do
  echo "========== Starting PED Fold ${FOLD} (BrainMVP JSON) =========="
  
  # 直接引用 BrainMVP 的 V3 JSON 文件路径
  JSON_REAL=/gpfs/share/home/2401111663/syy/braTS_5folds/pediatric_task/5_fold/fold_${FOLD}/ped_real_fold${FOLD}_v3.json
  JSON_MIXED=/gpfs/share/home/2401111663/syy/braTS_5folds/pediatric_task/5_fold/fold_${FOLD}/ped_mixed_fold${FOLD}_v3.json
  
  OUT_REAL=$BASE_DIR/training_runs/v3_3dino/ped_real_fold${FOLD}
  OUT_MIXED=$BASE_DIR/training_runs/v3_3dino/ped_mixed_fold${FOLD}
  mkdir -p $OUT_REAL $OUT_MIXED
  
  echo "Running Real Fold ${FOLD} on GPU 0..."
  CUDA_VISIBLE_DEVICES=0 stdbuf -oL $PYTHON_BIN $BASE_DIR/dinov2/eval/segmentation3d.py \
      --dataset-name BraTS \
      --task-type segmentation \
      --datalist-path $JSON_REAL \
      --pretrained-weights $PRETRAINED \
      --segmentation-head ViTAdapterUNETR \
      --epochs 150 \
      --batch-size 2 \
      --learning-rate 0.0003 \
      --image-size 96 \
      --num-workers 6 \
      --output-dir $OUT_REAL > /gpfs/share/home/2401111663/syy/3DINO-main-1/sbatch/output/v3_ped/real_fold${FOLD}.log 2>&1 &

  echo "Running Mixed Fold ${FOLD} on GPU 1..."
  CUDA_VISIBLE_DEVICES=1 stdbuf -oL $PYTHON_BIN $BASE_DIR/dinov2/eval/segmentation3d.py \
      --dataset-name BraTS \
      --task-type segmentation \
      --datalist-path $JSON_MIXED \
      --pretrained-weights $PRETRAINED \
      --segmentation-head ViTAdapterUNETR \
      --epochs 150 \
      --batch-size 2 \
      --learning-rate 0.0003 \
      --image-size 96 \
      --num-workers 6 \
      --output-dir $OUT_MIXED > /gpfs/share/home/2401111663/syy/3DINO-main-1/sbatch/output/v3_ped/mixed_fold${FOLD}.log 2>&1

  wait
  echo "========== Finished PED Fold ${FOLD} =========="
done
