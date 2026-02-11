#!/bin/bash
#SBATCH --job-name=3dino-ssa-v3
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=72:00:00
#SBATCH --output=/gpfs/share/home/2401111663/syy/3DINO-main-1/sbatch/output/v3_ssa/3dino-v3-%j.out
#SBATCH --error=/gpfs/share/home/2401111663/syy/3DINO-main-1/sbatch/output/v3_ssa/3dino-v3-%j.err

set -euo pipefail

BASE_DIR=/gpfs/share/home/2401111663/syy/3DINO-main-1
PYTHON_BIN=/gpfs/share/home/2401111663/anaconda3/envs/syy1/bin/python
PRETRAINED=/gpfs/share/home/2401111663/syy/3DINO-main/3dino_vit_weights.pth

mkdir -p /gpfs/share/home/2401111663/syy/3DINO-main-1/sbatch/output/v3_ssa
export PYTHONPATH=$BASE_DIR:${PYTHONPATH:-}

for FOLD in {1..4}; do
  echo "========== Starting SSA Fold ${FOLD} =========="
  
  JSON_REAL=/gpfs/share/home/2401111663/syy/braTS_5folds/ssa_task/5_fold/fold_${FOLD}/ssa_real_fold${FOLD}_v3.json
  JSON_MIXED=/gpfs/share/home/2401111663/syy/braTS_5folds/ssa_task/5_fold/fold_${FOLD}/ssa_mixed_fold${FOLD}_v3.json
  
  OUT_REAL=$BASE_DIR/training_runs/v3_3dino/ssa_real_fold${FOLD}
  mkdir -p $OUT_REAL
  
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
      --num-workers 8 \
      --output-dir $OUT_REAL > /gpfs/share/home/2401111663/syy/3DINO-main-1/sbatch/output/v3_ssa/real_fold${FOLD}.log 2>&1 &

  OUT_MIXED=$BASE_DIR/training_runs/v3_3dino/ssa_mixed_fold${FOLD}
  mkdir -p $OUT_MIXED

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
      --num-workers 8 \
      --output-dir $OUT_MIXED > /gpfs/share/home/2401111663/syy/3DINO-main-1/sbatch/output/v3_ssa/mixed_fold${FOLD}.log 2>&1

  wait
  echo "========== Finished SSA Fold ${FOLD} =========="
done
