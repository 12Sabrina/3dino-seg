#!/bin/bash
#SBATCH --job-name=3dino-ssa
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=72:00:00
#SBATCH --output=/gpfs/share/home/2401111663/syy/3DINO-main-1/sbatch/output/ssa/3dino-ssa-%j.out
#SBATCH --error=/gpfs/share/home/2401111663/syy/3DINO-main-1/sbatch/error/ssa/3dino-ssa-%j.err

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Configuration logic similar to BrainMVP
BASE_DIR=/gpfs/share/home/2401111663/syy/3DINO-main-1
DATA_ROOT=/gpfs/share/home/2401111663/syy/braTS_5folds/ssa_task/5_fold
export PYTHONPATH=${PYTHONPATH:-}:$BASE_DIR

# Find Python binary
PYTHON_BIN=${PYTHON_BIN:-/gpfs/share/home/2401111663/anaconda3/envs/syy1/bin/python}
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN=/gpfs/share/home/2401111663/anaconda3/envs/3dino/bin/python
fi
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN=/gpfs/share/home/2401111663/anaconda3/bin/python
fi

echo "Using python: $PYTHON_BIN"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PRETRAINED=/gpfs/share/home/2401111663/syy/3DINO-main/3dino_vit_weights.pth

for FOLD in {1..1}; do
  FOLD_DIR=$DATA_ROOT/fold_${FOLD}
  JSON_OUT=$DATA_ROOT/dataset_ssa_fold_${FOLD}.json
  OUTPUT_DIR="$BASE_DIR/training_runs/ssa_fold${FOLD}"
  mkdir -p ${OUTPUT_DIR}

  echo "Starting SSA fold ${FOLD}"

  # Prepare JSON
  $PYTHON_BIN $BASE_DIR/prepare_brats_json.py \
      --train-json $FOLD_DIR/train.json \
      --val-json $FOLD_DIR/val.json \
      --output-json $JSON_OUT

  cd $BASE_DIR

  $PYTHON_BIN dinov2/eval/segmentation3d.py \
      --dataset-name BraTS \
      --task-type segmentation \
      --datalist-path $JSON_OUT \
      --pretrained-weights $PRETRAINED \
      --segmentation-head ViTAdapterUNETR \
      --epochs 200 \
      --epoch-length 24 \
      --batch-size 1 \
      --learning-rate 0.0001 \
      --image-size 64 \
      --num-workers 6 \
      --output-dir ${OUTPUT_DIR}
done
