#!/bin/bash
#SBATCH --job-name=3dino-glioma
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=72:00:00
#SBATCH --output=/gpfs/share/home/2401111663/syy/3DINO-main-1/sbatch/output/classification/glioma/3dino-glioma-%j.out
#SBATCH --error=/gpfs/share/home/2401111663/syy/3DINO-main-1/sbatch/error/classification/glioma/3dino-glioma-%j.err

set -euo pipefail

# Configuration logic similar to BrainMVP
BASE_DIR=/gpfs/share/home/2401111663/syy/3DINO-main-1
DATA_ROOT=/gpfs/share/home/2401111663/labShare/2301210592/json/braTS/glioma_task/5_fold
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

PRETRAINED=/gpfs/share/home/2401111663/syy/3DINO-main/3dino_vit_weights.pth

for FOLD in {1..5}; do
  FOLD_DIR=$DATA_ROOT/fold_${FOLD}
  OUTPUT_DIR="$BASE_DIR/training_runs/classification/glioma/fold_${FOLD}"
  mkdir -p ${OUTPUT_DIR}

  echo "Starting Glioma classification (segmentation) fold ${FOLD}"

  JSON_OUT=${OUTPUT_DIR}/datalist.json
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
      --batch-size 2 \
      --learning-rate 0.0001 \
      --image-size 128 \
      --num-workers 8 \
      --output-dir ${OUTPUT_DIR}
done
