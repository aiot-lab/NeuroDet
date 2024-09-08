#!/bin/bash

# Training script for model
DATASET_PATHS=(
    "path/to/dataset/folder"
)

CALIBRATION_PATHS=(
    "/path/to/calibration/folder"
)

python train_model.py \
  --dataset_paths "${DATASET_PATHS[@]}" \
  --calibration_paths "${CALIBRATION_PATHS[@]}" \
  --num_layers 2 \
  --hidden_dim 256 \
  --order 256 \
  --dt_min 1e-3 \
  --dt_max 8e-5 \
  --channels 1 \
  --dropout 0.0 \
  --learning_rate 1e-2 \
  --batch_size 4 \
  --num_workers 4 \
  --total_steps 10000 \
  --weight_decay 1e-1 \
  --optimizer AdamW \
  --step_size 300 \
  --gamma 0.5 \
  --save_dir "./checkpoints" \
  --visualization_stride 100 \
  --gpus 1 \
  --log_dir "./logs" \
  --loss_type "bce"
