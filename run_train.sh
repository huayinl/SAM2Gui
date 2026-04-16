#!/bin/bash

# Example usage:
# bash run_train.sh /path/to/checkpoint.pt /path/to/train_data /path/to/val_data /path/to/output_dir mask_decoder

CHECKPOINT_PATH="$1"
TRAIN_DATA_DIR="$2"
VAL_DATA_DIR="$3"
OUTPUT_DIR="$4"
FINETUNE_MODE="${5:-mask_decoder}"

python train.py \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --train_data_dir "$TRAIN_DATA_DIR" \
  --val_data_dir "$VAL_DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --finetune_mode "$FINETUNE_MODE"