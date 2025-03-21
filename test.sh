#!/bin/bash

# Variables
ROOT_DIR=""  # Path to CSV files
BATCH_SIZE=32
IMG_SIZE=224
CKPT_PATH=""       



python test.py \
    --root_dir "$ROOT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --img_size "$IMG_SIZE" \
    --ckpt_path "$CKPT_PATH" \
