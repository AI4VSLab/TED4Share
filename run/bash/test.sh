#!/bin/bash

# Variables
ROOT_DIR="/data/michael/ted_rashmi/TED/data/TED/12_mar_2025"  # Path to CSV files
BATCH_SIZE=32
IMG_SIZE=224
CKPT_PATH="/data/michael/ted_rashmi/TED/checkpoints/train_ted/epoch=5-step=36.ckpt"       
fast_dev_run="False"



python test.py \
    --root_dir "$ROOT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --img_size "$IMG_SIZE" \
    --ckpt_path "$CKPT_PATH" \
    #--fast_dev_run "$fast_dev_run" 
