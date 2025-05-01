#!/bin/bash

# Variables
ROOT_DIR="/data/michael/ted_rashmi/TED/data/TED/19_mar_2025/fold_2"  # Path to CSV files
CHECKPOINT_DIR="./checkpoints"
COMMENT="whitespace_cropped_simclr_finetune" 
MODEL_ARCHITECTURE="resnet50"
PRETRAINED="True"
BATCH_SIZE=32
IMG_SIZE=224
EPOCHS=20
FEATURE_DIM=2048
NUM_CLASSES=2
CKPT_PATH="/data/michael/ted_rashmi/TED/checkpoints/whitespace_cropped_simclr/epoch=18-step=114.ckpt"       # Provide path if fine-tuning, else leave empty
FINETUNE="True"   # Set to "True" if fine-tuning a checkpoint
loss="focal"
fast_dev_run="False"



# Run training with or without checkpoint path
if [ -z "$CKPT_PATH" ]; then
    python train.py \
        --root_dir "$ROOT_DIR" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --comment "$COMMENT" \
        --model_architecture "$MODEL_ARCHITECTURE" \
        --pretrained "$PRETRAINED" \
        --batch_size "$BATCH_SIZE" \
        --img_size "$IMG_SIZE" \
        --epochs "$EPOCHS" \
        --feature_dim "$FEATURE_DIM" \
        --num_classes "$NUM_CLASSES" \
        --finetune "$FINETUNE" \
        --loss_type "$loss" \
        #--fast_dev_run "$fast_dev_run" 
else
    python train.py \
        --root_dir "$ROOT_DIR" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --comment "$COMMENT" \
        --model_architecture "$MODEL_ARCHITECTURE" \
        --pretrained "$PRETRAINED" \
        --batch_size "$BATCH_SIZE" \
        --img_size "$IMG_SIZE" \
        --epochs "$EPOCHS" \
        --feature_dim "$FEATURE_DIM" \
        --num_classes "$NUM_CLASSES" \
        --ckpt_path "$CKPT_PATH" \
        --finetune "$FINETUNE" \
        #--fast_dev_run "$fast_dev_run" 
fi
