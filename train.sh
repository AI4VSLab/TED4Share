#!/bin/bash

# Variables
ROOT_DIR="/data/michael/ted_rashmi/TED/data/TED/12_mar_2025"  # Path to CSV files
CHECKPOINT_DIR="./checkpoints"
COMMENT="train_ted"
MODEL_ARCHITECTURE="resnet50"
PRETRAINED="True"
BATCH_SIZE=32
IMG_SIZE=224
EPOCHS=20 
FEATURE_DIM=2048
NUM_CLASSES=2
CKPT_PATH=""       # Provide path if fine-tuning, else leave empty
FINETUNE="False"   # Set to "True" if fine-tuning a checkpoint
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
