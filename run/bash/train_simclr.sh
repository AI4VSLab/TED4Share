#!/bin/bash
export PYTHONPATH="/data/michael/TED"

# Variables
# ROOT_DIR="/data/michael/ted_rashmi/TED/data/TED/19_mar_2025/fold_2"  # Path to CSV files
# ROOT_DIR="/home/CenteredData/CelebA/by_us/entire_dataset"
ROOT_DIR="/data/michael/TED/data/ffhq_cropped_aligned/entire_dataset_train"

CHECKPOINT_DIR="./checkpoints"
COMMENT="simclr_ffhq_stronger_aug" 
MODEL_ARCHITECTURE='microsoft/resnet-50'   #"resnet50"
PRETRAINED="True"
BATCH_SIZE=32
IMG_SIZE=224
EPOCHS=20
FEATURE_DIM=2048
NUM_CLASSES=2
CKPT_PATH=""       # Provide path if fine-tuning, else leave empty
FINETUNE="False"   # Set to "True" if fine-tuning a checkpoint
loss="simclr"

# Run training with or without checkpoint path
if [ -z "$CKPT_PATH" ]; then
    python ../train.py \
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
        --loss_type "$loss"
else
    python ../train.py \
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
        --loss_type "$loss" \
        --finetune "$FINETUNE"
fi