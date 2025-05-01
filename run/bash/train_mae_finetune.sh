#!/bin/bash
export PYTHONPATH="/data/michael/TED"

# Variables
# ROOT_DIR="/home/CenteredData/CelebA/by_us/entire_dataset"
ROOT_DIR="/data/michael/TED/data/TED/14_apr_2025_cropped_5_cv/fold_1"  # Path to CSV files

CHECKPOINT_DIR="./checkpoints"
COMMENT="mae_ffhq_eye_area_entire4train_then_292_class" 
MODEL_ARCHITECTURE='facebook/vit-mae-base'   #"resnet50"
PRETRAINED="True"
BATCH_SIZE=64
IMG_SIZE=224
EPOCHS=20
FEATURE_DIM=0
NUM_CLASSES=2
CKPT_PATH="/data/michael/TED/run/checkpoints/mae_ffhq_eye_area_entire4train/tb_logs/version_1/checkpoints/epoch=9-step=1370.ckpt"       # Provide path if fine-tuning, else leave empty
FINETUNE="False"   # Set to "True" if fine-tuning a checkpoint
loss="ce"
lr=1e-2 


# make sure to set this
USE_FOR_CLASSIFICATION="True" # Set to "True" if using for classification


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
        --loss_type "$loss" \
        --lr "$lr" 
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
        --use_for_classification "$USE_FOR_CLASSIFICATION" \
        --loss_type "$loss" \
        --finetune "$FINETUNE" \
        --lr "$lr" 
fi