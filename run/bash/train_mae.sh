#!/bin/bash
export PYTHONPATH="/data/michael/TED"

# ROOT_DIR="/home/CenteredData/CelebA/by_us/entire_dataset"
ROOT_DIR="/data/michael/TED/data/TED/11_apr_2025_diff_seed/fold_1"  # Path to CSV files
ROOT_DIR="/data/michael/TED/data/ffhq_cropped_aligned/entire_dataset_train"
ROOT_DIR="/data/michael/TED/data/ffhq_cropped_aligned/cropped_eye_area/entire_dataset4train"

CHECKPOINT_DIR="./checkpoints"
COMMENT="mae_ffhq_eye_area_entire4train" 
MODEL_ARCHITECTURE='facebook/vit-mae-base'   #"resnet50"
PRETRAINED="True"
BATCH_SIZE=512
IMG_SIZE=224
EPOCHS=10
FEATURE_DIM=0
NUM_CLASSES=2
CKPT_PATH=""       # Provide path if fine-tuning, else leave empty
FINETUNE="False"   # Set to "True" if fine-tuning a checkpoint
loss="mae"
lr=2e-4 

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
        --loss_type "$loss" \
        --finetune "$FINETUNE" \
        --lr "$lr" 
fi