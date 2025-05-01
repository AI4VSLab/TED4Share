#!/bin/bash
export PYTHONPATH="/data/michael/TED"

# ------------------------------------------- fl arguments -------------------------------------------
n_clients=2
num_rounds=2
script="run/train_fl.py"
key_metric="accuracy"
export_config="True"


# ------------------------------------------- training variables -------------------------------------------
ROOT_DIR="/data/michael/TED/data/TED/16_apr_2025_big_crop_10_cv/fold_1"  # Path to CSV files

CHECKPOINT_DIR="./checkpoints"
COMMENT="eye_cropped_less_aug_expanded_dataset" # Comment for the run
MODEL_ARCHITECTURE="microsoft/resnet-18" #"resnet50"
BATCH_SIZE=32
IMG_SIZE=512
EPOCHS=50
FEATURE_DIM=512 #2048
NUM_CLASSES=2
CKPT_PATH=""       # Provide path if fine-tuning, else leave empty
FINETUNE="False"   # Set to "True" if fine-tuning a checkpoint
loss="focal"
fast_dev_run="False"
lr=0.00003 #0.0001  3e-5: 0.00003


# Run training with or without checkpoint path
if [ -z "$CKPT_PATH" ]; then
    python flare_job.py \
        --root_dir "$ROOT_DIR" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --comment "$COMMENT" \
        --model_architecture "$MODEL_ARCHITECTURE" \
        --batch_size "$BATCH_SIZE" \
        --img_size "$IMG_SIZE" \
        --epochs "$EPOCHS" \
        --feature_dim "$FEATURE_DIM" \
        --num_classes "$NUM_CLASSES" \
        --loss_type "$loss" \
        --lr "$lr" \
        --n_clients "$n_clients" \
        --num_rounds "$num_rounds" \
        --script "$script" \
        --key_metric "$key_metric" \
        #--fast_dev_run "$fast_dev_run" 
else
    python flare_job.py \
        --root_dir "$ROOT_DIR" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --comment "$COMMENT" \
        --model_architecture "$MODEL_ARCHITECTURE" \
        --batch_size "$BATCH_SIZE" \
        --img_size "$IMG_SIZE" \
        --epochs "$EPOCHS" \
        --feature_dim "$FEATURE_DIM" \
        --num_classes "$NUM_CLASSES" \
        --ckpt_path "$CKPT_PATH" \
        --loss_type "$loss" \
        --lr "$lr" \
        --n_clients "$n_clients" \
        --num_rounds "$num_rounds" \
        --script "$script" \
        --key_metric "$key_metric" \
        #--fast_dev_run "$fast_dev_run"
fi 
