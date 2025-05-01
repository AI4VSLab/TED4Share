#!/bin/bash
export PYTHONPATH="/data/michael/TED"

# Base configuration
CHECKPOINT_DIR="../checkpoints"
COMMENT="22_apr_25_3" # Comment for the run
MODEL_ARCHITECTURE="microsoft/resnet-18" 
PRETRAINED="True"
BATCH_SIZE=32
IMG_SIZE=512
EPOCHS=50
FEATURE_DIM=512 #512 2048
NUM_CLASSES=2
CKPT_PATH=""       # Provide path if fine-tuning, else leave empty
loss="focal"
lr=0.00002 #0.0001  3e-5: 0.00003

# Loop through folds 1 to 5
for fold in {1..10}; do
    echo "Running training for fold ${fold}..."
    
    # Set fold-specific directory
    ROOT_DIR="/data/michael/TED/data/TED/16_apr_2025_big_crop_10_cv/fold_${fold}"
    
    # Run training with or without checkpoint path
    if [ -z "$CKPT_PATH" ]; then
        python ../train_fl.py \
            --root_dir "$ROOT_DIR" \
            --checkpoint_dir "$CHECKPOINT_DIR" \
            --comment "$COMMENT" \
            --model_architecture "$MODEL_ARCHITECTURE" \
            --pretrained  \
            --batch_size "$BATCH_SIZE" \
            --img_size "$IMG_SIZE" \
            --epochs "$EPOCHS" \
            --feature_dim "$FEATURE_DIM" \
            --num_classes "$NUM_CLASSES" \
            --loss_type "$loss" \
            --lr "$lr" \
            --no-finetune 
    else
        python ../train_fl.py \
            --root_dir "$ROOT_DIR" \
            --checkpoint_dir "$CHECKPOINT_DIR" \
            --comment "$COMMENT" \
            --model_architecture "$MODEL_ARCHITECTURE" \
            --pretrained  \
            --batch_size "$BATCH_SIZE" \
            --img_size "$IMG_SIZE" \
            --epochs "$EPOCHS" \
            --feature_dim "$FEATURE_DIM" \
            --num_classes "$NUM_CLASSES" \
            --ckpt_path "$CKPT_PATH" \
            --loss_type "$loss" \
            --lr "$lr" \
            --no-finetune 
    fi
    
    echo "Completed training for fold ${fold}"
    echo "--------------------------------"
done

echo "Cross-validation training complete for all folds"
