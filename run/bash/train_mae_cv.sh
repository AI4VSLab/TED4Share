#!/bin/bash
# MAE pretraining but for all cross validation folds
export PYTHONPATH="/data/michael/public_code/TED4Share"

ROOT_DIR_BASE=""  

CHECKPOINT_DIR="../checkpoints"
COMMENT="24_5_2025_1" 
MODEL_ARCHITECTURE='facebook/vit-mae-base'   #"resnet50"
BATCH_SIZE=64
IMG_SIZE=224
EPOCHS=50
FEATURE_DIM=0 # keep this as 0 for MAE
NUM_CLASSES=2
CKPT_PATH=""       
loss="mae"
lr=2e-4


for fold in {1..10} ; do  #
    echo "Running training for fold ${fold}..."
    ROOT_DIR="${ROOT_DIR_BASE}/fold_${fold}"

    # Run training with or without checkpoint path
    if [ -z "$CKPT_PATH" ]; then
        python ../train.py \
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
            --no-finetune  \
            --loss_type "$loss" \
            --scheduler "step" \
            --lr "$lr" 
    else
        python ../train.py \
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
            --no-finetune \
            --scheduler "step" \
            --lr "$lr" 
    fi
      
    echo "Completed training for fold ${fold}"
    echo "----------------------------------------------------------------"
done
echo "Cross-validation pretraining complete for all folds"
