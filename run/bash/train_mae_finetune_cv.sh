#!/bin/bash
export PYTHONPATH="/data/michael/public_code/TED4Share"

ROOT_DIR_BASE=""  
CKPT_PATH_BASE="" # include tb_logs

CHECKPOINT_DIR="../checkpoints"
COMMENT="24_5_2025_1_finetune_5" 
MODEL_ARCHITECTURE='facebook/vit-mae-base'   #"resnet50"
BATCH_SIZE=64
IMG_SIZE=224
EPOCHS=50
FEATURE_DIM=0
NUM_CLASSES=2
loss="ce"
lr=3e-5 #1e-3

for fold  in {1..10} ; do # 
    echo "Running training for fold ${fold}..."
    ROOT_DIR="${ROOT_DIR_BASE}/fold_${fold}"
    version=$((fold -1)) # fold -1
    ckpt_path=$(find ${CKPT_PATH_BASE}/version_${version}/checkpoints -name '*.ckpt')
    echo "Ckpt for Version $fold: $ckpt_path"

    # Run training with or without checkpoint path
    if [ -z "$ckpt_path" ]; then
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
            --loss_type "$loss" \
            --no-finetune  \
            --for_cls \
            --lr "$lr" \
            --wd 5e-2
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
            --ckpt_path "$ckpt_path" \
            --loss_type "$loss" \
            --no-finetune  \
            --for_cls  \
            --lr "$lr" \
            --wd 5e-2
    fi
    echo "Completed training for fold ${fold}"
    echo "----------------------------------------------------------------"
done
echo "Cross-validation training complete for all folds"


'''
for i in {1..10}; do
  version=$((i - 1))
  ckpt_path=$(find ${CKPT_PATH_BASE}/version_${version}/checkpoints -name '*.ckpt')
  echo "Version $i: $ckpt_path"
done
'''