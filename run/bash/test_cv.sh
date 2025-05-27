#!/bin/bash
export PYTHONPATH="/data/michael/public_code/TED4Share"


# ------ change these variables ------
ckpt_fl_final="" 
ROOT_DIR_BASE=""  
CKPT_PATH_BASE="" # include tb_logs 
MODEL_ARCHITECTURE='microsoft/resnet-50' 
# ------ dont need to change rest ------


COMMENT="23_5_2025_2_finetune" # Comment for the run
BATCH_SIZE=64
IMG_SIZE=224
FEATURE_DIM=0 #2048
NUM_CLASSES=2
loss="ce" #ce focal
lr=0.00002 #0.0001  3e-5: 0.00003


for fold  in {1..10} ; do # 
    echo "Running testing for fold ${fold}..."
    ROOT_DIR="${ROOT_DIR_BASE}/fold_${fold}"
    version=$((fold -1)) 
    CKPT_PATH=$(find ${CKPT_PATH_BASE}/version_${version}/checkpoints -name '*.ckpt')
    echo "Ckpt for Version $version: $CKPT_PATH"
    echo "root_dir: $ROOT_DIR"
    
    python ../test.py \
        --root_dir "$ROOT_DIR" \
        --comment "$COMMENT" \
        --model_architecture "$MODEL_ARCHITECTURE" \
        --pretrained \
        --batch_size "$BATCH_SIZE" \
        --img_size "$IMG_SIZE" \
        --feature_dim "$FEATURE_DIM" \
        --num_classes "$NUM_CLASSES" \
        --loss_type "$loss" \
        --lr "$lr" \
        --no-finetune  \
        --for_cls \
        --ckpt_path "$CKPT_PATH" \
        --ckpt_fl_final  "$ckpt_fl_final"
    
done