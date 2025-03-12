# TED Image Classification with PyTorch Lightning

## Overview
This project trains a deep learning model using **ResNet50** for classifying TED_1 and CONT_ images. The model is fine-tuned using **PyTorch Lightning** and evaluated on precision, recall, F1-score, sensitivity, and specificity.

---

## Install Dependencies

pip install -r requirements.txt

## to train run the following command

python train.py \
    --root_dir "your path to the root directory containing the images" \
    --comment "train_ted" \
    --model_architecture "resnet50" \
    --pretrained True \
    --batch_size 32 \
    --img_size 224 \
    --epochs 20 \
    --feature_dim 2048 \
    --checkpoint_dir "your path to the checkpoints directory " \
    --num_classes 2



## alternatively to train in the terminal
1. Give Execute Permission (Only If Needed)
If train.sh does not have execute permissions

chmod +x train.sh

2.Run the train.sh Script
./train.sh

Please change all the paths accordingly in your .sh files.
