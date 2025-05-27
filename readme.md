# TED Image Classification with PyTorch Lightning

## Overview
This project trains a deep learning model using **ResNet50** for classifying TED_1 and CONT_ images. The model is fine-tuned using **PyTorch Lightning** and evaluated on precision, recall, F1-score, sensitivity, and specificity.

---

## Install Dependencies

conda env create -f env.yaml

## to train locally run the following command

`cv` for cross validation

supervised training: `bash run/bash/train_cv.sh` 
MAE training: `bash run/bash/train_mae_cv.sh`
Finetune MAE pretrained:  `bash run/bash/train_mae_finetune_cv.sh`

make sure to change comment for each experiment and it will create a new folder under `../checkpoints`

## Preprocessing
`preprocess_images.ipynb` contains code to use mediapipe to extract eye area of the face and saves them to a path. For images that can't be automatically cropped, `preprocess_images.ipynb` will have a list that you can save to csv for `crop_maual.ipynb`. `crop_maual.ipynb` allows you to crop directly in the notebook. 


## Testing
`bash plot_roc.sh`: creates ROC curve with base path to checkpoint folder


