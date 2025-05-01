# TED Image Classification with PyTorch Lightning

## Overview
This project trains a deep learning model using **ResNet50** for classifying TED_1 and CONT_ images. The model is fine-tuned using **PyTorch Lightning** and evaluated on precision, recall, F1-score, sensitivity, and specificity.

---

## Install Dependencies

conda env create -f env.yaml

## to train locally run the following command

`bash run/bash/train.sh` 
or training with all folds
`bash run/bash/train_cv_all_folds.sh` 


## Preprocessing
`preprocess_images.ipynb` contains code to use mediapipe to extract eye area of the face and saves them to a path. For images that can't be automatically cropped, `preprocess_images.ipynb` will have a list that you can save to csv for `crop_maual.ipynb`. `crop_maual.ipynb` allows you to crop directly in the notebook. 



