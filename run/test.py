# testing final fl model 
import os
import argparse
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from PIL import Image
import pandas as pd

from dataset.datamodule import CustomDatamodule
from models.classification import ClassificationNet
from util.get_models import get_baseline_model
from util.seed import set_seed
from util.apply_transformation import get_transforms
from util.util_test import evaluate_testset

from run.parser import define_parser



def main():
    set_seed(42) # 42
    parser, args = define_parser()


    n_img_views = 1
    # ---------------------------------------------------------------------------------------------------
    # 1) Define transforms
    # ---------------------------------------------------------------------------------------------------
    train_transform, test_transform = get_transforms(args)

    # ---------------------------------------------------------------------------------------------------
    # 2) Create DataModule
    # ---------------------------------------------------------------------------------------------------
    classes = {"TED_1": 1, "CONT_": 0}
    data_module = CustomDatamodule(
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        train_transform=train_transform,
        test_transform=test_transform,
        classes=classes,
        n_img_views=n_img_views
    )
    data_module.prepare_data()
    data_module.setup("train")
    data_module.setup("val")
    data_module.setup("test")

    # ---------------------------------------------------------------------------------------------------
    # 3) Setup PL Trainer
    # ---------------------------------------------------------------------------------------------------
    # get path up to "version_i"
    save_path = os.path.dirname(os.path.dirname(args.ckpt_path))  #os.path.join(args.checkpoint_dir, args.comment)
    os.makedirs(save_path, exist_ok=True)

    callbacks = []
    if args.loss_type != "simclr" and args.loss_type != 'mae':  callbacks.append(EarlyStopping(monitor="val_loss", patience=10, mode="min"))
    if args.loss_type != 'simclr' and args.loss_type != 'mae': callbacks.append( ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")) # dirpath=save_path
    else: callbacks.append( ModelCheckpoint(save_top_k=1, monitor="train_loss", mode="min"))
    callbacks.append( LearningRateMonitor(log_momentum = True, log_weight_decay = True))

    trainer = pl.Trainer(
        accelerator= 'gpu',
        devices=[0],
        max_epochs=args.epochs,
        logger=TensorBoardLogger(save_path, name="tb_logs"),
        callbacks=callbacks,
        log_every_n_steps=1,
        default_root_dir=save_path,
        #fast_dev_run= True,
    )

    # ---------------------------------------------------------------------------------------------------
    # 4) Build Model
    # ---------------------------------------------------------------------------------------------------
    model = ClassificationNet(
        feature_dim=args.feature_dim,
        classes=classes,
        lr=args.lr,              # can tune
        wd = 1e-6,
        loss_type=args.loss_type,     # or "ce"
        model_architecture = args.model_architecture,
        pretrained=args.pretrained,
        for_cls=args.for_cls,
    )



    if args.ckpt_path:
        print(f"Loading checkpoint from {args.ckpt_path}...")
        state_dict = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    elif args.ckpt_fl_final:
        ckpt = torch.load(args.ckpt_fl_final)
        model.load_state_dict(ckpt['model'])

    

    # ---------------------------------------------------------------------------------------------------
    # 6) Train and Evaluate, depeonds on args.fl
    # ---------------------------------------------------------------------------------------------------
    # log_dir different now, so make sure we save to same folder
    model.log_dir = save_path
    
    trainer.test(model, datamodule=data_module)

    '''
    # ---------------------------------------------------------------------------------------------------
    # 7) Single-Image Inference (Optional)
    # ---------------------------------------------------------------------------------------------------
    if args.inference_image is not None:
        print(f"\nRunning inference on single image: {args.inference_image}")
        prediction_idx = predict_single_image(
            model, 
            args.inference_image, 
            transform=test_transform,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        # Reverse-lookup classes
        # classes = {"TED_1": 1, "CONT_": 0}
        # so "TED_1" has index 1, "CONT_" has index 0
        # we can invert that:
        inv_map = {v: k for k, v in classes.items()}
        predicted_class = inv_map[prediction_idx]
        print(f"Predicted Class for {args.inference_image} is: {predicted_class}")
    '''


if __name__ == "__main__":
    main()