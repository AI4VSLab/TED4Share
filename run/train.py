# --------------------------------------------------------------------------------------------------------------------------
# Modified from: https://github.com/sinagh72/TED by Sina Gholami
# view modifications to Flare here: https://github.com/NVIDIA/NVFlare/blob/main/examples/hello-world/ml-to-fl/pt/src/cifar10_lightning_fl.py
# --------------------------------------------------------------------------------------------------------------------------
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

# -------------------------------------------- nvflare import --------------------------------------------
import nvflare.client.lightning as flare

from run.parser import define_parser
import random


def predict_single_image(model, image_path, transform, device='cuda'):
    """
    Inference on a single image. Returns the predicted label.
    """
    model.to(device)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img)
        pred = torch.argmax(logits, dim=1).item()

    return pred

def fit(trainer, model, data_module):
    trainer.fit(model, data_module)
    # The best checkpoint is stored at:
    best_ckpt = trainer.checkpoint_callback.best_model_path
    print(f"\nBest checkpoint is at: {best_ckpt}")
    return best_ckpt

def evaluate(args, trainer, model, data_module, best_ckpt, save_path):
    if os.path.exists(best_ckpt) and args.loss_type != "simclr" and args.loss_type != "mae":
        # load best checkpoint weights
        best_state_dict = torch.load(best_ckpt, map_location="cpu")["state_dict"]
        model.load_state_dict(best_state_dict)
        print("\nEvaluating on test set with the best checkpoint...")
        evaluate_testset(trainer, model, data_module, device="cuda" if torch.cuda.is_available() else "cpu",save_path = save_path)
    else:
        print(f"ERROR: best checkpoint not found at {best_ckpt}. Skipping test evaluation.")



def main():
    seed = random.randint(0, 10000)
    seed = random.randint(0, 2**32 - 1)
    set_seed(42) # 32
    print(f"Using random seed: {42}")

    parser, args = define_parser()

    n_img_views = 1
    if args.loss_type == "simclr": n_img_views = 2

    
   
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
    save_path = os.path.join(args.checkpoint_dir, args.comment)
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
        fast_dev_run= True,
    )


    # ---------------------------------------------------------------------------------------------------
    # 4) patch the lightning trainer
    # ---------------------------------------------------------------------------------------------------
    if args.fl: flare.patch(trainer)

    # ---------------------------------------------------------------------------------------------------
    # 5) Build Model
    # ---------------------------------------------------------------------------------------------------
    
    model = ClassificationNet(
        feature_dim=args.feature_dim,
        classes=classes,
        lr=args.lr,              # can tune
        wd = args.wd,
        loss_type=args.loss_type,     # or "ce"
        model_architecture = args.model_architecture,
        pretrained=args.pretrained,
        for_cls=args.for_cls,
        log_dir = trainer.log_dir,
        warmup_epochs=args.warmup_epochs,
        scheduler=args.scheduler,
        epochs = args.epochs,
    )

    # If resuming from a checkpoint (fine-tuning or continuing)
    if args.ckpt_path:
        print(f"Loading checkpoint from {args.ckpt_path}...")
        # only load state_dict, so our args dont get overwritten
        state_dict = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        # optionally reset MLP or freeze
        if args.finetune:
            model.reset_mlp()


    # ---------------------------------------------------------------------------------------------------
    # 6) Train and Evaluate, depeonds on args.fl
    # ---------------------------------------------------------------------------------------------------
    if args.fl:
        while flare.is_running():
            input_model = flare.receive()
            print(f"\n[Current Round={input_model.current_round}, Site = {flare.get_site_name()}]\n")
            #best_ckpt = trainer.fit(model=model, datamodule=data_module)
            best_ckpt = fit(trainer, model, data_module)
            #evaluate(args, trainer, model, data_module, best_ckpt, save_path)
            trainer.test(datamodule=data_module)
    else: 
        trainer.validate(model, datamodule=data_module)
        best_ckpt = fit(trainer, model, data_module)
        #evaluate(args, trainer, model, data_module, best_ckpt, save_path)
        if args.loss_type != 'simclr' and args.loss_type != 'mae': trainer.test(datamodule=data_module)

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