# --------------------------------------------------------------------------------------------------------------------------
# Modified from: https://github.com/sinagh72/TED by Sina Gholami
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
from run.util_test import evaluate_testset

def parse_args():
    parser = argparse.ArgumentParser(description="Training, Testing, and Inference Script")

    # Common training arguments
    parser.add_argument('--comment', type=str, default="local_model")
    parser.add_argument('--model_architecture', type=str, default="resnet50")
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-5) # 1e-3
    parser.add_argument('--feature_dim', type=int, default=2048)
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints")
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--root_dir', type=str, required=True,
                        help="Root directory containing train.csv, val.csv, test.csv")
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help="Checkpoint path to resume or fine-tune from. e.g. /path/to/epoch=6-step=28.ckpt")
    parser.add_argument('--finetune', type=bool, default=False,
                        help="Whether to reset MLP or freeze backbone for fine-tuning")
    parser.add_argument('--loss_type', type=str, default="focal", choices=["focal", "ce", "simclr", "mae"], help='which loss to use')

    parser.add_argument('--use_for_classification', type=bool, required=False, default=False,
                        help="For MAE, whether to use the model for classification or not. Default is False.")

    # Inference-only argument (to test a single image at the end)
    parser.add_argument('--inference_image', type=str, default=None,
                        help="Path to a single image for testing inference. e.g. /path/to/image.png")

    return parser.parse_args()


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

def main():
    set_seed(42) # 42
    args = parse_args()

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
        use_for_classification=args.use_for_classification,
    )

    # If resuming from a checkpoint (fine-tuning or continuing)
    if args.ckpt_path:
        print(f"Loading checkpoint from {args.ckpt_path}...")
        state_dict = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        # optionally reset MLP or freeze
        if args.finetune:
            model.reset_mlp()



    # ---------------------------------------------------------------------------------------------------
    # 5) TRAIN
    # ---------------------------------------------------------------------------------------------------
    trainer.fit(model=model, datamodule=data_module)

    # The best checkpoint is stored at:
    best_ckpt = trainer.checkpoint_callback.best_model_path
    print(f"\nBest checkpoint is at: {best_ckpt}")

    # ---------------------------------------------------------------------------------------------------
    # 6) Evaluate on Test Set with Best Checkpoint
    # ---------------------------------------------------------------------------------------------------
    
    
    if os.path.exists(best_ckpt) and args.loss_type != "simclr" and args.loss_type != "mae":
        # load best checkpoint weights
        best_state_dict = torch.load(best_ckpt, map_location="cpu")["state_dict"]
        model.load_state_dict(best_state_dict)
        print("\nEvaluating on test set with the best checkpoint...")
        evaluate_testset(trainer, model, data_module, device="cuda" if torch.cuda.is_available() else "cpu",save_path = save_path)
    else:
        print(f"ERROR: best checkpoint not found at {best_ckpt}. Skipping test evaluation.")

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

