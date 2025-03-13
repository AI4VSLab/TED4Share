# --------------------------------------------------------
# Modified from: https://github.com/sinagh72/TED by Sina Gholami
# --------------------------------------------------------

import os
import argparse
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from PIL import Image

from dataset.datamodule import CustomDatamodule
from models.classification import ClassificationNet
from util.get_models import get_baseline_model
from util.seed import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Training, Testing, and Inference Script")

    # Common training arguments
    parser.add_argument('--comment', type=str, default="local_model")
    parser.add_argument('--model_architecture', type=str, default="resnet50")
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--feature_dim', type=int, default=2048)
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints")
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--root_dir', type=str, required=True,
                        help="Root directory containing train.csv, val.csv, test.csv")
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help="Checkpoint path to resume or fine-tune from. e.g. /path/to/epoch=6-step=28.ckpt")
    parser.add_argument('--finetune', type=bool, default=False,
                        help="Whether to reset MLP or freeze backbone for fine-tuning")

    # Inference-only argument (to test a single image at the end)
    parser.add_argument('--inference_image', type=str, default=None,
                        help="Path to a single image for testing inference. e.g. /path/to/image.png")
    

    return parser.parse_args()

def evaluate_testset(trainer, model, datamodule, device='cuda'):
    """
    Evaluate model on the test set. Prints accuracy, classification report, confusion matrix.
    """
    # Load test data
    test_loader = datamodule.test_dataloader()

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc * 100:.2f}%")

    # Classification report
    classes_list = list(datamodule.classes.keys())  # e.g. ["TED_1", "CONT_"]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes_list))

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

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

if __name__ == "__main__":
    set_seed(42)
    args = parse_args()

    # ---------------------------------
    # 1) Define transforms
    # ---------------------------------
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),   # or RandomResizedCrop
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # ---------------------------------
    # 2) Create DataModule
    # ---------------------------------
    classes = {"TED_1": 1, "CONT_": 0}
    data_module = CustomDatamodule(
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        train_transform=train_transform,
        test_transform=test_transform,
        classes=classes
    )
    data_module.prepare_data()
    data_module.setup("train")
    data_module.setup("val")
    data_module.setup("test")

    # ---------------------------------
    # 3) Setup PL Trainer
    # ---------------------------------
    save_path = os.path.join(args.checkpoint_dir, args.comment)
    os.makedirs(save_path, exist_ok=True)

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=args.epochs,
        logger=TensorBoardLogger(save_path, name="tb_logs"),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            ModelCheckpoint(dirpath=save_path, save_top_k=1, monitor="val_loss", mode="min")
        ],
        log_every_n_steps=1,
    )

    # ---------------------------------
    # 4) Build Model
    # ---------------------------------
    encoder = get_baseline_model(pretrained=args.pretrained, model_architecture=args.model_architecture)
    model = ClassificationNet(
        feature_dim=args.feature_dim,
        encoder=encoder,
        classes=classes,
        lr=3e-5,              # can tune
        loss_type="focal"     # or "ce"
    )

    # If resuming from a checkpoint (fine-tuning or continuing)
    if args.ckpt_path:
        print(f"Loading checkpoint from {args.ckpt_path}...")
        state_dict = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict)
        # optionally reset MLP or freeze
        if args.finetune:
            model.reset_mlp()

    # ---------------------------------
    # 5) TRAIN
    # ---------------------------------
    trainer.fit(model=model, datamodule=data_module)

    # The best checkpoint is stored at:
    best_ckpt = trainer.checkpoint_callback.best_model_path
    print(f"\nBest checkpoint is at: {best_ckpt}")

    # ---------------------------------
    # 6) Evaluate on Test Set with Best Checkpoint
    # ---------------------------------
    if os.path.exists(best_ckpt):
        # load best checkpoint weights
        best_state_dict = torch.load(best_ckpt, map_location="cpu")["state_dict"]
        model.load_state_dict(best_state_dict)
        print("\nEvaluating on test set with the best checkpoint...")
        evaluate_testset(trainer, model, data_module, device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        print(f"ERROR: best checkpoint not found at {best_ckpt}. Skipping test evaluation.")

    # ---------------------------------
    # 7) Single-Image Inference (Optional)
    # ---------------------------------
    if args.inference_image is not None:
        print(f"\nRunning inference on single image: {args.inference_image}")
        prediction_idx = predict_single_image(
            model, 
            args.inference_image, 
            transform=test_transform,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        # Reverse-lookup classes
        # classes = {"TED_1": 1, "CONT_": 0}
        # so "TED_1" has index 1, "CONT_" has index 0
        # we can invert that:
        inv_map = {v: k for k, v in classes.items()}
        predicted_class = inv_map[prediction_idx]
        print(f"Predicted Class for {args.inference_image} is: {predicted_class}")

