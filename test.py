import os
import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from PIL import Image

from dataset.datamodule import CustomDataset, load_from_csv
from models.classification import ClassificationNet
from util.get_models import get_baseline_model

def parse_args():
    parser = argparse.ArgumentParser(description="Testing configuration")
    parser.add_argument('--root_dir', type=str, required=True, help="Root directory with test.csv")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to the trained checkpoint")
    parser.add_argument('--img_size', type=int, default=224, help="Image size")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for testing")
    parser.add_argument('--classes', type=str, nargs='+', default=["TED_1", "CONT_"], help="Class names")
    return parser.parse_args()

def load_model(ckpt_path, classes, img_size):
    """
    Load the trained model from checkpoint.
    """
    # Load encoder (backbone)
    encoder = get_baseline_model(pretrained=False, model_architecture="resnet50")

    # Load classification model
    model = ClassificationNet(
        feature_dim=2048,
        encoder=encoder,
        classes={name: idx for idx, name in enumerate(classes)},
        lr=3e-5
    )

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def evaluate_model(model, dataloader, device, class_names):
    """
    Evaluate the model on the test dataset.
    """
    model.to(device)
    model.eval()

    predictions, ground_truths = [], []

    with torch.no_grad():
        for batch in dataloader:
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            ground_truths.extend(labels.cpu().numpy())

    # Accuracy
    accuracy = accuracy_score(ground_truths, predictions)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(ground_truths, predictions, target_names=class_names))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(ground_truths, predictions))

def predict_single_image(model, image_path, transform, device):
    """
    Predict the class of a single image.
    """
    model.to(device)
    model.eval()

    with torch.no_grad():
        # Load and transform the image
        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        # Forward pass
        logits = model(img)
        preds = torch.argmax(logits, dim=1)
        return preds.item()

if __name__ == "__main__":
    args = parse_args()

    assert args.ckpt_path != '', "Please provide the path to the trained checkpoint"
    assert args.root_dir != '', "Please provide the root directory with test.csv"


    # Define test image transforms
    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load the test dataset
    test_data = load_from_csv(args.root_dir, "test.csv")
    test_dataset = CustomDataset(transform=test_transform, dataset=test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    # Load the trained model
    model = load_model(args.ckpt_path, args.classes, args.img_size)

    # Evaluate the model on the test dataset
    print("\nEvaluating the model on the test dataset...")
    evaluate_model(model, test_dataloader, device="cuda" if torch.cuda.is_available() else "cpu", class_names=args.classes)

    # Example: Test with a single image
    single_image_path = "/home/CenteredData/TED Federated Learning Project/Photos/TED_1019.png"  # Replace with the path to your image
    print("\nTesting a single image...")
    prediction = predict_single_image(
        model,
        single_image_path,
        test_transform,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Predicted Class for '{single_image_path}': {args.classes[prediction]}")
