# --------------------------------------------------------
# get the image transformation for training and validation
# Written by Sina Gholami
# -------------------
import torch
import torchvision.transforms.v2 as T
from torchvision.transforms import InterpolationMode, AutoAugmentPolicy, AutoAugment


def ensure_three_channels(img):
    """Convert an image to 3 channels (RGB)."""
    if img.mode != 'RGB':
        img = img.convert("RGB")
    return img


def get_train_transformation(img_size):
    return T.Compose([
        T.Resize((img_size, img_size), InterpolationMode.BICUBIC),
        T.Lambda(ensure_three_channels),
        AutoAugment(AutoAugmentPolicy.IMAGENET),
        T.ToImage(),  # Convert to tensor, only if you had a PIL image
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


def get_test_transformation(img_size):
    return T.Compose([
        T.Resize((img_size, img_size), InterpolationMode.BICUBIC),
        T.Lambda(ensure_three_channels),
        T.ToImage(),  # Convert to tensor, only if you had a PIL image
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
