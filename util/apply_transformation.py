# --------------------------------------------------------
# get the image transformation for training and validation
# Written by Sina Gholami
# -------------------
import torch
import torchvision.transforms.v2 as transforms
from torchvision.transforms import InterpolationMode, AutoAugmentPolicy, AutoAugment, RandAugment
#from torchvision import transforms
from dataset.custom_aug import RandomCutOut


def ensure_three_channels(img):
    """Convert an image to 3 channels (RGB)."""
    if img.mode != 'RGB':
        img = img.convert("RGB")
    return img




def get_transforms(args):
    if args.loss_type == 'simclr':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.3, 1.0)),
            transforms.RandomRotation((0,75)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
        ])

        # simclr augmnetations
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.1, 0.5)),
            transforms.RandomRotation((0,360)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            RandomCutOut( (args.img_size, args.img_size), min_cutout=0.05, max_cutout=0.5 )
        ])
    elif args.loss_type == 'mae':
        train_transform = transforms.Compose([
            #transforms.Resize((args.img_size, args.img_size)),   # or RandomResizedCrop
            transforms.RandomResizedCrop(args.img_size, scale=(0.3, 1.0)),
            transforms.ToTensor(),
        ])

    elif args.loss_type == 'focal':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.3, 1.0)), #0.9, 0.4
            transforms.RandomRotation((0,75)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    elif args.loss_type == 'ce': 
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.9, 1.0)),
            #transforms.RandomRotation((0,5)),
            transforms.RandomHorizontalFlip(),
            RandAugment(num_ops=2, magnitude=7),
            transforms.ToTensor(),
        ])
        
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")

    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406],
        #                     [0.229, 0.224, 0.225]),
    ])
    return train_transform, test_transform