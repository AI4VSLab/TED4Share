import os
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl

class CustomDataset(Dataset):
    def __init__(self, transform=None, dataset=None, img_type="RGB"):
        self.dataset = dataset
        self.transform = transform
        self.img_type = img_type

    def __getitem__(self, index):
        img_path, label = self.dataset[index]
        img_path = img_path.replace("\\", "/")
        img_view = self.load_img(img_path)
        # If a transform is provided, it can further resize or augment
        if self.transform:
            img_view = self.transform(img_view)
        else:
            # fallback -> convert to Tensor if transform is None
            img_view = torch.tensor(np.array(img_view)).permute(2, 0, 1).float() / 255.0
        return img_view, label

    def __len__(self):
        return len(self.dataset)

    def load_img(self, img_path):
        try:
            img = Image.open(img_path).convert(self.img_type)
            # Force 224x224 if you want a guarantee
            img = img.resize((224, 224))
            return img
        except (OSError, UnidentifiedImageError) as e:
            print(f"Error loading image {img_path}: {e}")
            return Image.new(self.img_type, (224, 224))

class CustomDatamodule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=32, train_transform=None, test_transform=None, classes={}, num_workers=4):
        super().__init__()
        self.root_dir = root_dir
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.classes = classes
        self.num_workers = num_workers

    def prepare_data(self):
        self.train_data = load_from_csv(self.root_dir, "train.csv")
        self.val_data = load_from_csv(self.root_dir, "val.csv")
        self.test_data = load_from_csv(self.root_dir, "test.csv")

    def setup(self, stage=None):
        if stage == "train":
            self.data_train = CustomDataset(transform=self.train_transform, dataset=self.train_data)
            print(f'Training data: {len(self.data_train)}')
        if stage == "val":
            self.data_val = CustomDataset(transform=self.test_transform, dataset=self.val_data)
        if stage == "test":
            self.data_test = CustomDataset(transform=self.test_transform, dataset=self.test_data)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

def load_from_csv(root_dir, csv_file):
    data_df = pd.read_csv(os.path.join(root_dir, csv_file))
    return [(row['directory'], row['label']) for _, row in data_df.iterrows()]
