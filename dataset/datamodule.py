import os
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl

class CustomDataset(Dataset):
    def __init__(self, 
                 transform=None, 
                 num_view = 1, 
                 dataset=None, 
                 img_type="RGB"
                 ):
        '''
        @params:
            num_view: this is for using simclr training, since we need 2 views
        '''
        self.dataset = dataset
        self.transform = transform
        self.img_type = img_type
        self.num_view = num_view

    def __getitem__(self, index):
        img_path, label = self.dataset[index]
        img_path = img_path.replace("\\", "/")
        img_view = self.load_img(img_path)

        # If a transform is provided, it can further resize or augment
        if self.transform:
            
            if self.num_view > 1:
                img_view = torch.stack([
                                        self.transform(img_view) 
                                        for _ in range(self.num_view)
                                        ])
            else: 
                img_view = self.transform(img_view)
        else:
            # fallback -> convert to Tensor if transform is None
            if self.num_view > 1:
                img_view = torch.stack([
                                        torch.tensor(np.array(img_view)).permute(2, 0, 1).float() / 255.0 
                                        for _ in range(self.num_view)
                                        ])
            else: 
                img_view = torch.tensor(np.array(img_view)).permute(2, 0, 1).float() / 255.0
        
        # Return a dictionary instead of a tuple
        return {
            'img': img_view,
            'label': label,
            'img_path': img_path
        }

    def __len__(self):
        return len(self.dataset)

    def load_img(self, img_path):
        try:
            img = Image.open(img_path).convert(self.img_type)
            W,H = img.size

            # this is for cropping white space
            # img_array = np.array(img)
            # region = img_array[:, int(W*0.12): int(W- W*0.12), : ]
            # img = Image.fromarray(region)

            # Force 224x224 if you want a guarantee
            # img = img.resize((224, 224))
            return img
        except (OSError, UnidentifiedImageError) as e:
            print(f'error loading image')
            return Image.new(self.img_type, (224, 224))

class CustomDatamodule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=32, 
                 train_transform=None, 
                 test_transform=None, 
                 classes={}, 
                 num_workers=32,
                 n_img_views = 1
                 ):
        super().__init__()
        self.root_dir = root_dir
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.classes = classes
        self.num_workers = num_workers
        self.n_img_views = n_img_views

    def prepare_data(self):
        self.train_data = load_from_csv(self.root_dir, "train.csv")
        self.val_data = load_from_csv(self.root_dir, "val.csv")
        self.test_data = load_from_csv(self.root_dir, "test.csv")

    def setup(self, stage=None):
        if stage == "train":
            self.data_train = CustomDataset(transform=self.train_transform, dataset=self.train_data, num_view = self.n_img_views )
            print(f'Training data: {len(self.data_train)}')
        if stage == "val":
            self.data_val = CustomDataset(transform=self.test_transform, dataset=self.val_data, num_view = self.n_img_views )
        if stage == "test":
            self.data_test = CustomDataset(transform=self.test_transform, dataset=self.test_data)

    def train_dataloader(self):
        return DataLoader(
                        self.data_train, 
                        batch_size=self.batch_size, 
                        shuffle=True, 
                        num_workers=self.num_workers
                        )

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

def load_from_csv(root_dir, csv_file):
    data_df = pd.read_csv(os.path.join(root_dir, csv_file))
    return [(row['directory'], row['label']) for _, row in data_df.iterrows()]
