"""PyTorch based class to read our dataset."""
import os
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


OUTPUT_CLASS = {"normal": 0, "bacteria": 1, "virus": 2}
ZERO_TENSOR = torch.zeros([3], dtype=torch.float32)

class XRayDataset(Dataset):
    """XRayDataset Class."""

    def __init__(self, overview_file, dataset_path, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(os.path.join(dataset_path, overview_file))
        self.dataset_path = dataset_path
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, self.img_labels.iloc[idx, 2])
        image = read_image(img_path) / 255.
        label = torch.zeros([3])
        label[OUTPUT_CLASS[self.img_labels.iloc[idx, 1]]] = 1.
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


