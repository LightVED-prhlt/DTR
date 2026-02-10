import multiprocessing
import numpy as np

from torch import Generator
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split

from timm.data import (
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
)

class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = '/home/Data/datasets', batch_size: int = 128, image_size: int = 224):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = 4 # multiprocessing.cpu_count() - 1
        self.num_classes = 100

        self.train_transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandAugment(num_ops=9, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            transforms.RandomErasing(p=0.25),
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])

    def prepare_data(self):
        # Download CIFAR-100 dataset
        datasets.CIFAR100(root=self.data_dir, train=True, download=True)
        datasets.CIFAR100(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_full_dataset = datasets.CIFAR100(root=self.data_dir, train=True, transform=self.train_transforms)
            targets = np.array(train_full_dataset.targets)

            # Split dataset into training and validation sets (80-20 split) | Stratiﬁed sampling
            train_idx, val_idx = train_test_split(np.arange(len(targets)), test_size=0.2, stratify=targets, random_state=42)

            self.train_dataset = Subset(train_full_dataset, train_idx)
            self.val_dataset = Subset(train_full_dataset, val_idx)

        if stage == 'validate' or stage is None:
            train_full_dataset = datasets.CIFAR100(root=self.data_dir, train=True, transform=self.train_transforms)
            targets = np.array(train_full_dataset.targets)

            # Split dataset into training and validation sets (80-20 split) | Stratiﬁed sampling
            train_idx, val_idx = train_test_split(np.arange(len(targets)), test_size=0.2, stratify=targets, random_state=42)

            self.train_dataset = Subset(train_full_dataset, train_idx)
            self.val_dataset = Subset(train_full_dataset, val_idx)

        if stage == 'test' or stage is None:
            self.test_dataset = datasets.CIFAR100(root=self.data_dir, train=False, transform=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, generator=Generator().manual_seed(42))
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, generator=Generator().manual_seed(42))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, generator=Generator().manual_seed(42))
