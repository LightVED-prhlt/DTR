import os
import multiprocessing

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl

class TinyImageNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = '/home/Data/datasets/tiny-imagenet-200', batch_size: int = 128, image_size: int = 224):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = multiprocessing.cpu_count() - 1
        self.num_classes = 200

        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        ])

    def prepare_data(self):
        # Load Tiny ImageNet dataset
        datasets.ImageFolder(root=os.path.join(self.data_dir, 'train'), transform=self.train_transforms)
        datasets.ImageFolder(root=os.path.join(self.data_dir, 'val'), transform=self.val_transforms)
        datasets.ImageFolder(root=os.path.join(self.data_dir, 'test'), transform=self.val_transforms)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = datasets.ImageFolder(root=os.path.join(self.data_dir, 'train'), transform=self.train_transforms)
            self.val_dataset = datasets.ImageFolder(root=os.path.join(self.data_dir, 'val'), transform=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
