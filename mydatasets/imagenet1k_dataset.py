import os
import multiprocessing

import timm
from torch import Generator
from torch.utils.data import DataLoader
from torchvision import datasets
import lightning as L


class ImageNet1kDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "/home/Data/datasets/imagenet1k", batch_size: int = 128, image_size: int = 224):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = 4 # multiprocessing.cpu_count() - 1
        self.num_classes = 1000

        self.train_transforms = timm.data.create_transform(
            input_size=(3, self.image_size, self.image_size),
            is_training=True,
            # auto_augment="original"
            auto_augment='rand-m9-mstd0.5-inc1'
        )

        self.val_transforms = timm.data.create_transform(
            input_size=(3, self.image_size, self.image_size),
            is_training=False,
            # auto_augment="original",
            auto_augment='rand-m9-mstd0.5-inc1'
        )

    def prepare_data(self):
        # Comprobar si los datos están en la estructura correcta
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'val')
        
        # Comprobar que existen los directorios
        if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
            raise FileNotFoundError("Los directorios 'train' y 'val' no se encontraron en la ruta especificada.")
        
        # Comprobar que cada directorio contiene subdirectorios para cada clase
        if not all(os.path.isdir(os.path.join(train_dir, d)) for d in os.listdir(train_dir)):
            raise FileNotFoundError("El directorio 'train' no contiene subdirectorios para cada clase.")
        if not all(os.path.isdir(os.path.join(val_dir, d)) for d in os.listdir(val_dir)):
            raise FileNotFoundError("El directorio 'val' no contiene subdirectorios para cada clase.")
        
    def setup(self, stage=None):
        self.train_dataset = datasets.ImageFolder(root=os.path.join(self.data_dir, 'train'), transform=self.train_transforms)
        self.val_dataset = datasets.ImageFolder(root=os.path.join(self.data_dir, 'val'), transform=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, generator=Generator().manual_seed(42), drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, generator=Generator().manual_seed(42))
