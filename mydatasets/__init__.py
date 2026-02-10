from .cifar100_dataset import CIFAR100DataModule
from .tinyimagenet_dataset import TinyImageNetDataModule
from .imagenet1k_dataset import ImageNet1kDataModule

def load_dataset(dataset_name: str, data_dir: str, batch_size: int, image_size: int):
    if dataset_name == 'cifar100':
        return CIFAR100DataModule(data_dir=data_dir, batch_size=batch_size, image_size=image_size)
    elif dataset_name == 'tinyimagenet':
        return TinyImageNetDataModule(data_dir=data_dir, batch_size=batch_size, image_size=image_size)
    elif dataset_name == 'imagenet1k':
        return ImageNet1kDataModule(data_dir=data_dir, batch_size=batch_size, image_size=image_size)
    else:
        raise ValueError(f"Dataset desconocido: {dataset_name}")