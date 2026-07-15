"""Data loading module for all datasets."""

import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from .augmentations import get_train_transform, get_eval_transform

# ---------- ImageNet paths ----------
TINY_IMAGENET_PATH = "/mnt/e/Project/MDPI/data/tiny-imagenet-200"
CIFAR10C_PATH = "/mnt/e/Project/MDPI/data/CIFAR-10-C"


def get_dataset(dataset: str, aug_strategy: str, split: str = "train",
                input_size: int = 224):
    """
    Get dataset with specified augmentation.

    Args:
        dataset: 'cifar10', 'cifar100', 'tinyimagenet'
        aug_strategy: key into augmentation strategies
        split: 'train' or 'test'
        input_size: target image size (224 for ViT, 32 for CIFAR-ResNet)
    """
    data_dir = "/mnt/e/Project/MDPI/data"

    if dataset in ("cifar10", "cifar100"):
        cls = CIFAR10 if dataset == "cifar10" else CIFAR100
        need_download = (dataset == "cifar10")  # CIFAR-10 needs DL, CIFAR-100 already exists
        if split == "train":
            transform = get_train_transform(dataset, aug_strategy, input_size=input_size)
            return cls(root=data_dir, train=True, download=need_download, transform=transform)
        else:
            transform = get_eval_transform(dataset, input_size=input_size)
            return cls(root=data_dir, train=False, download=need_download, transform=transform)

    elif dataset == "tinyimagenet":
        if split == "train":
            transform = get_train_transform(dataset, aug_strategy, input_size=input_size)
            path = os.path.join(TINY_IMAGENET_PATH, "train")
            return ImageFolder(root=path, transform=transform)
        else:
            transform = get_eval_transform(dataset, input_size=input_size)
            path = os.path.join(TINY_IMAGENET_PATH, "val")
            return ImageFolder(root=path, transform=transform)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_dataloader(dataset: str, aug_strategy: str, split: str,
                   batch_size: int = 128, num_workers: int = 4,
                   input_size: int = 224, shuffle: bool = None,
                   data_fraction: float = 1.0, seed: int = 42):
    """Get DataLoader for experiment.

    data_fraction: if < 1.0, subsample the TRAIN set to that fraction (class-balanced-ish
                   via seeded shuffle) — used for data-efficiency experiments.
    """
    ds = get_dataset(dataset, aug_strategy, split, input_size=input_size)

    # Subsample training data for data-efficiency experiments
    if split == "train" and data_fraction < 1.0:
        g = torch.Generator().manual_seed(seed)
        n = len(ds)
        n_keep = max(1, int(n * data_fraction))
        idx = torch.randperm(n, generator=g)[:n_keep].tolist()
        ds = Subset(ds, idx)

    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=(split == "train"),
    )
