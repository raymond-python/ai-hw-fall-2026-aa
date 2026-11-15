"""MNIST loading. Images stay as raw pixels in [0, 1]; models normalise inside.

Rule of the assignment: train on the TRAIN split, test on the TEST split.
The last 5,000 training images are held out as validation (to pick the best
epoch), so the test set is never used for any training decision.
"""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

VAL_SIZE = 5000
to_tensor = transforms.ToTensor()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_val_loaders(data_dir="data", batch_size=128, workers=2):
    full = datasets.MNIST(data_dir, train=True, download=True, transform=to_tensor)
    n = len(full)
    train_ds = Subset(full, range(n - VAL_SIZE))
    val_ds = Subset(full, range(n - VAL_SIZE, n))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=workers)
    return train_loader, val_loader


def test_loader(data_dir="data", batch_size=500, workers=2):
    ds = datasets.MNIST(data_dir, train=False, download=True, transform=to_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)


def attack_set(data_dir="data", n=2000):
    """The first n TEST images, as tensors. Every model is attacked on exactly
    the same images so the numbers are comparable."""
    ds = datasets.MNIST(data_dir, train=False, download=True)
    n = min(n, len(ds))
    x = ds.data[:n].float().div(255.0).unsqueeze(1)          # (n, 1, 28, 28)
    y = ds.targets[:n].clone()
    return x, y


def tensor_loader(x, y, batch_size=500):
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
