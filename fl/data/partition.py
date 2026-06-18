"""
Téléchargement et partitionnement du dataset MNIST.
Supporte la distribution IID et Non-IID (Dirichlet).
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms


def load_mnist(data_dir: str = "/data/mnist"):
    """Télécharge MNIST et retourne (train_dataset, test_dataset)."""
    os.makedirs(data_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return train, test


def partition_iid(dataset, num_workers: int, worker_id: int):
    """
    Partitionnement IID : chaque worker reçoit un fragment aléatoire égal.
    worker_id : 0-indexed (0 à num_workers-1)
    """
    n = len(dataset)
    indices = np.random.permutation(n)
    size = n // num_workers
    start = worker_id * size
    end   = start + size if worker_id < num_workers - 1 else n
    return Subset(dataset, indices[start:end])


def partition_dirichlet(dataset, num_workers: int, worker_id: int, alpha: float = 0.5):
    """
    Partitionnement Non-IID via distribution de Dirichlet.
    alpha faible (0.1) = très hétérogène ; alpha élevé (10) ≈ IID.
    """
    labels = np.array(dataset.targets)
    num_classes = 10
    # Distribution Dirichlet par classe
    np.random.seed(42)
    label_dist = np.random.dirichlet([alpha] * num_workers, num_classes)

    worker_indices = []
    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        np.random.shuffle(class_idx)
        # Répartir selon la distribution Dirichlet
        splits = (label_dist[c] * len(class_idx)).astype(int)
        splits[-1] = len(class_idx) - splits[:-1].sum()  # ajustement
        start = 0
        for w in range(num_workers):
            end = start + splits[w]
            if w == worker_id:
                worker_indices.extend(class_idx[start:end].tolist())
            start = end

    return Subset(dataset, worker_indices)


def get_worker_dataloader(cfg: dict, worker_id: int, batch_size: int = 32):
    """
    Retourne un DataLoader pour le worker donné.
    Lit la config pour choisir IID vs Non-IID.
    """
    d = cfg.get("dataset", {})
    num_workers = d.get("num_workers", 5)
    iid         = d.get("iid", True)
    alpha       = d.get("dirichlet_alpha", 0.5)

    train_ds, _ = load_mnist()

    if iid:
        subset = partition_iid(train_ds, num_workers, worker_id)
    else:
        subset = partition_dirichlet(train_ds, num_workers, worker_id, alpha)

    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    print(f"[Worker-{worker_id}] Dataset: {len(subset)} samples "
          f"({'IID' if iid else f'Non-IID α={alpha}'})")
    return loader, len(subset)


def get_validation_dataloader(batch_size: int = 256):
    """Dataset de validation central (test set MNIST complet)."""
    _, test_ds = load_mnist()
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)
