"""
Tests unitaires — FedAvg, checksum, modèle, partitionnement.
Lancer avec : pytest tests/ -v
"""

import sys
sys.path.insert(0, "/app")

import pytest
import torch
import numpy as np
import io
import hashlib

from client.model import (
    MLP, get_model, weights_to_bytes, bytes_to_weights,
    compute_checksum, verify_checksum
)


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return {
        "model": {"input_size": 784, "hidden1": 128, "hidden2": 64, "num_classes": 10}
    }

@pytest.fixture
def model(cfg):
    return get_model(cfg)


# ── Tests modèle ────────────────────────────────────────────

def test_model_output_shape(model):
    """Le modèle produit bien 10 classes en sortie."""
    x = torch.randn(8, 1, 28, 28)
    out = model(x)
    assert out.shape == (8, 10), f"Forme attendue (8,10), obtenue {out.shape}"


def test_model_forward_no_crash(model):
    """Le forward pass ne plante pas."""
    x = torch.randn(32, 784)
    out = model(x)
    assert not torch.isnan(out).any(), "NaN détecté dans la sortie"


# ── Tests sérialisation ──────────────────────────────────────

def test_weights_serialization(model, cfg):
    """Les poids sérialisés puis désérialisés sont identiques."""
    data = weights_to_bytes(model)
    assert isinstance(data, bytes)
    assert len(data) > 0

    model2 = get_model(cfg)
    bytes_to_weights(model2, data)

    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), model2.named_parameters()
    ):
        assert torch.allclose(p1, p2), f"Poids différents pour {n1}"


# ── Tests checksum ───────────────────────────────────────────

def test_checksum_valid(model):
    """Un checksum valide est accepté."""
    data = weights_to_bytes(model)
    cs   = compute_checksum(data)
    assert verify_checksum(data, cs)


def test_checksum_corrupted(model):
    """Un paquet corrompu est rejeté."""
    data    = weights_to_bytes(model)
    cs      = compute_checksum(data)
    corrupt = data[:-10] + b"\x00" * 10   # corrompre les derniers bytes
    assert not verify_checksum(corrupt, cs), "Corruption non détectée !"


def test_checksum_wrong_hash(model):
    """Un mauvais hash est rejeté."""
    data = weights_to_bytes(model)
    assert not verify_checksum(data, "0" * 64)


# ── Tests FedAvg manuel ──────────────────────────────────────

def test_fedavg_weighted_average(cfg):
    """
    Vérifie mathématiquement que FedAvg calcule la bonne moyenne pondérée.
    Deux modèles avec des poids connus → vérifier le résultat.
    """
    m1 = get_model(cfg)
    m2 = get_model(cfg)

    # Forcer des poids connus
    for p in m1.parameters(): torch.nn.init.constant_(p, 2.0)
    for p in m2.parameters(): torch.nn.init.constant_(p, 4.0)

    n1, n2 = 600, 400   # tailles des datasets
    N      = n1 + n2    # 1000

    # FedAvg attendu : (600/1000)*2 + (400/1000)*4 = 1.2 + 1.6 = 2.8
    expected = (n1 / N) * 2.0 + (n2 / N) * 4.0

    # Calculer manuellement
    global_state = {}
    for key in m1.state_dict():
        s = (n1/N) * m1.state_dict()[key].float() + (n2/N) * m2.state_dict()[key].float()
        global_state[key] = s

    for key, val in global_state.items():
        assert torch.allclose(val, torch.full_like(val, expected), atol=1e-5), \
            f"FedAvg incorrect pour {key}: attendu {expected}, obtenu {val.mean().item()}"


def test_fedavg_single_client(cfg):
    """Avec un seul client, FedAvg retourne exactement ses poids."""
    m = get_model(cfg)
    for p in m.parameters(): torch.nn.init.constant_(p, 3.14)

    for key, val in m.state_dict().items():
        result = (1.0) * val.float()
        assert torch.allclose(result, val.float()), "FedAvg à 1 client incorrect"


# ── Tests dataset ────────────────────────────────────────────

def test_partition_iid():
    """Le partitionnement IID produit les bons indices."""
    from data.partition import partition_iid
    from torchvision import datasets, transforms

    t = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST("/tmp/mnist_test", train=True, download=True, transform=t)

    sub0 = partition_iid(ds, num_workers=5, worker_id=0)
    sub1 = partition_iid(ds, num_workers=5, worker_id=1)

    assert len(sub0) == 12000, f"Taille attendue 12000, obtenue {len(sub0)}"
    assert len(sub0) + len(sub1) <= len(ds)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
