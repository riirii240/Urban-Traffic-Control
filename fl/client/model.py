"""
Modèle MLP PyTorch pour MNIST.
Architecture : 784 → 128 → 64 → 10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import hashlib
import numpy as np


class MLP(nn.Module):
    """Perceptron multicouche pour la classification MNIST."""

    def __init__(self, input_size=784, hidden1=128, hidden2=64, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 784)          # aplatir l'image 28×28
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def get_model(cfg: dict) -> MLP:
    """Instancie le modèle depuis la configuration."""
    m = cfg.get("model", {})
    return MLP(
        input_size=m.get("input_size", 784),
        hidden1=m.get("hidden1", 128),
        hidden2=m.get("hidden2", 64),
        num_classes=m.get("num_classes", 10),
    )


# ── Sérialisation / Désérialisation des poids ──────────────────

def weights_to_bytes(model: nn.Module) -> bytes:
    """Convertit les poids du modèle en bytes (pour gRPC)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def bytes_to_weights(model: nn.Module, data: bytes) -> nn.Module:
    """Charge des bytes dans un modèle existant."""
    buf = io.BytesIO(data)
    state_dict = torch.load(buf, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def compute_checksum(data: bytes) -> str:
    """Calcule un SHA-256 des bytes pour vérifier l'intégrité."""
    return hashlib.sha256(data).hexdigest()


def verify_checksum(data: bytes, expected: str) -> bool:
    """Vérifie que le SHA-256 correspond."""
    return compute_checksum(data) == expected
