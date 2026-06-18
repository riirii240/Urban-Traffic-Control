"""
Worker (client FL) — entraînement local + communication gRPC avec le serveur.
Chaque worker :
  1. S'enregistre auprès du serveur
  2. Récupère le modèle global
  3. Entraîne localement sur ses données
  4. Envoie ses poids mis à jour au serveur
  5. Répète jusqu'à convergence
"""

import os
import sys
import time
import random
import logging
import yaml
import grpc
import torch
import torch.nn as nn
import torch.optim as optim

# Ajouter les chemins pour les imports
sys.path.insert(0, "/app")

from server.proto import federated_pb2, federated_pb2_grpc
from client.model  import get_model, weights_to_bytes, bytes_to_weights, compute_checksum, verify_checksum
from data.partition import get_worker_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] Worker-%(worker_id)s: %(message)s",
    defaults={"worker_id": os.environ.get("WORKER_ID", "?")}
)
log = logging.getLogger(__name__)


def load_config(path="/app/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def create_grpc_stub(host: str, port: int, retries: int = 10):
    """Crée un stub gRPC avec retry exponentiel."""
    for attempt in range(retries):
        try:
            channel = grpc.insecure_channel(
                f"{host}:{port}",
                options=[
                    ('grpc.max_send_message_length',    50 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 50 * 1024 * 1024),
                ]
            )
            grpc.channel_ready_future(channel).result(timeout=5)
            log.info(f"Connecté au serveur {host}:{port}")
            return federated_pb2_grpc.FederatedServiceStub(channel)
        except Exception as e:
            wait = min(2 ** attempt, 30)
            log.warning(f"Tentative {attempt+1}/{retries} échouée ({e}). Retry dans {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Impossible de se connecter au serveur {host}:{port}")


def train_local(model, loader, epochs: int, lr: float, device):
    """
    Entraînement local : E époques sur les données privées du worker.
    Retourne (loss_moyenne, accuracy_moyenne).
    """
    model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    for epoch in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss    += loss.item() * len(labels)
            preds          = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)

    avg_loss = total_loss    / total_samples
    avg_acc  = total_correct / total_samples
    return avg_loss, avg_acc


def simulate_fault(cfg: dict, worker_id: int):
    """Simule une panne aléatoire selon la configuration."""
    fault = cfg.get("fault", {})
    crash_prob = fault.get("crash_probability", 0.0)
    byz_workers = fault.get("byzantine_workers", [])

    if worker_id in byz_workers:
        log.warning("Mode BYZANTINE activé — envoi de poids aléatoires")
        return "byzantine"

    if random.random() < crash_prob:
        log.warning("Simulation de CRASH — ce round est ignoré")
        return "crash"

    latency_max = fault.get("latency_max_ms", 0)
    if latency_max > 0:
        latency_min = fault.get("latency_min_ms", 0)
        delay = random.uniform(latency_min, latency_max) / 1000.0
        log.info(f"Simulation latence : {delay*1000:.0f}ms")
        time.sleep(delay)

    return "ok"


def run_worker():
    cfg = load_config()

    worker_id  = int(os.environ.get("WORKER_ID", "0"))
    server_host = cfg["server"]["host"] if cfg["server"]["host"] != "0.0.0.0" else "server"
    server_host = os.environ.get("SERVER_HOST", "server")
    server_port = int(os.environ.get("SERVER_PORT", cfg["server"]["port"]))
    epochs      = cfg["training"]["epochs"]
    lr          = cfg["training"]["learning_rate"]
    batch_size  = cfg["training"]["batch_size"]
    device      = torch.device("cpu")  # CPU pour Docker

    log.info(f"Démarrage Worker-{worker_id} → serveur {server_host}:{server_port}")

    # Charger les données locales
    loader, num_samples = get_worker_dataloader(cfg, worker_id, batch_size)
    log.info(f"Dataset local : {num_samples} samples")

    # Connexion gRPC
    stub = create_grpc_stub(server_host, server_port)

    # Enregistrement auprès du serveur
    ack = stub.Register(federated_pb2.ClientInfo(
        client_id=f"worker-{worker_id}",
        num_samples=num_samples,
        host=os.environ.get("HOSTNAME", "localhost"),
    ))
    log.info(f"Enregistrement : {ack.message}")

    # Instancier le modèle local
    model = get_model(cfg)

    # ── BOUCLE PRINCIPALE FL ──────────────────────────────────
    current_round = 0
    while True:
        try:
            # 1. Récupérer le modèle global
            global_weights = stub.GetGlobalModel(
                federated_pb2.RoundRequest(
                    client_id=f"worker-{worker_id}",
                    round_id=current_round,
                ),
                timeout=cfg["server"]["round_timeout"],
            )

            # Si le serveur n'est pas prêt, attendre
            if global_weights.round_id == -1:
                log.info("Serveur pas encore prêt, attente...")
                time.sleep(3)
                continue

            # Si round_id == -2, l'entraînement est terminé
            if global_weights.round_id == -2:
                log.info("Entraînement terminé par le serveur. Au revoir !")
                break

            current_round = global_weights.round_id
            log.info(f"Round {current_round} démarré")

            # Vérifier l'intégrité des poids reçus
            if not verify_checksum(global_weights.weights, global_weights.checksum):
                log.error("Checksum invalide ! Poids corrompus, round ignoré.")
                time.sleep(2)
                continue

            # Charger les poids globaux dans le modèle local
            bytes_to_weights(model, global_weights.weights)

            # 2. Simuler une panne éventuelle
            fault_status = simulate_fault(cfg, worker_id)

            if fault_status == "crash":
                time.sleep(cfg["server"]["round_timeout"] + 10)
                continue

            # 3. Entraînement local
            log.info(f"Entraînement local : {epochs} époques...")

            if fault_status == "byzantine":
                # Envoyer des poids aléatoires (attaque Byzantine)
                import io
                noise_model = get_model(cfg)
                for param in noise_model.parameters():
                    nn.init.uniform_(param, -10, 10)
                weights_data = weights_to_bytes(noise_model)
                train_loss, train_acc = 9.99, 0.10
            else:
                train_loss, train_acc = train_local(model, loader, epochs, lr, device)
                weights_data = weights_to_bytes(model)

            log.info(f"Round {current_round} — loss={train_loss:.4f}, acc={train_acc:.4f}")

            # 4. Envoyer les poids mis à jour
            checksum = compute_checksum(weights_data)
            ack = stub.SubmitUpdate(
                federated_pb2.LocalUpdate(
                    client_id=f"worker-{worker_id}",
                    weights=weights_data,
                    round_id=current_round,
                    num_samples=num_samples,
                    checksum=checksum,
                    train_loss=train_loss,
                    train_acc=train_acc,
                ),
                timeout=cfg["server"]["round_timeout"],
            )

            if ack.success:
                log.info(f"Round {current_round} — poids envoyés avec succès")
            else:
                log.warning(f"Round {current_round} — rejeté : {ack.message}")

            current_round += 1

        except grpc.RpcError as e:
            log.error(f"Erreur gRPC : {e.code()} — {e.details()}")
            log.info("Reconnexion dans 5s...")
            time.sleep(5)
            stub = create_grpc_stub(server_host, server_port, retries=5)

        except Exception as e:
            log.error(f"Erreur inattendue : {e}")
            time.sleep(3)


if __name__ == "__main__":
    run_worker()
