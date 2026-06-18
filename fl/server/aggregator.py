"""
Serveur d'agrégation — implémente FedAvg et coordonne les rounds fédérés.

Fonctionnement :
  1. Initialise le modèle global
  2. Attend que les workers s'enregistrent (min_clients)
  3. Pour chaque round :
     a. Sélectionne fraction_fit% des workers
     b. Leur envoie le modèle global
     c. Attend leurs mises à jour (avec timeout)
     d. Agrège via FedAvg
     e. Évalue le modèle global
  4. S'arrête après max_rounds ou convergence
"""

import sys
import time
import logging
import threading
import io
import hashlib
import yaml
import torch
import torch.nn as nn
import numpy as np
import redis
import grpc
from concurrent import futures
from datetime import datetime

sys.path.insert(0, "/app")

from server.proto import federated_pb2, federated_pb2_grpc
from client.model  import get_model, weights_to_bytes, bytes_to_weights, compute_checksum, verify_checksum
from data.partition import get_validation_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] Serveur: %(message)s"
)
log = logging.getLogger(__name__)


def load_config(path="/app/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


class FederatedServer(federated_pb2_grpc.FederatedServiceServicer):
    """
    Serveur gRPC qui implémente le protocole d'apprentissage fédéré.
    """

    def __init__(self, cfg: dict):
        self.cfg          = cfg
        self.max_rounds   = cfg["server"]["max_rounds"]
        self.min_clients  = cfg["server"]["min_clients"]
        self.fraction_fit = cfg["server"]["fraction_fit"]
        self.timeout      = cfg["server"]["round_timeout"]

        # État du système
        self.current_round  = 0
        self.is_training    = False
        self.training_done  = False

        # Registre des clients
        self.clients        = {}         # client_id → {num_samples, host, alive}
        self.clients_lock   = threading.Lock()

        # Poids reçus pour le round courant
        self.round_updates  = {}         # client_id → LocalUpdate
        self.updates_lock   = threading.Lock()
        self.updates_event  = threading.Event()

        # Métriques globales
        self.global_accuracy = 0.0
        self.global_loss     = 2.3
        self.metrics_history = []        # [{round, acc, loss}]
        self.client_metrics  = {}        # client_id → {loss, acc, responded}

        # Modèle global
        self.model       = get_model(cfg)
        self.model_bytes = weights_to_bytes(self.model)
        self.model_lock  = threading.Lock()

        # Dataset de validation
        self.val_loader = get_validation_dataloader()

        # Redis pour le registry
        r_cfg = cfg.get("redis", {})
        try:
            self.redis = redis.Redis(
                host=r_cfg.get("host", "redis"),
                port=r_cfg.get("port", 6379),
                decode_responses=True,
                socket_timeout=2,
            )
            self.redis.ping()
            log.info("Redis connecté")
        except Exception:
            log.warning("Redis non disponible — registry en mémoire uniquement")
            self.redis = None

        log.info(f"Serveur initialisé — max_rounds={self.max_rounds}, "
                 f"min_clients={self.min_clients}")

    # ── gRPC RPC ────────────────────────────────────────────────

    def Register(self, request, context):
        """Un worker s'enregistre au démarrage."""
        cid = request.client_id
        with self.clients_lock:
            self.clients[cid] = {
                "num_samples": request.num_samples,
                "host":        request.host,
                "alive":       True,
                "registered":  datetime.now().isoformat(),
            }
            self.client_metrics[cid] = {
                "loss": 0.0, "acc": 0.0, "responded": False
            }

        if self.redis:
            self.redis.hset(f"client:{cid}", mapping={
                "num_samples": request.num_samples,
                "host": request.host,
                "status": "registered",
            })

        log.info(f"Worker enregistré : {cid} ({request.num_samples} samples)")
        return federated_pb2.Ack(success=True,
                                  message=f"Bienvenue {cid} !")

    def GetGlobalModel(self, request, context):
        """Un worker demande le modèle global courant."""
        cid = request.client_id

        # Marquer le client comme vivant
        with self.clients_lock:
            if cid in self.clients:
                self.clients[cid]["alive"] = True

        # Si l'entraînement est terminé
        if self.training_done:
            return federated_pb2.ModelWeights(
                weights=b"", round_id=-2, checksum=""
            )

        # Si pas encore commencé, attendre
        if not self.is_training:
            return federated_pb2.ModelWeights(
                weights=b"", round_id=-1, checksum=""
            )

        with self.model_lock:
            data     = self.model_bytes
            checksum = compute_checksum(data)

        return federated_pb2.ModelWeights(
            weights=data,
            round_id=self.current_round,
            checksum=checksum,
        )

    def SubmitUpdate(self, request, context):
        """Un worker envoie ses poids après entraînement local."""
        cid = request.client_id

        # Vérifier que c'est le bon round
        if request.round_id != self.current_round:
            return federated_pb2.Ack(
                success=False,
                message=f"Round {request.round_id} expiré (courant={self.current_round})"
            )

        # Vérifier le checksum (intégrité)
        if not verify_checksum(request.weights, request.checksum):
            log.warning(f"Checksum invalide de {cid} — rejeté")
            return federated_pb2.Ack(success=False, message="Checksum invalide")

        # Stocker la mise à jour
        with self.updates_lock:
            self.round_updates[cid] = request
            self.client_metrics[cid] = {
                "loss":      request.train_loss,
                "acc":       request.train_acc,
                "responded": True,
            }

        log.info(f"Mise à jour reçue de {cid} — "
                 f"loss={request.train_loss:.4f}, acc={request.train_acc:.4f}")

        # Notifier si on a assez de réponses
        needed = self._needed_responses()
        if len(self.round_updates) >= needed:
            self.updates_event.set()

        return federated_pb2.Ack(success=True, message="Poids reçus")

    def GetTrainingStatus(self, request, context):
        """Retourne l'état complet du système (pour le dashboard)."""
        with self.clients_lock:
            client_statuses = []
            for cid, info in self.clients.items():
                metrics = self.client_metrics.get(cid, {})
                client_statuses.append(federated_pb2.ClientStatus(
                    client_id=cid,
                    is_alive=info.get("alive", False),
                    has_responded=metrics.get("responded", False),
                    last_loss=metrics.get("loss", 0.0),
                    last_acc=metrics.get("acc", 0.0),
                    num_samples=info.get("num_samples", 0),
                ))

        return federated_pb2.TrainingStatus(
            current_round=self.current_round,
            global_accuracy=self.global_accuracy,
            global_loss=self.global_loss,
            active_clients=sum(1 for c in self.clients.values() if c.get("alive")),
            is_training=self.is_training,
            total_rounds=self.max_rounds,
            clients=client_statuses,
        )

    # ── Logique FL ───────────────────────────────────────────────

    def _needed_responses(self) -> int:
        """Calcule le nombre minimum de réponses attendues."""
        n_clients = len(self.clients)
        needed = max(self.min_clients, int(n_clients * self.fraction_fit))
        return min(needed, n_clients)

    def _fedavg(self, updates: dict) -> None:
        """
        Algorithme FedAvg : moyenne pondérée des poids locaux.
        w_global = Σ (n_k / N) * w_k
        """
        total_samples = sum(u.num_samples for u in updates.values())
        if total_samples == 0:
            log.warning("FedAvg : aucun sample reçu")
            return

        # Charger tous les modèles locaux
        local_models = {}
        for cid, update in updates.items():
            m = get_model(self.cfg)
            bytes_to_weights(m, update.weights)
            local_models[cid] = (m, update.num_samples)

        # Calculer la moyenne pondérée paramètre par paramètre
        global_state = {}
        first_state  = list(local_models.values())[0][0].state_dict()

        for key in first_state.keys():
            weighted_sum = torch.zeros_like(first_state[key], dtype=torch.float32)
            for cid, (m, n_k) in local_models.items():
                weight = n_k / total_samples
                weighted_sum += m.state_dict()[key].float() * weight
            global_state[key] = weighted_sum

        # Mettre à jour le modèle global
        with self.model_lock:
            self.model.load_state_dict(global_state)
            self.model_bytes = weights_to_bytes(self.model)

        log.info(f"FedAvg : {len(updates)} workers, {total_samples} samples total")

    def _evaluate(self) -> tuple:
        """Évalue le modèle global sur le dataset de validation."""
        self.model.eval()
        criterion  = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct    = 0
        total      = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                outputs = self.model(images)
                loss    = criterion(outputs, labels)
                total_loss += loss.item() * len(labels)
                preds   = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += len(labels)

        acc  = correct / total
        loss = total_loss / total
        return acc, loss

    def _reset_round_state(self):
        """Réinitialise l'état pour un nouveau round."""
        with self.updates_lock:
            self.round_updates = {}
        # Réinitialiser les flags responded
        for cid in self.client_metrics:
            self.client_metrics[cid]["responded"] = False
        self.updates_event.clear()

    def run_training(self):
        """
        Boucle principale de l'entraînement fédéré.
        Appelée dans un thread séparé.
        """
        # Attendre que assez de clients soient connectés
        log.info(f"En attente de {self.min_clients} workers...")
        while len(self.clients) < self.min_clients:
            time.sleep(1)

        log.info(f"{len(self.clients)} workers connectés — démarrage de l'entraînement")
        self.is_training = True

        for rnd in range(1, self.max_rounds + 1):
            self.current_round = rnd
            self._reset_round_state()

            log.info(f"═══ Round {rnd}/{self.max_rounds} ═══")

            # Attendre les mises à jour (avec timeout)
            needed = self._needed_responses()
            log.info(f"Attente de {needed} workers (timeout={self.timeout}s)...")

            deadline = time.time() + self.timeout
            while len(self.round_updates) < needed:
                remaining = deadline - time.time()
                if remaining <= 0:
                    log.warning(f"Timeout ! {len(self.round_updates)}/{needed} réponses reçues")
                    break
                self.updates_event.wait(timeout=min(remaining, 2.0))
                self.updates_event.clear()

            updates = dict(self.round_updates)
            if len(updates) == 0:
                log.warning(f"Round {rnd} ignoré — aucune réponse")
                continue

            # FedAvg
            self._fedavg(updates)

            # Évaluation
            acc, loss = self._evaluate()
            self.global_accuracy = acc
            self.global_loss     = loss

            self.metrics_history.append({
                "round": rnd,
                "accuracy": acc,
                "loss": loss,
                "num_updates": len(updates),
                "timestamp": datetime.now().isoformat(),
            })

            log.info(f"Round {rnd} — Accuracy={acc:.4f} ({acc*100:.2f}%), "
                     f"Loss={loss:.4f}, Updates={len(updates)}/{len(self.clients)}")

            # Sauvegarder dans Redis
            if self.redis:
                self.redis.rpush("metrics", str({
                    "round": rnd, "accuracy": acc, "loss": loss
                }))

            # Critère de convergence
            if acc >= 0.95:
                log.info(f"🎉 Convergence atteinte au round {rnd} ! Accuracy={acc*100:.2f}%")
                break

        self.is_training   = False
        self.training_done = True
        log.info("Entraînement terminé.")


def serve():
    cfg  = load_config()
    host = cfg["server"]["host"]
    port = cfg["server"]["port"]

    server_obj = FederatedServer(cfg)

    # Lancer l'entraînement dans un thread séparé
    train_thread = threading.Thread(target=server_obj.run_training, daemon=True)
    train_thread.start()

    # Démarrer le serveur gRPC
    grpc_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=20),
        options=[
            ('grpc.max_send_message_length',    50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    )
    federated_pb2_grpc.add_FederatedServiceServicer_to_server(server_obj, grpc_server)
    grpc_server.add_insecure_port(f"{host}:{port}")
    grpc_server.start()

    log.info(f"Serveur gRPC démarré sur {host}:{port}")

    try:
        grpc_server.wait_for_termination()
    except KeyboardInterrupt:
        log.info("Arrêt du serveur...")
        grpc_server.stop(0)


if __name__ == "__main__":
    serve()
