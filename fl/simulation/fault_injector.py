"""
Injecteur de pannes — scripts de simulation pour les scénarios de démonstration.
Utilisé pour le chaos testing (Sprint 3).
"""

import subprocess
import time
import random
import logging
import os

log = logging.getLogger(__name__)


def kill_worker(worker_id: int, compose_project: str = "fl"):
    """Tue un conteneur worker Docker."""
    container = f"{compose_project}-worker-{worker_id}-1"
    try:
        subprocess.run(["docker", "kill", container], check=True, capture_output=True)
        log.info(f"Worker {worker_id} tué ({container})")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Impossible de tuer {container} : {e.stderr.decode()}")
        return False


def restart_worker(worker_id: int, compose_project: str = "fl"):
    """Redémarre un conteneur worker Docker."""
    container = f"{compose_project}-worker-{worker_id}-1"
    try:
        subprocess.run(["docker", "start", container], check=True, capture_output=True)
        log.info(f"Worker {worker_id} redémarré")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Impossible de redémarrer {container} : {e.stderr}")
        return False


def add_network_latency(worker_id: int, delay_ms: int = 500, jitter_ms: int = 100):
    """
    Ajoute de la latence réseau sur un worker via tc netem.
    Nécessite NET_ADMIN capability dans Docker.
    """
    container = f"fl-worker-{worker_id}-1"
    cmd = (f"tc qdisc add dev eth0 root netem "
           f"delay {delay_ms}ms {jitter_ms}ms distribution normal")
    try:
        subprocess.run(["docker", "exec", container, "sh", "-c", cmd],
                       check=True, capture_output=True)
        log.info(f"Latence {delay_ms}ms ajoutée sur worker-{worker_id}")
        return True
    except Exception as e:
        log.warning(f"tc netem non disponible : {e}")
        return False


def remove_network_latency(worker_id: int):
    """Supprime la latence réseau ajoutée."""
    container = f"fl-worker-{worker_id}-1"
    cmd = "tc qdisc del dev eth0 root"
    try:
        subprocess.run(["docker", "exec", container, "sh", "-c", cmd],
                       check=True, capture_output=True)
        log.info(f"Latence supprimée sur worker-{worker_id}")
    except Exception:
        pass


def network_partition(worker_ids: list, server_ip: str = "172.18.0.2"):
    """
    Simule un partitionnement réseau en bloquant les connexions
    entre les workers indiqués et le serveur.
    """
    for wid in worker_ids:
        container = f"fl-worker-{wid}-1"
        cmd = f"iptables -A OUTPUT -d {server_ip} -j DROP"
        try:
            subprocess.run(["docker", "exec", "--privileged", container, "sh", "-c", cmd],
                           check=True, capture_output=True)
            log.info(f"Partition réseau activée sur worker-{wid}")
        except Exception as e:
            log.warning(f"iptables non disponible sur worker-{wid} : {e}")


def heal_network_partition(worker_ids: list, server_ip: str = "172.18.0.2"):
    """Rétablit la connexion réseau après une partition."""
    for wid in worker_ids:
        container = f"fl-worker-{wid}-1"
        cmd = f"iptables -D OUTPUT -d {server_ip} -j DROP"
        try:
            subprocess.run(["docker", "exec", "--privileged", container, "sh", "-c", cmd],
                           check=True, capture_output=True)
            log.info(f"Partition réseau guérie sur worker-{wid}")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# SCÉNARIOS DE DÉMO (Sprint 5)
# ─────────────────────────────────────────────────────────────

def scenario_normal():
    """Scénario 1 : Convergence normale sans panne."""
    print("=== Scénario 1 : Convergence normale ===")
    print("Lancement : docker compose up --scale worker=5")
    print("Dashboard  : http://localhost:8501")
    print("Attendre la convergence (accuracy ≥ 95%)...")


def scenario_crash(worker_id: int = 2, delay_s: int = 15):
    """Scénario 2 : Kill d'un worker pendant l'entraînement."""
    print(f"=== Scénario 2 : Crash du worker-{worker_id} dans {delay_s}s ===")
    time.sleep(delay_s)
    success = kill_worker(worker_id)
    if success:
        print(f"Worker-{worker_id} tué — observer le dashboard (nœud rouge)")
        print(f"Redémarrage dans 30s...")
        time.sleep(30)
        restart_worker(worker_id)
        print(f"Worker-{worker_id} redémarré")


def scenario_byzantine(worker_id: int = 3):
    """Scénario 3 : Simulation d'un worker byzantin via variable d'env."""
    print(f"=== Scénario 3 : Worker-{worker_id} devient byzantin ===")
    print(f"Relancer avec : BYZANTINE_WORKERS={worker_id} docker compose up")
    print("Observer la détection dans les logs du serveur")


if __name__ == "__main__":
    import sys
    scenarios = {"1": scenario_normal, "2": scenario_crash, "3": scenario_byzantine}
    choice = sys.argv[1] if len(sys.argv) > 1 else "1"
    scenarios.get(choice, scenario_normal)()
