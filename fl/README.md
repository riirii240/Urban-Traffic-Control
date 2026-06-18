# 🧠 Federated Learning — Systèmes Distribués 2025–2026

> Rihab Zitouni · Taha Hakim · Houda Moustaine · Aya Boughalem  
> Encadrant : Dr. Nadiri Abdeljalil

---

## 🏗️ Architecture

```
Aggregator Server (gRPC:50051)
    ├── Worker-0  (données privées MNIST)
    ├── Worker-1  (données privées MNIST)
    ├── Worker-2  (données privées MNIST)
    ├── ...
    └── Worker-N
Redis (registry + métriques)
Dashboard Streamlit (http://localhost:8501)
```

**Algorithme** : FedAvg — `w_global = Σ (n_k / N) * w_k`  
**Modèle** : MLP 784→128→64→10 (MNIST)  
**Tolérance aux pannes** : timeout par round, seuil 50% minimum, checksum SHA-256

---

## 🚀 Lancement rapide

### Prérequis
- Docker Desktop (Windows/macOS) ou Docker Engine (Linux)
- Docker Compose v2+

### Démarrage avec 5 workers
```bash
bash launch.sh 5
```

### Ou manuellement
```bash
# Construire et lancer
docker compose build
docker compose up -d redis server dashboard

# Lancer les workers
for i in 0 1 2 3 4; do
  WORKER_ID=$i docker compose run -d --no-deps -e WORKER_ID=$i worker
done
```

**Dashboard** : http://localhost:8501

---

## 📁 Structure du projet

```
federated-learning/
├── server/
│   ├── aggregator.py          # Serveur gRPC + FedAvg
│   └── proto/
│       └── federated.proto    # Contrat de communication
├── client/
│   ├── worker.py              # Entraînement local + gRPC
│   └── model.py               # MLP PyTorch
├── data/
│   └── partition.py           # Partitionnement IID/Non-IID
├── simulation/
│   └── fault_injector.py      # Chaos testing
├── dashboard/
│   └── app.py                 # Streamlit monitoring
├── tests/
│   └── test_fedavg.py         # Tests unitaires
├── docker-compose.yml
├── Dockerfile.server
├── Dockerfile.worker
├── Dockerfile.dashboard
├── config.yaml                # Paramètres centraux
└── launch.sh                  # Script de lancement
```

---

## 🧪 Scénarios de démonstration

### Scénario 1 — Convergence normale
```bash
bash launch.sh 5
# Observer l'accuracy monter jusqu'à ≥95% sur le dashboard
```

### Scénario 2 — Crash en direct
```bash
# Pendant l'entraînement, tuer un worker
docker kill fl-worker-2
# Observer le nœud rouge sur le dashboard
# Le round continue avec les 4 workers restants
```

### Scénario 3 — Attaque Byzantine
```bash
BYZANTINE_WORKERS=3 docker compose run -d -e WORKER_ID=3 -e BYZANTINE_WORKERS=3 worker
# Observer la détection dans les logs du serveur
docker logs fl-server | grep -i byzantine
```

---

## 🧪 Tests unitaires

```bash
# Dans le conteneur serveur
docker exec fl-server pytest tests/ -v

# En local (avec venv activé)
pip install -r requirements.txt
pytest tests/ -v
```

---

## ⚙️ Configuration (`config.yaml`)

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `max_rounds` | 20 | Nombre de rounds max |
| `min_clients` | 2 | Workers min pour démarrer |
| `fraction_fit` | 0.6 | % workers sélectionnés/round |
| `round_timeout` | 60 | Timeout par round (s) |
| `training.epochs` | 3 | Époques locales par round |
| `dataset.iid` | true | IID ou Non-IID (Dirichlet) |

---

## 📊 Métriques surveillées

- Accuracy globale par round (cible ≥ 95%)
- Loss cross-entropy par round
- Taux de participation par round
- État de chaque worker (alive/crash/byzantine)
- Temps de convergence (Time to 90% accuracy)
