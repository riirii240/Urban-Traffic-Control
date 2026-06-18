#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Script de lancement — Federated Learning
# Usage : bash launch.sh [nb_workers]
# Exemple : bash launch.sh 5
# ─────────────────────────────────────────────────────────────

NB_WORKERS=${1:-5}

echo "╔══════════════════════════════════════════════╗"
echo "║   Federated Learning — Lancement             ║"
echo "║   Workers : $NB_WORKERS                            ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Arrêter les conteneurs existants
docker compose down 2>/dev/null

# Construire les images
echo "🔨 Construction des images Docker..."
docker compose build

# Lancer Redis + Serveur + Dashboard
echo "🚀 Démarrage Redis, Serveur et Dashboard..."
docker compose up -d redis server dashboard

# Attendre que le serveur soit prêt
echo "⏳ Attente du serveur gRPC..."
sleep 8

# Lancer les workers avec des IDs uniques
echo "👷 Lancement de $NB_WORKERS workers..."
for i in $(seq 0 $((NB_WORKERS - 1))); do
    WORKER_ID=$i docker compose run -d --no-deps \
        -e WORKER_ID=$i \
        --name "fl-worker-$i" \
        worker
    echo "   ✅ Worker-$i lancé"
done

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ✅ Système démarré !                        ║"
echo "║  📊 Dashboard : http://localhost:8501        ║"
echo "║  📡 Serveur   : localhost:50051              ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Pour voir les logs du serveur :"
echo "  docker logs fl-server -f"
echo ""
echo "Pour arrêter tout :"
echo "  docker compose down"
