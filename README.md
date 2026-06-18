

# 🚦 Urban Traffic Control — Q-Learning Agent

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8-orange)
![Reinforcement Learning](https://img.shields.io/badge/RL-Q--Learning-purple)
![MDP](https://img.shields.io/badge/Model-MDP-green)
![Status](https://img.shields.io/badge/Status-Academic%20Project-lightgrey)

> **Authors:** Houda Moustaine · Taha Hakim · Rihab Zitouni · Aya Boughalem  
> **Supervisor:** Dr. Issam Quaffou — EMSI Marrakech, IAD & SMA, 2025–2026

A **tabular Q-Learning agent** trained from scratch to control a 4-branch urban intersection, modeled as a **Markov Decision Process (MDP)**. The agent learns an adaptive traffic light policy that outperforms a fixed-cycle baseline, especially under asymmetric traffic loads.

---

## ✨ Key Results

| Scenario | Traffic | Q-Learning advantage |
|----------|---------|----------------------|
| SC1 | Balanced λ=(0.4, 0.4, 0.4, 0.4) | Moderate gain — adapts to local surges |
| SC2 | Asymmetric λ=(0.7, 0.7, 0.2, 0.2) | **Significant gain** — allocates more green to NS axis |

- ✅ Convergence reached after ~1000 episodes
- ✅ TD error |δ| → 0, confirming Q-table stability
- ✅ ε reaches floor (~0.05) at episode 650
- ✅ Optimal params: α=0.1, γ=0.95, ε_decay=0.995

---

## 🏗️ MDP Formalization

**State:** `s = (qN, qS, qE, qO, p)` — queue levels (0–4) per direction + traffic light phase (NS / Orange / EO)  
**State space size:** 5⁴ × 3 = **1,875 states**

**Actions:** `{0: maintain phase, 1: change phase}`  
NS → Orange → EO (mandatory intermediate orange phase)

**Reward:**
```
R(s,a) = -Σ qᵢ - 0.5 · 𝟙[phase change]
```
Penalizes total waiting vehicles + unnecessary phase switches.

**Parameters:** γ=0.95 (effective horizon ≈ 20 steps), arrivals ~ Poisson(λᵢ)

---

## 🤖 Agent Architecture

Hybrid **Reactive-Learning** architecture:
- Perceives current state → consults Q-table → selects action via ε-greedy
- Updates Q via Bellman equation after each step
- No prior model of transition dynamics required (model-free)

```
Q(s,a) ← Q(s,a) + α [r + γ · max_a' Q(s',a') - Q(s,a)]
```

---

## 📁 Project Structure

```
urban-traffic-qlearning/
├── Gestion_Trafic_QLearning.ipynb   # Main notebook (Google Colab)
├── traffic_env.py                    # Intersection simulation (OpenAI Gym interface)
├── qlearning_agent.py               # Q-Learning agent (ε-greedy + Bellman update)
├── baseline.py                       # Fixed-cycle baseline policy
├── train.py                          # Training script
├── evaluate.py                       # Evaluation & comparison script
├── outputs/
│   ├── convergence.png              # SC1 convergence curves
│   ├── con2.png                     # SC2 convergence curves
│   ├── comparison.png               # Q-Learning vs Baseline SC1
│   ├── comp2.png                    # Q-Learning vs Baseline SC2
│   ├── sensitivity.png              # α × ε_decay sensitivity heatmap
│   └── image.png                    # Intersection diagram (3 phases)
└── README.md
```

---

## 🚀 Quick Start

### Install dependencies
```bash
pip install numpy pandas matplotlib seaborn
```

### Run in Google Colab
Open `Gestion_Trafic_QLearning.ipynb` directly in Colab — no setup needed.

### Run locally
```bash
# Train the agent
python train.py

# Evaluate and compare with baseline
python evaluate.py
```

---

## ⚙️ Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| α (learning rate) | 0.10 | Stable under Poisson noise; α>0.3 causes oscillations |
| γ (discount) | 0.95 | Effective horizon ≈ 20 steps — one traffic light cycle |
| ε₀ (initial exploration) | 1.00 | Full exploration at start — unknown state space |
| ε_min | 0.05 | Residual exploration for traffic variation adaptation |
| ε_decay | 0.995 | Reaches ε_min at episode ~650 |
| Episodes | 2000 | Sufficient for convergence and policy stabilization |
| Steps/episode | 200 | Simulates ~200 seconds of traffic |

---

## 📊 Sensitivity Analysis

Best combinations from α × ε_decay grid search:

| α | ε_decay | Avg wait (veh/step) |
|---|---------|---------------------|
| 0.1 | 0.995 | **7.82** ✅ |
| 0.4 | 0.999 | 7.36 (unstable on SC2) |
| 0.05 | 0.995 | 8.65 (too slow) |

---

## 🔬 Simulation Scenarios

| Scenario | λN | λS | λE | λO | Description |
|----------|----|----|----|----|-------------|
| SC1 | 0.4 | 0.4 | 0.4 | 0.4 | Balanced traffic |
| SC2 | 0.7 | 0.7 | 0.2 | 0.2 | Rush hour on NS axis |

Arrivals follow independent **Poisson processes** per direction.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Simulation | Custom OpenAI Gym-style environment |
| RL Algorithm | Tabular Q-Learning (from scratch) |
| Visualization | Matplotlib, Seaborn |
| Notebook | Google Colab |

---

## 📚 References

- Sutton & Barto — *Reinforcement Learning: An Introduction*, MIT Press, 2018
- Watkins & Dayan — *Q-Learning*, Machine Learning, 1992
- Wei et al. — *PressLight*, KDD, 2019
- Russell & Norvig — *AI: A Modern Approach*, 4th ed., Pearson, 2020

---

## 📝 Report

Full technical report (MDP formalization, convergence analysis, sensitivity study) available upon request.

---

## 📄 License

Academic project — EMSI Marrakech, IAD & SMA course, 2025–2026.

---

**🔖 Topics GitHub :**
```
reinforcement-learning  q-learning  traffic-control  mdp  python  multi-agent-systems  markov-decision-process  urban-computing  openai-gym
```
