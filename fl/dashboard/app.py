"""
Dashboard Streamlit — Federated Learning Monitor
Se connecte au serveur via gRPC et affiche les métriques en temps réel.
"""

import sys, os, time, math, random, logging
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

sys.path.insert(0, "/app")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FL Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #080c18; }

section[data-testid="stSidebar"] {
    background: #0b0f1e !important;
    border-right: 1px solid #19253d;
}
section[data-testid="stSidebar"] label {
    font-size: 11px !important; text-transform: uppercase;
    letter-spacing: .08em; color: #3a5a8a !important;
    font-family: 'JetBrains Mono', monospace !important;
}

[data-testid="metric-container"] {
    background: #0f1625; border: 1px solid #19253d;
    border-radius: 12px; padding: 14px 18px !important;
}
[data-testid="metric-container"] label {
    color: #3a5a8a !important; font-size: 10px !important;
    text-transform: uppercase; letter-spacing: .12em;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #dce8ff !important; font-size: 1.9rem !important;
    font-weight: 800 !important; font-family: 'Syne', sans-serif !important;
}

.stButton > button {
    background: #0f2044 !important; color: #7aaaff !important;
    border: 1px solid #1a3a6b !important; border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important;
}
.stButton > button:hover {
    background: #1a3a6b !important; color: #dce8ff !important;
}

.sec { font-family: 'JetBrains Mono', monospace; font-size: 10px;
       text-transform: uppercase; letter-spacing: .15em;
       color: #2a4a7a; margin-bottom: 8px;
       border-bottom: 1px solid #141e33; padding-bottom: 5px; }

.dot-live { display:inline-flex; align-items:center; gap:6px;
  background:#061a0e; border:1px solid #1a4a2a; border-radius:20px;
  padding:4px 12px; font-family:'JetBrains Mono',monospace;
  font-size:11px; color:#22dd77; }
.dot-live::before { content:''; width:6px; height:6px;
  background:#22dd77; border-radius:50%; animation:blink 1.4s infinite; }

.dot-off { display:inline-flex; align-items:center; gap:6px;
  background:#1a0606; border:1px solid #4a1414; border-radius:20px;
  padding:4px 12px; font-family:'JetBrains Mono',monospace;
  font-size:11px; color:#ff4444; }
.dot-off::before { content:''; width:6px; height:6px;
  background:#ff4444; border-radius:50%; }

@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.25} }

.pgbar { height:6px; border-radius:3px; background:#141e33; overflow:hidden; margin:6px 0 16px; }
.pgbar-fill { height:100%; border-radius:3px;
  background: linear-gradient(90deg,#1a6bff,#22dd77); transition:width .5s; }

.wcard { background:#0b1424; border:1px solid #18253a;
  border-radius:10px; padding:11px 13px;
  font-family:'JetBrains Mono',monospace; font-size:11px; }
.wcard.ok   { border-color:#1a4a2a; }
.wcard.dead { border-color:#4a1414; background:#0b0e14; }
.wcard.byz  { border-color:#4a3000; background:#0c0d08; }
.wname { font-size:12px; font-weight:700; color:#8abeff; margin-bottom:5px; }
.wstat { color:#3a5a8a; margin-top:2px; }
.wstat span { color:#b8d0f0; }

.logbox { background:#050912; border:1px solid #141e33; border-radius:9px;
  padding:11px 14px; font-family:'JetBrains Mono',monospace; font-size:11px;
  max-height:230px; overflow-y:auto; color:#3a5a8a; }
.log-ok   { color:#22dd77; }
.log-warn { color:#ffaa33; }
.log-err  { color:#ff4444; }
.log-info { color:#44aaff; }

.infochip { background:#0f1625; border:1px solid #19253d; border-radius:9px;
  padding:9px 15px; font-family:'JetBrains Mono',monospace; }
.infochip .k { font-size:10px; color:#2a4a7a; text-transform:uppercase; letter-spacing:.08em; }
.infochip .v { font-size:13px; color:#b8d0f0; font-weight:600; margin-top:1px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "history_rounds": [], "history_acc": [], "history_loss": [],
        "logs": [], "training_active": False,
        "sim_round": 0, "num_clients": 5, "num_rounds": 20,
        "crash_prob": 0.0, "byzantine_ids": [],
        "server_host": "server", "server_port": 50051,
        "aggregator": "FedAvg", "local_epochs": 3,
        "dataset": "MNIST IID",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def add_log(msg, level="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append({"ts": ts, "msg": msg, "level": level})
    if len(st.session_state.logs) > 120:
        st.session_state.logs = st.session_state.logs[-120:]

# ─────────────────────────────────────────────────────────────
# CONNEXION gRPC
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_grpc_stub(host, port):
    try:
        import grpc
        from server.proto import federated_pb2_grpc
        channel = grpc.insecure_channel(
            f"{host}:{port}",
            options=[('grpc.connect_timeout_ms', 1500)]
        )
        grpc.channel_ready_future(channel).result(timeout=2)
        return federated_pb2_grpc.FederatedServiceStub(channel), True
    except Exception:
        return None, False


def fetch_real_status(host, port):
    """Tente de récupérer l'état réel via gRPC."""
    try:
        import grpc
        from server.proto import federated_pb2, federated_pb2_grpc
        stub, ok = get_grpc_stub(host, port)
        if not ok or stub is None:
            return None
        s = stub.GetTrainingStatus(federated_pb2.Empty(), timeout=2)
        clients = [{
            "id": c.client_id, "alive": c.is_alive,
            "responded": c.has_responded,
            "loss": c.last_loss, "acc": c.last_acc,
            "samples": c.num_samples, "byzantine": False,
        } for c in s.clients]
        return {
            "connected": True,
            "round": s.current_round,
            "total_rounds": s.total_rounds or st.session_state.num_rounds,
            "accuracy": s.global_accuracy,
            "loss": s.global_loss,
            "active_clients": s.active_clients,
            "is_training": s.is_training,
            "clients": clients,
        }
    except Exception:
        return None


def simulate_status():
    """Simulation réaliste quand le serveur n'est pas accessible."""
    r   = st.session_state.sim_round
    tot = st.session_state.num_rounds
    nc  = st.session_state.num_clients

    acc  = min(0.97, 0.42 + 0.55 * (1 - math.exp(-r / 7))) + random.uniform(-.004, .004) if r > 0 else 0.10
    loss = max(0.03, 2.1 * math.exp(-r / 5.5) + random.uniform(-.008, .008)) if r > 0 else 2.30

    byz  = st.session_state.byzantine_ids
    cp   = st.session_state.crash_prob
    rng  = random.Random(r * 100)
    clients = []
    for i in range(nc):
        alive   = rng.random() > cp
        byz_f   = i in byz
        clients.append({
            "id": f"worker-{i}", "alive": alive and not byz_f,
            "responded": alive and rng.random() > .06,
            "loss": loss + rng.uniform(-.05, .18),
            "acc":  acc  - rng.uniform(0, .08),
            "samples": 12000 // nc + rng.randint(-400, 400),
            "byzantine": byz_f,
        })
    return {
        "connected": False, "round": r, "total_rounds": tot,
        "accuracy": acc, "loss": loss,
        "active_clients": sum(1 for c in clients if c["alive"]),
        "is_training": st.session_state.training_active,
        "clients": clients,
    }


def get_status():
    real = fetch_real_status(st.session_state.server_host, st.session_state.server_port)
    return real if real is not None else simulate_status()

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 FL Monitor")

    st.markdown('<div class="sec">Serveur gRPC</div>', unsafe_allow_html=True)
    st.session_state.server_host = st.text_input("Host",  value=st.session_state.server_host, label_visibility="visible")
    st.session_state.server_port = st.number_input("Port", value=st.session_state.server_port, step=1, label_visibility="visible")

    st.markdown('<div class="sec" style="margin-top:14px">Configuration FL</div>', unsafe_allow_html=True)
    st.session_state.dataset    = st.selectbox("Dataset", ["MNIST IID", "MNIST Non-IID (α=0.5)", "MNIST Non-IID (α=0.1)"])
    st.session_state.aggregator = st.selectbox("Agrégateur", ["FedAvg", "FedProx", "FedAvg + DP"])
    st.session_state.num_clients  = st.slider("Workers",         2, 20, st.session_state.num_clients)
    st.session_state.num_rounds   = st.slider("Rounds max",      5, 100, st.session_state.num_rounds)
    st.session_state.local_epochs = st.slider("Époques locales", 1, 10,  st.session_state.local_epochs)

    st.markdown('<div class="sec" style="margin-top:14px">Simulation pannes</div>', unsafe_allow_html=True)
    crash_pct = st.slider("Crash (%)", 0, 60, int(st.session_state.crash_prob * 100))
    st.session_state.crash_prob = crash_pct / 100
    byz_raw = st.text_input("Workers byzantins (ex: 0,2)", value="")
    if byz_raw.strip():
        try:    st.session_state.byzantine_ids = [int(x) for x in byz_raw.split(",")]
        except: st.session_state.byzantine_ids = []

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶"):
            if not st.session_state.training_active:
                st.session_state.update({
                    "training_active": True, "sim_round": 0,
                    "history_rounds": [], "history_acc": [], "history_loss": [],
                })
                add_log(f"Démarrage — {st.session_state.num_clients} workers, {st.session_state.num_rounds} rounds", "ok")
    with c2:
        if st.button("⏸"):
            st.session_state.training_active = False
            add_log("Pause", "warn")
    with c3:
        if st.button("⏹"):
            st.session_state.update({
                "training_active": False, "sim_round": 0,
                "history_rounds": [], "history_acc": [], "history_loss": [],
            })
            add_log("Arrêt et réinitialisation", "err")

    st.markdown("---")
    st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#1a3a6a">Systèmes Distribués 2025–2026<br>Apprentissage Fédéré</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# AVANCER LA SIMULATION
# ─────────────────────────────────────────────────────────────
if st.session_state.training_active:
    if st.session_state.sim_round < st.session_state.num_rounds:
        st.session_state.sim_round += 1
        r = st.session_state.sim_round
        nc = st.session_state.num_clients
        cp = st.session_state.crash_prob
        byz = st.session_state.byzantine_ids
        responded = max(1, round(nc * (1 - cp)))
        if byz:
            add_log(f"Round {r} — Byzantine détecté sur worker-{byz[0]}, exclu par Krum", "warn")
        elif cp > 0 and random.random() < cp:
            crashed = random.randint(0, nc - 1)
            add_log(f"Round {r} — worker-{crashed} crash détecté, timeout déclenché", "warn")
        else:
            add_log(f"Round {r}/{st.session_state.num_rounds} — {responded}/{nc} workers agrégés ({st.session_state.aggregator})", "info")
    else:
        st.session_state.training_active = False
        add_log(f"Convergence atteinte en {st.session_state.num_rounds} rounds ✓", "ok")

# Récupérer l'état
status = get_status()

# Mettre à jour l'historique
r = status["round"]
if r > 0 and (not st.session_state.history_rounds or st.session_state.history_rounds[-1] != r):
    st.session_state.history_rounds.append(r)
    st.session_state.history_acc.append(round(status["accuracy"] * 100, 2))
    st.session_state.history_loss.append(round(status["loss"], 4))

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
hc1, hc2 = st.columns([3, 1])
with hc1:
    st.markdown(
        '<h1 style="font-family:Syne,sans-serif;font-size:1.75rem;font-weight:800;'
        'color:#dce8ff;margin:0;letter-spacing:-.01em">'
        '🧠 Federated Learning — Contrôle & Monitoring</h1>',
        unsafe_allow_html=True)
with hc2:
    st.markdown("<br>", unsafe_allow_html=True)
    is_live = status["is_training"] or st.session_state.training_active
    conn    = "CONNECTÉ" if status["connected"] else "SIMULATION"
    if is_live:
        st.markdown(f'<div class="dot-live">LIVE · {conn}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="dot-off">ARRÊTÉ</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MÉTRIQUES
# ─────────────────────────────────────────────────────────────
r_cur = status["round"]
r_tot = status["total_rounds"]
pct   = (r_cur / r_tot * 100) if r_tot > 0 else 0
acc   = status["accuracy"] * 100
loss  = status["loss"]
actv  = status["active_clients"]
tot_c = len(status["clients"])

hist_acc  = st.session_state.history_acc
hist_loss = st.session_state.history_loss

m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Round courant",    f"{r_cur}/{r_tot}")
with m2: st.metric("Accuracy globale", f"{acc:.1f}%",
                   delta=f"+{hist_acc[-1]-hist_acc[-2]:.1f}%" if len(hist_acc) >= 2 else None)
with m3: st.metric("Loss globale",     f"{loss:.4f}",
                   delta=f"{hist_loss[-1]-hist_loss[-2]:.4f}" if len(hist_loss) >= 2 else None,
                   delta_color="inverse")
with m4: st.metric("Workers actifs",   f"{actv}/{tot_c}")

# Barre de progression
st.markdown(f"""
<div style="margin:2px 0 16px">
  <div style="display:flex;justify-content:space-between;
    font-family:JetBrains Mono,monospace;font-size:10px;color:#2a4a7a;margin-bottom:3px">
    <span>Progression entraînement</span><span>{pct:.1f}%</span>
  </div>
  <div class="pgbar"><div class="pgbar-fill" style="width:{pct}%"></div></div>
</div>
""", unsafe_allow_html=True)

# Info chips
ic1, ic2, ic3, ic4 = st.columns(4)
for col, k, v in [
    (ic1, "Modèle", "MLP 784→128→64→10"),
    (ic2, "Dataset", st.session_state.dataset),
    (ic3, "Agrégateur", st.session_state.aggregator),
    (ic4, "Workers attendus", str(st.session_state.num_clients)),
]:
    col.markdown(f'<div class="infochip"><div class="k">{k}</div><div class="v">{v}</div></div>',
                 unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TOPOLOGIE + COURBES
# ─────────────────────────────────────────────────────────────
col_topo, col_curves = st.columns([1, 1], gap="medium")

# ── Topologie ──
with col_topo:
    st.markdown('<div class="sec">Schéma dynamique serveur-clients</div>', unsafe_allow_html=True)
    clients = status["clients"]
    n  = len(clients)
    R  = 1.65
    ex, ey, nx, ny = [], [], [], []
    nc_txt, nc_col, nc_sz, nc_sym = [], [], [], []

    # Serveur
    nx.append(0); ny.append(0)
    nc_txt.append("SERVEUR"); nc_col.append("#1a6bff")
    nc_sz.append(30); nc_sym.append("square")

    for i, c in enumerate(clients):
        a  = 2 * math.pi * i / n - math.pi / 2
        wx = R * math.cos(a); wy = R * math.sin(a)
        nx.append(wx); ny.append(wy)
        nc_txt.append(c["id"])
        ex += [0, wx, None]; ey += [0, wy, None]
        if c["byzantine"]:
            nc_col.append("#ff8800"); nc_sz.append(17)
        elif not c["alive"]:
            nc_col.append("#ff3333"); nc_sz.append(14)
        elif c["responded"]:
            nc_col.append("#22dd77"); nc_sz.append(17)
        else:
            nc_col.append("#ffcc00"); nc_sz.append(15)
        nc_sym.append("circle")

    htexts = ["<b>Serveur</b><br>Round: " + str(r_cur)] + [
        f"<b>{c['id']}</b><br>"
        f"{'⚠️ Byzantine' if c['byzantine'] else ('✅ Actif' if c['alive'] else '❌ Crash')}<br>"
        f"Loss: {c['loss']:.4f}<br>Samples: {c['samples']}"
        for c in clients
    ]

    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
        line=dict(color="#19253d", width=1.2), hoverinfo="none", showlegend=False))
    fig_t.add_trace(go.Scatter(x=nx, y=ny, mode="markers+text",
        marker=dict(color=nc_col, size=nc_sz, symbol=nc_sym,
                    line=dict(color="#080c18", width=2)),
        text=nc_txt,
        textposition=["middle center"] + ["top center"] * n,
        textfont=dict(family="JetBrains Mono", size=9, color="#5a7aad"),
        hovertext=htexts, hoverinfo="text", showlegend=False))

    for sym, col, lbl in [
        ("circle","#22dd77","Actif"), ("circle","#ffcc00","En attente"),
        ("circle","#ff3333","Crash"), ("circle","#ff8800","Byzantine"),
        ("square","#1a6bff","Serveur"),
    ]:
        fig_t.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
            marker=dict(symbol=sym, size=9, color=col), name=lbl, showlegend=True))

    fig_t.update_layout(
        paper_bgcolor="#0b1020", plot_bgcolor="#0b1020",
        xaxis=dict(visible=False, range=[-2.3,2.3]),
        yaxis=dict(visible=False, range=[-2.3,2.3], scaleanchor="x"),
        margin=dict(l=0,r=0,t=8,b=0), height=370,
        legend=dict(font=dict(family="JetBrains Mono",size=10,color="#4a6a9a"),
                    bgcolor="#080c18", bordercolor="#19253d", borderwidth=1,
                    x=0.76, y=0.02),
    )
    st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar": False})

# ── Courbes ──
with col_curves:
    st.markdown('<div class="sec">Convergence — Accuracy & Loss</div>', unsafe_allow_html=True)
    rnds  = st.session_state.history_rounds
    accs  = st.session_state.history_acc
    losss = st.session_state.history_loss

    BG = "#0b1020"

    def empty_chart(msg):
        f = go.Figure()
        f.add_annotation(text=msg, xref="paper", yref="paper", x=.5, y=.5,
                         showarrow=False, font=dict(family="JetBrains Mono",size=12,color="#1a3a6a"))
        f.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=168,
                        margin=dict(l=0,r=0,t=0,b=0),
                        xaxis=dict(visible=False), yaxis=dict(visible=False))
        return f

    if len(rnds) < 2:
        st.plotly_chart(empty_chart("En attente du démarrage…"), use_container_width=True,
                        config={"displayModeBar": False})
        st.plotly_chart(empty_chart("Appuyez sur ▶ Start dans la sidebar"),
                        use_container_width=True, config={"displayModeBar": False})
    else:
        # Accuracy
        fa = go.Figure()
        fa.add_trace(go.Scatter(x=rnds, y=accs, mode="lines+markers",
            line=dict(color="#1a6bff", width=2.5, shape="spline"),
            marker=dict(size=5), fill="tozeroy",
            fillcolor="rgba(26,107,255,0.07)",
            hovertemplate="Round %{x}<br>Accuracy: %{y:.2f}%<extra></extra>"))
        fa.add_hline(y=95, line_dash="dot", line_color="#22dd77",
                     annotation_text="Cible 95%",
                     annotation_font=dict(family="JetBrains Mono",size=9,color="#22dd77"))
        fa.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=168,
                         margin=dict(l=40,r=10,t=6,b=28), showlegend=False,
                         xaxis=dict(color="#2a4a7a", gridcolor="#111b2e",
                                    tickfont=dict(family="JetBrains Mono",size=9)),
                         yaxis=dict(color="#2a4a7a", gridcolor="#111b2e", range=[0,100],
                                    title="Accuracy (%)",
                                    title_font=dict(size=9,family="JetBrains Mono"),
                                    tickfont=dict(family="JetBrains Mono",size=9)))
        st.plotly_chart(fa, use_container_width=True, config={"displayModeBar": False})

        # Loss
        fl = go.Figure()
        fl.add_trace(go.Scatter(x=rnds, y=losss, mode="lines+markers",
            line=dict(color="#ff4444", width=2.5, shape="spline"),
            marker=dict(size=5), fill="tozeroy",
            fillcolor="rgba(255,68,68,0.06)",
            hovertemplate="Round %{x}<br>Loss: %{y:.4f}<extra></extra>"))
        fl.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=168,
                         margin=dict(l=40,r=10,t=6,b=28), showlegend=False,
                         xaxis=dict(color="#2a4a7a", gridcolor="#111b2e",
                                    title="Round",
                                    title_font=dict(size=9,family="JetBrains Mono"),
                                    tickfont=dict(family="JetBrains Mono",size=9)),
                         yaxis=dict(color="#2a4a7a", gridcolor="#111b2e",
                                    title="Loss (CE)",
                                    title_font=dict(size=9,family="JetBrains Mono"),
                                    tickfont=dict(family="JetBrains Mono",size=9)))
        st.plotly_chart(fl, use_container_width=True, config={"displayModeBar": False})

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# WORKERS + LOGS
# ─────────────────────────────────────────────────────────────
cw, cl = st.columns([1.2, 1], gap="medium")

with cw:
    st.markdown('<div class="sec">État des workers</div>', unsafe_allow_html=True)
    cols_per_row = 3
    client_rows = [clients[i:i+cols_per_row] for i in range(0, len(clients), cols_per_row)]
    for row in client_rows:
        rcols = st.columns(len(row))
        for col, c in zip(rcols, row):
            if c["byzantine"]:
                cls = "byz"; icon = "⚠️"; stxt = "BYZANTINE"; scol = "#ff8800"
            elif not c["alive"]:
                cls = "dead"; icon = "❌"; stxt = "CRASH"; scol = "#ff4444"
            elif c["responded"]:
                cls = "ok"; icon = "✅"; stxt = "OK"; scol = "#22dd77"
            else:
                cls = ""; icon = "⏳"; stxt = "ATTENTE"; scol = "#ffcc00"
            col.markdown(
                f'<div class="wcard {cls}">'
                f'<div class="wname">{icon} {c["id"]}</div>'
                f'<div class="wstat">Statut: <span style="color:{scol}">{stxt}</span></div>'
                f'<div class="wstat">Loss: <span>{c["loss"]:.4f}</span></div>'
                f'<div class="wstat">Samples: <span>{c["samples"]:,}</span></div>'
                f'</div>', unsafe_allow_html=True)

with cl:
    st.markdown('<div class="sec">Journal des événements</div>', unsafe_allow_html=True)
    if not st.session_state.logs:
        st.markdown(
            '<div class="logbox"><span style="color:#1a3a6a">Aucun événement…<br>'
            'Appuyez sur ▶ Start pour démarrer.</span></div>',
            unsafe_allow_html=True)
    else:
        lines = "".join(
            f'<div><span style="color:#1a3a6a">[{e["ts"]}]</span> '
            f'<span class="log-{e["level"]}">{e["msg"]}</span></div>'
            for e in reversed(st.session_state.logs[-50:])
        )
        st.markdown(f'<div class="logbox">{lines}</div>', unsafe_allow_html=True)
    if st.button("🗑 Vider"):
        st.session_state.logs = []
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# GRAPHIQUES BAS DE PAGE
# ─────────────────────────────────────────────────────────────
cb, ch = st.columns(2, gap="medium")

with cb:
    st.markdown('<div class="sec">Loss locale par worker</div>', unsafe_allow_html=True)
    wids = [c["id"] for c in clients]
    wloss = [c["loss"] for c in clients]
    wcols = ["#ff8800" if c["byzantine"] else "#ff4444" if not c["alive"] else "#1a6bff"
             for c in clients]
    fb = go.Figure(go.Bar(x=wids, y=wloss, marker_color=wcols,
        text=[f"{l:.3f}" for l in wloss], textposition="outside",
        textfont=dict(family="JetBrains Mono",size=9,color="#4a6a9a")))
    fb.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=210,
                     margin=dict(l=30,r=10,t=8,b=40),
                     xaxis=dict(color="#2a4a7a",tickfont=dict(family="JetBrains Mono",size=9)),
                     yaxis=dict(color="#2a4a7a",gridcolor="#111b2e",
                                tickfont=dict(family="JetBrains Mono",size=9)),
                     showlegend=False, bargap=0.28)
    st.plotly_chart(fb, use_container_width=True, config={"displayModeBar": False})

with ch:
    st.markdown('<div class="sec">Participation par round (simulation)</div>', unsafe_allow_html=True)
    nshow = max(min(len(rnds), 15), 6)
    nw    = len(clients)
    rng2  = np.random.default_rng(r_cur + 1)
    mat   = (rng2.random((nw, nshow)) > st.session_state.crash_prob).astype(float)
    xlabs = [f"R{x}" for x in (rnds[-nshow:] if len(rnds) >= nshow else list(range(1, nshow+1)))]
    fh = go.Figure(go.Heatmap(
        z=mat, x=xlabs, y=wids,
        colorscale=[[0,"#200808"],[1,"#083a1a"]],
        showscale=False, zmin=0, zmax=1,
        hovertemplate="Worker: %{y}<br>Round: %{x}<br>Participé: %{z}<extra></extra>"))
    fh.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=210,
                     margin=dict(l=80,r=10,t=8,b=28),
                     xaxis=dict(color="#2a4a7a",tickfont=dict(family="JetBrains Mono",size=8)),
                     yaxis=dict(color="#2a4a7a",tickfont=dict(family="JetBrains Mono",size=8)))
    st.plotly_chart(fh, use_container_width=True, config={"displayModeBar": False})

# ─────────────────────────────────────────────────────────────
# AUTO-REFRESH
# ─────────────────────────────────────────────────────────────
if st.session_state.training_active:
    time.sleep(1.5)
    st.rerun()
