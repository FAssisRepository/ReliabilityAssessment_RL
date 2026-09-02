# =====================================================================
# Inputs (place in INPUT_DIR):
#   training_dynamics_performance_1.txt   -> RL-GNN training dynamics and performance
#   training_dynamics_performance_2.txt   -> SL-GNN training dynamics and performance
#   AC-OPF-1.txt                   -> AC-OPF solution (reference) for one state
#   AC-PF-GNN-PT-1.txt             -> GNN-SL solution for the same state
#   AC-PF-GNN-RL-1.txt             -> GNN-RL solution for the same state
#
# Requires: numpy, matplotlib
# =====================================================================
import os, re
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
import matplotlib as mpl
from matplotlib.ticker import LogLocator
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24
})

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams["mathtext.fontset"] = "stix"  # keeps math consistent with serif fonts

INPUT_DIR  = os.path.dirname(os.path.abspath(__file__))   # adjust if needed
OUTPUT_DIR = INPUT_DIR
DPI = 160
SMOOTH_WIN = 25   # moving-average window (in logged points; 1 point = 128 episodes)

plt.rcParams.update({
    "font.size": 16, "axes.grid": True, "grid.alpha": 0.1,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.constrained_layout.use": True,
})
# policy colors
C_OPF, C_SP, C_SL, C_RL = "#1f77b4", "#9467bd", "#c67593", "#2ca02c"
# constraint-family colors (V, QG, CL, PG)
C_V, C_QG, C_CL, C_PG = "#444444", "#1f77b4", "#2ca02c", "#ff7f0e"

def _read(path):
    with open(path, encoding="latin1") as f:
        return f.read().replace("\r", "")

def smooth(y, w=SMOOTH_WIN):
    """Centered moving average with edge handling via reflection."""
    y = np.asarray(y, dtype=float)
    if w <= 1 or y.size < w:
        return y
    if w % 2 == 0:
        w += 1
    pad = w // 2
    yp = np.pad(y, pad, mode="reflect")
    ker = np.ones(w) / w
    return np.convolve(yp, ker, mode="valid")

# ---------------------------------------------------------------------
# 1) Parse the [LAG] training trace from Test_1
def parse_training(path):
    pat = re.compile(
        r"\[LAG\]\s+Ep\s+(\d+)\s+\|\s+R\(comb\):\s+([-\d.]+)\s+\|\s+Smooth:\s+([-\d.]+)"
        r"\s+\|\s+Obj:\s+([-\d.]+)\s+\|\s+Cost\[.*?\]:\s+\[([^\]]+)\]\s+\|\s+lam:\s+\[([^\]]+)\]"
        r"\s+\|\s+Actor L:\s+([-\d.]+)\s+\|\s+Ent:\s+([-\d.]+)"
    )
    ep, R, Rs, obj, ent = [], [], [], [], []
    cost = {k: [] for k in ("V", "QG", "CL", "PG")}
    lam  = {k: [] for k in ("V", "QG", "CL", "PG")}
    for m in pat.finditer(_read(path)):
        ep.append(int(m.group(1)))
        R.append(float(m.group(2))); Rs.append(float(m.group(3)))
        obj.append(float(m.group(4)))
        cv = [float(x) for x in m.group(5).split(",")]
        lv = [float(x) for x in m.group(6).split(",")]
        for k, i in zip(("V", "QG", "CL", "PG"), range(4)):
            cost[k].append(cv[i]); lam[k].append(lv[i])
        ent.append(float(m.group(8)))
    return (np.array(ep), np.array(R), np.array(Rs), np.array(obj),
            {k: np.array(v) for k, v in cost.items()},
            {k: np.array(v) for k, v in lam.items()}, np.array(ent))

def fig_objective(train):
    ep, R, Rs, obj, cost, lam, ent = train
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(ep, obj, color=C_SL, lw=2.0)
    #ax.plot(ep, obj, color="#9ec5e8", lw=0.9, label="Raw")
    #ax.plot(ep, smooth(obj), color=C_SL, lw=2.0, label="Smoothed")
    ax.set_xlabel("Episode"); ax.set_ylabel("Load-shedding objective")
    ax.legend(loc="lower right", fontsize=14)
    out = os.path.join(OUTPUT_DIR, "rbts_objective.png")
    fig.savefig(out, dpi=DPI); plt.close(fig); print("wrote", out)

def fig_costs(train):
    ep, R, Rs, obj, cost, lam, ent = train
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for k, c in zip(("V", "QG", "CL", "PG"), (C_V, C_QG, C_CL, C_PG)):
        y = np.clip(cost[k], 1e-6, None)
        ax.plot(ep, y, color=c, lw=0.6, alpha=0.22)                  # raw (faint)
        ax.plot(ep, np.clip(smooth(y), 1e-6, None), color=c, lw=1.8,
                label=fr"$C_{{\mathrm{{{k}}}}}$")                     # smoothed
    ax.set_yscale("log"); ax.set_xlabel("Episode"); ax.set_ylabel("Constraint cost")
    ax.set_yscale("log"); ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    #ax.legend(fontsize=24, loc="lower right", bbox_to_anchor=(0.98, 0.01))
    ax.legend(fontsize=24, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol = 4, handlelength=0.8, columnspacing=0.4, borderpad=0.1)
    out = os.path.join(OUTPUT_DIR, "rbts_costs.png")
    fig.savefig(out, dpi=DPI); plt.close(fig); print("wrote", out)

def fig_multipliers(train):
    ep, R, Rs, obj, cost, lam, ent = train
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for k, c in zip(("V", "QG", "CL", "PG"), (C_V, C_QG, C_CL, C_PG)):
        ax.plot(ep, lam[k], lw=1.8, color=c, label=fr"$\lambda_{{\mathrm{{{k}}}}}$")
    ax.set_xlabel("Episode"); ax.set_ylabel("Lagrange multiplier")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.legend(fontsize=24, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol = 4, handlelength=0.8, columnspacing=0.4, borderpad=0.1)
    out = os.path.join(OUTPUT_DIR, "rbts_multipliers.png")
    fig.savefig(out, dpi=DPI); plt.close(fig); print("wrote", out)

def fig_entropy(train):
    ep, R, Rs, obj, cost, lam, ent = train
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(ep, ent, color="#2ca02c", lw=1.8)
    ax.set_xlabel("Episode"); ax.set_ylabel("Policy entropy")
    out = os.path.join(OUTPUT_DIR, "rbts_entropy.png")
    fig.savefig(out, dpi=DPI); plt.close(fig); print("wrote", out)

# ---------------------------------------------------------------------
# 2) Parse evaluation blocks (1000-state performance metrics)
def parse_eval_blocks(path):
    txt = _read(path); blocks = []
    for blk in txt.split("CURRENT GNN TEST - 1000 STATE EVALUATIONS")[1:]:
        def g(p, d=0.0):
            m = re.search(p, blk); return float(m.group(1)) if m else d
        blocks.append(dict(
            absdev=g(r"Absolute deviation\s*=\s*([-\d.]+)"),
            ndiff =g(r"differ by more than 0\.01 MW\s*=\s*(\d+)"),
            Vn    =g(r"Number of samples with V violation:\s*(\d+)"),
            QGn   =g(r"Number of samples with QG violation:\s*(\d+)"),
            Sijn  =g(r"Number of samples with Sij violation:\s*(\d+)"),
            PGn   =g(r"Number of samples with PG violation:\s*(\d+)"),
            issue =g(r"at least one issue\s*=\s*(\d+)"),
        ))
    return blocks

def collect_metrics(test1, test3):
    b1 = parse_eval_blocks(test1)   # [0]=GNN-SP (post-PT), [-1]=GNN-RL (post-RL)
    b3 = parse_eval_blocks(test3)   # [0]=GNN-SL (post-PT)
    return {"GNN-SP": b1[0], "GNN-SL": b3[0], "GNN-RL": b1[-1]}

def fig_feasibility(test1, test3, Neval=1000):
    M = collect_metrics(test1, test3)
    pols = ["GNN-SP", "GNN-SL", "GNN-RL"]; cols = [C_SP, C_SL, C_RL]
    keys   = ["Vn", "QGn", "Sijn", "PGn", "issue"]
    labels = ["V", "QG", "CL", "PG", "Shield\nactivation"]
    x = np.arange(len(keys)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for i, (p, c) in enumerate(zip(pols, cols)):
        vals = [M[p][k] for k in keys]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=p, color=c)
        ax.bar_label(bars, fmt="%d", fontsize=14, padding=1)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(f"Number of states (out of {Neval})")
    ax.legend(fontsize=14)
    out = os.path.join(OUTPUT_DIR, "rbts_feasibility.png")
    fig.savefig(out, dpi=DPI); plt.close(fig); print("wrote", out)
    return M

# ---------------------------------------------------------------------
# 3) Parse single-state bus reports and compare controls
def parse_bus_report(path):
    rows = {}; inbr = False
    for ln in _read(path).split("\n"):
        if "BUS REPORT" in ln:
            inbr = True; continue
        if inbr:
            if ln.strip().startswith("Total"):
                break
            f = ln.split()
            if len(f) >= 10 and re.match(r"^\d+$", f[0]):
                b = int(f[0])
                rows[b] = dict(V=float(f[1]), PG=float(f[5]), QG=float(f[6]), PL=float(f[8]))
    return rows

def fig_controls(opf_file, sl_file, rl_file):
    O = parse_bus_report(opf_file); S = parse_bus_report(sl_file); R = parse_bus_report(rl_file)
    vbus = [1, 2, 3]   # PV + slack(1)
    gbus = [1, 2, 3]   # dispatchable gens + slack last
    w = 0.26

    # --- voltage setpoints ---
    x = np.arange(len(vbus))
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.bar(x - w, [O[b]["V"] for b in vbus], w, label="AC-OPF", color=C_OPF)
    ax.bar(x, [R[b]["V"] for b in vbus], w, label="RL-GNN", color=C_RL)
    ax.bar(x + w,     [S[b]["V"] for b in vbus], w, label="SL-GNN", color=C_SL)        
    ax.set_xticks(x); ax.set_xticklabels(vbus); ax.set_ylim(1.00, 1.10)
    ax.set_xlabel("Bus"); ax.set_ylabel("Voltage setpoint (pu)"); 
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    #ax.legend(fontsize=24, loc="lower left")    
    ax.legend(fontsize=22, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol = 4, handlelength=0.8, columnspacing=0.4, borderpad=0.1)
    out = os.path.join(OUTPUT_DIR, "rbts_controls_voltage.png")
    fig.savefig(out, dpi=DPI); plt.close(fig); print("wrote", out)

    # --- active-power dispatch ---
    x = np.arange(len(gbus))
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.bar(x - w, [O[b]["PG"] for b in gbus], w, label="AC-OPF", color=C_OPF)
    ax.bar(x, [R[b]["PG"] for b in gbus], w, label="RL-GNN", color=C_RL)
    ax.bar(x + w,     [S[b]["PG"] for b in gbus], w, label="SL-GNN", color=C_SL)
    #lbl = [str(b) for b in gbus]; lbl[0] = "1\n(slack)"
    #ax.set_xticks(x); ax.set_xticklabels(lbl); ax.set_ylim(1.00, 85.0)
    ax.set_xlabel("Bus"); ax.set_ylabel("Active power (MW)"); 
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.legend(fontsize=22, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol = 4, handlelength=0.8, columnspacing=0.4, borderpad=0.1)
    #ax.legend(fontsize=24, loc="lower left")
    out = os.path.join(OUTPUT_DIR, "rbts_controls_dispatch.png")
    fig.savefig(out, dpi=DPI); plt.close(fig); print("wrote", out)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    t1 = os.path.join(INPUT_DIR, "training_dynamics_performance_1.txt")
    t3 = os.path.join(INPUT_DIR, "training_dynamics_performance_2.txt")

    train = parse_training(t1)
    #fig_objective(train)
    fig_costs(train)
    fig_multipliers(train)
    #fig_entropy(train)

    #fig_feasibility(t1, t3)

    fig_controls(os.path.join(INPUT_DIR, "AC-OPF-1.txt"),
                 os.path.join(INPUT_DIR, "AC-PF-GNN-PT-1.txt"),
                 os.path.join(INPUT_DIR, "AC-PF-GNN-RL-1.txt"))