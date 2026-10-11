"""Multi-panel result figures for the paper (Figs. 3-8).

Every value is read from the saved run files (seed*.json, seed*_scores.npz);
nothing is typed by hand. Usage:
    python scripts/paper_figures.py --results ../results --out ../results/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve, roc_curve

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.3, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

COL = {
    "sgmf_nodp": "#1f4e79", "concat_nodp": "#5b9bd5", "abl_no_cycle_nodp": "#8064a2",
    "fedprox_nodp": "#2e8b57", "gbm_both": "#c55a11", "lr_both": "#7f7f7f",
    "probe_gbm_ehr": "#bf9000", "probe_gbm_grid": "#a6a6a6",
    "sgmf_eps16": "#7b2c2c", "sgmf_eps8": "#b03a2e", "sgmf_eps4": "#d9534f", "sgmf_eps1": "#f1948a",
    "atk1_signflip_fedavg_nodp": "#c00000", "atk1_signflip_trim_nodp": "#2e8b57",
}
NAME = {
    "sgmf_nodp": "SG-MedFusion (no DP)", "concat_nodp": "Late fusion", "abl_no_cycle_nodp": "No match-predict",
    "fedprox_nodp": "FedProx", "gbm_both": "Gradient boosting", "lr_both": "Logistic regression",
    "probe_gbm_ehr": "EHR-only probe", "probe_gbm_grid": "Power-only probe",
    "sgmf_eps16": r"DP $\varepsilon$ = 16", "sgmf_eps8": r"DP $\varepsilon$ = 8",
    "sgmf_eps4": r"DP $\varepsilon$ = 4", "sgmf_eps1": r"DP $\varepsilon$ = 1",
    "atk1_signflip_fedavg_nodp": "Attack, FedAvg", "atk1_signflip_trim_nodp": "Attack, trimmed mean",
}
TYPES = [("fabricate", "Fabricated"), ("omit", "Omitted"), ("wrong_patient", "Wrong patient"),
         ("shift", "Shifted"), ("consistent", "Consistent\n(false alarm)")]
SEEDS = (0, 1, 2)
T95 = {3: 4.303}  # t(0.975, df=2)


def panel(ax, letter):
    ax.text(-0.13, 1.04, "(%s)" % letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


class Runs:
    def __init__(self, root):
        self.root = Path(root)
        self._j, self._z = {}, {}

    def j(self, e, s):
        k = (e, s)
        if k not in self._j:
            p = self.root / e / ("seed%d.json" % s)
            self._j[k] = json.loads(p.read_text()) if p.exists() else None
        return self._j[k]

    def z(self, e, s):
        k = (e, s)
        if k not in self._z:
            p = self.root / e / ("seed%d_scores.npz" % s)
            if p.exists():
                with np.load(p) as f:
                    self._z[k] = {n: f[n] for n in f.files}
            else:
                self._z[k] = None
        return self._z[k]

    def vals(self, e, path):
        out = []
        for s in SEEDS:
            d = self.j(e, s)
            if d is None:
                continue
            for p in path.split("/"):
                d = d.get(p) if isinstance(d, dict) else None
            if d is not None:
                out.append(float(d))
        return np.array(out)


def ci(v):
    m = v.mean()
    if len(v) < 2:
        return m, m, m
    h = T95.get(len(v), 2.0) * v.std(ddof=1) / np.sqrt(len(v))
    return m, m - h, m + h


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / ("%s.%s" % (name, ext)), dpi=600 if ext == "png" else None)
    plt.close(fig)


def flag_rates(R, e):
    """Per-type flag rate at the validation-calibrated 5%-false-alarm threshold."""
    rates = {t: [] for t, _ in TYPES}
    for s in SEEDS:
        d, z = R.j(e, s), R.z(e, s)
        if d is None or z is None:
            continue
        thr = d["test_at_val_fpr"]["threshold"]
        for t, _ in TYPES:
            m = z["ntype"] == t
            rates[t].append(float((z["score"][m] >= thr).mean()))
    return {t: np.array(v) for t, v in rates.items()}


# ------------------------------------------------------------------ Fig. 3
def fig_training(R, out):
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.55), gridspec_kw={"wspace": 0.5})
    ax = axs[0]
    for e in ["sgmf_nodp", "concat_nodp", "abl_no_cycle_nodp", "fedprox_nodp",
              "sgmf_eps16", "sgmf_eps8", "sgmf_eps4", "sgmf_eps1"]:
        H = [R.j(e, s)["history"] for s in SEEDS if R.j(e, s)]
        rounds = [h["round"] for h in H[0]]
        A = np.array([[h["val_auroc"] for h in hh] for hh in H])
        m, sd = A.mean(0), A.std(0, ddof=1)
        ls = "-" if "eps" not in e else "--"
        ax.plot(rounds, m, ls, marker="o", ms=2.8, lw=1.2, color=COL[e], label=NAME[e])
        ax.fill_between(rounds, m - sd, m + sd, color=COL[e], alpha=0.15, lw=0)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set(xlabel="Communication round", ylabel="Validation AUROC", xticks=[5, 10, 15, 20], ylim=(0.48, 0.82))
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.27), fontsize=6.3, handlelength=1.6, columnspacing=0.8)
    panel(ax, "a")

    ax = axs[1]
    for e in ["sgmf_eps1", "sgmf_eps4", "sgmf_eps8", "sgmf_eps16"]:
        for s in SEEDS:
            h = R.j(e, s)["history"]
            ax.plot([x["eps"] for x in h], [x["val_auroc"] for x in h], "-o", ms=2.5, lw=0.9,
                    color=COL[e], alpha=0.85, label=NAME[e] if s == 0 else None)
    ax.axhline(R.vals("sgmf_nodp", "val_auroc").mean(), color=COL["sgmf_nodp"], lw=1, ls="--")
    ax.text(1.02, R.vals("sgmf_nodp", "val_auroc").mean() - 0.025, "no DP", color=COL["sgmf_nodp"], fontsize=7)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8, 16], ["0.5", "1", "2", "4", "8", "16"])
    ax.minorticks_off()
    ax.set(xlabel=r"Privacy spent so far, $\varepsilon$ ($\delta=10^{-5}$)", ylabel="Validation AUROC", ylim=(0.48, 0.82))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, fontsize=6.3)
    panel(ax, "b")

    ax = axs[2]
    for e in ["sgmf_nodp", "atk1_signflip_trim_nodp", "atk1_signflip_fedavg_nodp"]:
        H = [R.j(e, s)["history"] for s in SEEDS]
        rounds = [h["round"] for h in H[0]]
        L = np.array([[h["val_loss"] for h in hh] for hh in H])
        lab = {"sgmf_nodp": "No attack, FedAvg"}.get(e, NAME[e])
        ax.plot(rounds, np.exp(np.log(L).mean(0)), "-o", ms=2.8, lw=1.2, color=COL[e] if e != "sgmf_nodp" else "#1f4e79", label=lab)
    ax.set_yscale("log")
    ax.set(xlabel="Communication round", ylabel="Validation loss (log scale)", xticks=[5, 10, 15, 20])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), fontsize=6.3)
    panel(ax, "c")
    save(fig, out, "fig3_training")


# ------------------------------------------------------------------ Fig. 4
def fig_discrimination(R, out):
    fig = plt.figure(figsize=(7.16, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.42)
    models = ["sgmf_nodp", "gbm_both", "lr_both", "probe_gbm_ehr", "probe_gbm_grid"]
    grid = np.linspace(0, 1, 501)
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for e in models:
        T, P = [], []
        for s in SEEDS:
            z = R.z(e, s)
            f, t, _ = roc_curve(z["y"], z["score"])
            T.append(np.interp(grid, f, t))
            p, r, _ = precision_recall_curve(z["y"], z["score"])
            P.append(np.interp(grid, r[::-1], p[::-1]))
        T, P = np.array(T), np.array(P)
        a = R.vals(e, "test/main/auroc")
        lab = "%s (%.3f)" % (NAME[e], a.mean())
        lw = 1.6 if e == "sgmf_nodp" else 1.0
        ax.plot(grid, T.mean(0), color=COL[e], lw=lw, label=lab)
        ax.fill_between(grid, T.min(0), T.max(0), color=COL[e], alpha=0.2, lw=0)
        ax2.plot(grid, P.mean(0), color=COL[e], lw=lw)
        ax2.fill_between(grid, P.min(0), P.max(0), color=COL[e], alpha=0.2, lw=0)
        if e in ("sgmf_nodp", "gbm_both", "lr_both"):
            fp, tp = R.vals(e, "test_at_val_fpr/fpr").mean(), R.vals(e, "test_at_val_fpr/recall").mean()
            ax.plot(fp, tp, "o", mfc="white", mec=COL[e], mew=1.3, ms=4.5, zorder=5)
            pr = R.vals(e, "test_at_val_fpr/precision").mean()
            ax2.plot(tp, pr, "o", mfc="white", mec=COL[e], mew=1.3, ms=4.5, zorder=5)
    ax.plot([0, 1], [0, 1], ":", color="k", lw=0.6)
    ax.axvline(0.05, color="#555", lw=0.6, ls="--")
    ax.text(0.07, 0.9, "5% false alarms", fontsize=6.5, color="#555")
    ax.set(xlabel="False-alarm rate", ylabel="Detection rate", xlim=(0, 1), ylim=(0, 1.01))
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], marker="o", ls="", mfc="white", mec="k", ms=4.5))
    l.append("operating point (5% FPR on validation)")
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.06, -0.04), ncol=3, fontsize=6.6)
    panel(ax, "a")
    ax2.axhline(0.5, color="k", lw=0.6, ls=":")
    ax2.set(xlabel="Detection rate (recall)", ylabel="Precision", xlim=(0, 1), ylim=(0.46, 1.01))
    ax2.text(0.02, 0.475, "chance (50% prevalence)", fontsize=6.3)
    panel(ax2, "b")

    ax = fig.add_subplot(gs[2])
    rows = [("sgmf_nodp", "Proposed"), ("concat_nodp", "Ablation"), ("abl_no_cycle_nodp", "Ablation"),
            ("fedprox_nodp", "Ablation"), ("gbm_both", "Baseline"), ("lr_both", "Baseline"),
            ("probe_gbm_ehr", "Probe"), ("probe_gbm_grid", "Probe"),
            ("sgmf_eps16", "DP"), ("sgmf_eps8", "DP"), ("sgmf_eps4", "DP"), ("sgmf_eps1", "DP")]
    for i, (e, _) in enumerate(rows):
        v = R.vals(e, "test/main/auroc")
        m, lo, hi = ci(v)
        ax.plot([lo, hi], [i, i], color=COL[e], lw=1.6, solid_capstyle="butt")
        ax.plot(m, i, "D" if e == "sgmf_nodp" else "o", color=COL[e], ms=4.2)
        ax.plot(v, [i] * len(v), "|", color="k", ms=4, mew=0.6, alpha=0.6)
        ax.text(0.86, i, "%.3f" % m, va="center", ha="right", fontsize=6.6)
    ax.axvline(0.5, color="k", lw=0.6, ls=":")
    ax.axvline(R.vals("sgmf_nodp", "test/main/auroc").mean(), color=COL["sgmf_nodp"], lw=0.6, ls="--")
    ax.set_yticks(range(len(rows)), [NAME[e] for e, _ in rows], fontsize=6.8)
    ax.invert_yaxis()
    ax.set(xlabel="Test AUROC (95% CI)", xlim=(0.48, 0.87))
    ax.grid(axis="y", visible=False)
    for y in (0.5, 3.5, 5.5, 7.5):
        ax.axhline(y, color="#cccccc", lw=0.5)
    panel(ax, "c")
    save(fig, out, "fig4_discrimination")


# ------------------------------------------------------------------ Fig. 5
def fig_operating(R, out):
    fig = plt.figure(figsize=(7.16, 5.2))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.32, width_ratios=[1.15, 1])

    ax = fig.add_subplot(gs[0, 0])
    z, d = R.z("sgmf_nodp", 0), R.j("sgmf_nodp", 0)
    thr = d["test_at_val_fpr"]["threshold"]
    data = [z["score"][z["ntype"] == t] for t, _ in TYPES]
    pal = ["#c00000", "#e46c0a", "#8064a2", "#5b9bd5", "#7f7f7f"]
    vp = ax.violinplot(data, positions=range(len(data)), widths=0.8, showextrema=False)
    for b, c in zip(vp["bodies"], pal):
        b.set_facecolor(c); b.set_alpha(0.45); b.set_edgecolor(c)
    ax.boxplot(data, positions=range(len(data)), widths=0.14, showfliers=False,
               medianprops=dict(color="k", lw=1), boxprops=dict(lw=0.7), whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
    ax.axhline(thr, color="k", lw=0.9, ls="--")
    ax.text(-0.45, thr - 0.06, r"threshold $\theta$", ha="left", fontsize=6.8)
    for i, (t, _) in enumerate(TYPES):
        ax.text(i, 1.02, "%.0f%%" % (100 * (data[i] >= thr).mean()), ha="center", fontsize=7, color=pal[i], fontweight="bold")
    ax.set_xticks(range(len(data)), [l for _, l in TYPES], fontsize=7)
    ax.set(ylabel="Inconsistency score p", ylim=(0, 1.08))
    ax.set_title("Score distribution by pair type (run 0); % flagged above", fontsize=8)
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    models = ["sgmf_nodp", "concat_nodp", "gbm_both", "lr_both"]
    w = 0.2
    for k, e in enumerate(models):
        fr = flag_rates(R, e)
        mu = [fr[t].mean() for t, _ in TYPES[:4]]
        sd = [fr[t].std(ddof=1) for t, _ in TYPES[:4]]
        x = np.arange(4) + (k - 1.5) * w
        ax.bar(x, mu, w, yerr=sd, color=COL[e], label=NAME[e], error_kw=dict(lw=0.6, capsize=1.5))
    ax.set_xticks(range(4), [l for _, l in TYPES[:4]], fontsize=7)
    ax.set(ylabel="Detection rate at 5% false alarms", ylim=(0, 1.32), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(ncol=2, loc="upper right", fontsize=6.3, columnspacing=0.8)
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    prev = np.logspace(-3, np.log10(0.5), 200)
    for e in ["sgmf_nodp", "gbm_both", "lr_both", "sgmf_eps16"]:
        tp, fp = R.vals(e, "test_at_val_fpr/recall"), R.vals(e, "test_at_val_fpr/fpr")
        P = np.array([t * prev / (t * prev + f * (1 - prev)) for t, f in zip(tp, fp)])
        ax.plot(prev, P.mean(0), color=COL[e], lw=1.4 if e == "sgmf_nodp" else 1, label=NAME[e])
        ax.fill_between(prev, P.min(0), P.max(0), color=COL[e], alpha=0.18, lw=0)
    for p in (0.01, 0.05):
        ax.axvline(p, color="#777", lw=0.5, ls=":")
        v = np.mean([t * p / (t * p + f * (1 - p)) for t, f in zip(R.vals("sgmf_nodp", "test_at_val_fpr/recall"),
                                                                    R.vals("sgmf_nodp", "test_at_val_fpr/fpr"))])
        ax.annotate("%.2f" % v, (p, v), xytext=(4, 6), textcoords="offset points", fontsize=6.5, color=COL["sgmf_nodp"])
        ax.plot(p, v, "o", color=COL["sgmf_nodp"], ms=3)
    ax.set_xscale("log")
    ax.set_xticks([0.001, 0.01, 0.05, 0.1, 0.5], ["0.1%", "1%", "5%", "10%", "50%"])
    ax.minorticks_off()
    ax.set(xlabel="Prevalence of inconsistent windows", ylabel="PPV (share of flags that are real)", ylim=(0, 1))
    ax.legend(loc="upper left", fontsize=6.5)
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    c = d["test_at_val_fpr"]["confusion"]
    M = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    Mn = M / M.sum(1, keepdims=True)
    ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, "%d\n(%.1f%%)" % (M[i, j], 100 * Mn[i, j]), ha="center", va="center",
                    fontsize=8, color="white" if Mn[i, j] > 0.6 else "black")
    ax.set_xticks([0, 1], ["Not flagged", "Flagged"]); ax.set_yticks([0, 1], ["Consistent", "Inconsistent"])
    ax.set(xlabel="Prediction", ylabel="Truth")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    panel(ax, "d")
    save(fig, out, "fig5_operating_point")


# ------------------------------------------------------------------ Fig. 6
def fig_privacy(R, out):
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.3), gridspec_kw={"wspace": 0.4})
    eps = [1, 4, 8, 16]
    E = ["sgmf_eps%d" % e for e in eps]
    nodp_a = R.vals("sgmf_nodp", "test/main/auroc")
    nodp_d = R.vals("sgmf_nodp", "test_at_val_fpr/recall")
    for ax, path, ref, lab, letter in [
            (axs[0], "test/main/auroc", nodp_a, "Test AUROC", "a"),
            (axs[1], "test_at_val_fpr/recall", nodp_d, "Detection rate at 5% false alarms", "b")]:
        means, los, his = [], [], []
        for e, x in zip(E, eps):
            v = R.vals(e, path)
            m, lo, hi = ci(v)
            means.append(m); los.append(lo); his.append(hi)
            ax.plot([x * f for f in (0.93, 1.0, 1.07)], v, "o", color=COL[e], ms=3, alpha=0.6)
        ax.fill_between(eps, los, his, color="#d9534f", alpha=0.15, lw=0)
        ax.plot(eps, means, "-s", color="#b03a2e", ms=4, lw=1.3, label="Patient-level DP-SGD")
        ax.axhspan(*ci(ref)[1:], color=COL["sgmf_nodp"], alpha=0.15, lw=0)
        ax.axhline(ref.mean(), color=COL["sgmf_nodp"], lw=1.1, ls="--", label="No DP")
        if letter == "a":
            g = R.vals("gbm_both", path).mean()
            ax.axhline(g, color=COL["gbm_both"], lw=0.9, ls="-.", label="Gradient boosting (no DP)")
            ax.axhline(0.5, color="k", lw=0.6, ls=":")
            ax.set_ylim(0.45, 0.82)
        else:
            ax.axhline(0.05, color="k", lw=0.6, ls=":")
            ax.text(9.3, 0.012, "chance", fontsize=6.3)
            ax.set_ylim(0, 0.6)
        ax.set_xscale("log")
        ax.set_xticks(eps, [str(e) for e in eps])
        ax.minorticks_off()
        ax.set(xlabel=r"Patient-level privacy budget $\varepsilon$ ($\delta=10^{-5}$)", ylabel=lab)
        panel(ax, letter)
    axs[0].legend(loc="center right", fontsize=6.3, bbox_to_anchor=(1.0, 0.62))

    ax = axs[2]
    cfg = ["sgmf_nodp"] + E
    for i, e in enumerate(cfg):
        v = R.vals(e, "mia/auc")
        m, lo, hi = ci(v)
        ax.bar(i, m - 0.5, bottom=0.5, color=COL[e], width=0.6, alpha=0.85)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color="k", lw=0.7, capsize=2)
        ax.plot([i] * len(v), v, "o", mfc="white", mec="k", ms=3, mew=0.6)
    ax.axhline(0.5, color="k", lw=0.8)
    ax.set_xticks(range(len(cfg)), ["no DP", "ε=1", "ε=4", "ε=8", "ε=16"], fontsize=7.2)
    ax.set(ylabel="Membership-inference AUC", ylim=(0.28, 0.68))
    ax.text(-0.35, 0.295, "0.5 = attacker cannot tell members\nfrom non-members", fontsize=6.3)
    panel(ax, "c")
    save(fig, out, "fig6_privacy")


# ------------------------------------------------------------------ Fig. 7
def fig_ablation_robust(R, out):
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.4), gridspec_kw={"wspace": 0.35, "width_ratios": [1.2, 1]})
    ax = axs[0]
    var = ["sgmf_nodp", "abl_no_cycle_nodp", "concat_nodp", "fedprox_nodp"]
    labs = ["Full model\n(255,208 params)", "No match-predict\n(246,435)", "Late fusion\n(188,008)", "FedProx\n(255,208)"]
    V = np.array([R.vals(e, "test/main/auroc") for e in var])
    for s in range(V.shape[1]):
        ax.plot(range(len(var)), V[:, s], "-", color="#999", lw=0.7, alpha=0.8,
                label="same run seed" if s == 0 else None)
    for i, e in enumerate(var):
        m, lo, hi = ci(V[i])
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="D", color=COL[e], ms=5, capsize=3, lw=1.2)
        ax.plot([i] * V.shape[1], V[i], "o", mfc="white", mec=COL[e], ms=3.5)
        ax.text(i + 0.12, m, "%.3f" % m, fontsize=6.8, va="center")
    ax.set_xticks(range(len(var)), labs, fontsize=6.8)
    ax.set(ylabel="Test AUROC", ylim=(0.74, 0.79))
    ax.legend(loc="lower left", fontsize=6.5)
    ax.set_title("Ablation without DP (mean, 95% CI, 3 paired runs)", fontsize=8)
    panel(ax, "a")

    ax = axs[1]
    cfg = [("sgmf_nodp", "No attack\nFedAvg"), ("atk1_signflip_fedavg_nodp", "1 of 4 sites\nmalicious\nFedAvg"),
           ("atk1_signflip_trim_nodp", "1 of 4 sites\nmalicious\ntrimmed mean")]
    for i, (e, _) in enumerate(cfg):
        v = R.vals(e, "test/main/auroc")
        c = "#1f4e79" if e == "sgmf_nodp" else COL[e]
        ax.bar(i, v.mean() - 0.5, bottom=0.5, width=0.55, color=c, alpha=0.85)
        ax.plot([i] * len(v), v, "o", mfc="white", mec="k", ms=3, mew=0.6)
        ax.text(i, v.mean() + 0.012, "%.3f" % v.mean(), ha="center", fontsize=7.2, fontweight="bold")
    ax.axhline(0.5, color="k", lw=0.8)
    ax.set_xticks(range(len(cfg)), [l for _, l in cfg], fontsize=6.8)
    ax.set(ylabel="Test AUROC", ylim=(0.45, 0.82))
    ax.set_title("Sign-flip attack (×5), no DP", fontsize=8)
    panel(ax, "b")
    save(fig, out, "fig7_ablation_robustness")


# ------------------------------------------------------------------ Fig. 8
def fig_subgroups(R, out):
    rows = []  # (group, label, values over runs, n)
    d0 = R.j("sgmf_nodp", 0)["test"]

    def sub(key, order=None):
        ks = order or sorted(d0["subgroups"][key], key=lambda k: -d0["subgroups"][key][k]["n"])
        for k in ks:
            v = R.vals("sgmf_nodp", "test/subgroups/%s/%s/auroc" % (key, k))
            rows.append((key, k, v, d0["subgroups"][key][k]["n"]))
    sub("gender", ["F", "M"])
    sub("age_group", sorted(d0["subgroups"]["age_group"]))
    sub("first_careunit")
    for h in sorted(d0["by_background"]):
        rows.append(("background", h.replace("ukdale_", "UK-DALE house "),
                     R.vals("sgmf_nodp", "test/by_background/%s/auroc" % h), d0["by_background"][h]["n"]))
    for sc in ("0.7", "1.3"):
        rows.append(("power", "device power ×%s" % sc, R.vals("sgmf_nodp", "profile_shift/%s/auroc" % sc), d0["n"]))
    rows.append(("power", "device power ×1.0 (all pairs)", R.vals("sgmf_nodp", "test/main/auroc"), d0["n"]))
    GN = {"gender": "Sex", "age_group": "Age group", "first_careunit": "ICU type",
          "background": "Background\nhousehold", "power": "Device power\nat test time"}
    LBL = {"F": "Female", "M": "Male"}
    fig, ax = plt.subplots(figsize=(3.5, 4.9))
    overall = R.vals("sgmf_nodp", "test/main/auroc").mean()
    y, ticks, labs, prev = 0, [], [], None
    gcol = {"gender": "#1f4e79", "age_group": "#2e8b57", "first_careunit": "#8064a2",
            "background": "#c55a11", "power": "#7f7f7f"}
    for g, k, v, n in rows:
        if g != prev and prev is not None:
            y += 0.6
            ax.axhline(y - 0.8, color="#ddd", lw=0.5)
        if g != prev:
            ax.text(0.585, y, GN[g], fontsize=7, fontweight="bold", va="center", ha="left",
                    transform=ax.get_yaxis_transform() if False else ax.transData)
        prev = g
        m, lo, hi = ci(v)
        ax.plot([lo, hi], [y, y], color=gcol[g], lw=1.5)
        ax.plot(m, y, "s", color=gcol[g], ms=3.8 * (0.7 + 0.6 * min(1, n / 8000)))
        ax.text(0.905, y, "%.3f" % m, fontsize=6.5, va="center")
        ax.text(0.96, y, "%d" % n, fontsize=6.5, va="center", color="#555")
        ticks.append(y); labs.append(LBL.get(k, k))
        y += 1
    ax.axvline(overall, color="k", lw=0.7, ls="--")
    ax.set_yticks(ticks, labs, fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlim(0.58, 1.0)
    ax.set_xticks([0.65, 0.7, 0.75, 0.8, 0.85])
    ax.set_xlabel("Test AUROC (mean and 95% CI over 3 runs)")
    ax.text(0.905, -1.1, "AUROC", fontsize=6.5, fontweight="bold")
    ax.text(0.96, -1.1, "pairs", fontsize=6.5, fontweight="bold")
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig8_subgroups")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R = Runs(a.results)
    fig_training(R, out)
    fig_discrimination(R, out)
    fig_operating(R, out)
    fig_privacy(R, out)
    fig_ablation_robust(R, out)
    fig_subgroups(R, out)
    fr = flag_rates(R, "sgmf_nodp")
    print("check, flag rates at 5%% FPR: " + ", ".join("%s %.3f" % (t, v.mean()) for t, v in fr.items()))


if __name__ == "__main__":
    main()
