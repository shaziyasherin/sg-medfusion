"""Aggregate results/<exp>/seed*.json into the manuscript tables (tables.md)
and figures (figures/*.png|pdf).  Every number and every plotted point comes
from these files; nothing is typed by hand.

    python scripts/make_report.py results
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sgm import stats as S  # noqa: E402

LABEL = {
    "sgmf_eps4": "SG-MedFusion (FedAvg, DP eps=4)", "sgmf_eps1": "SG-MedFusion eps=1",
    "sgmf_eps2": "SG-MedFusion eps=2", "sgmf_eps8": "SG-MedFusion eps=8",
    "sgmf_nodp": "SG-MedFusion (no DP)", "fedprox_eps4": "SG-MedFusion + FedProx (eps=4)",
    "abl_centralized_dp": "Centralised DP-SGD, same model (eps=4)",
    "centralized_nodp": "Centralised, same model, no DP (reference)",
    "concat_nodp": "Late fusion (concat), no DP", "abl_concat": "Late fusion (concat), eps=4",
    "concat_eps1": "Late fusion eps=1", "concat_eps8": "Late fusion eps=8",
    "sgmf_eps16": "SG-MedFusion eps=16", "abl_no_cycle_nodp": "- match-predict heads (no DP)",
    "fedprox_nodp": "SG-MedFusion + FedProx (no DP)",
    "atk1_signflip_fedavg_nodp": "1 of 4 sites sign-flip, FedAvg", "atk1_signflip_trim_nodp": "1 of 4 sites sign-flip, trimmed mean",
    "gbm_both": "Gradient boosting on summary features", "lr_both": "Logistic regression on summary features",
    "abl_no_cycle": "- match-predict (cycle) loss", "abl_grid_only": "Grid stream only",
    "abl_ehr_only": "EHR stream only", "abl_attn_noise": "+ logit-noise attention (sigma=1)",
    "probe_gbm_grid": "Probe: GBM, grid features only", "probe_gbm_ehr": "Probe: GBM, EHR features only",
}
MAIN = "sgmf_nodp"   # reference for architecture comparisons (DP at eps<=4 is near chance on eICU-demo)
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def load(root):
    runs = {}
    for f in sorted(glob.glob(str(Path(root) / "*" / "seed*.json"))):
        r = json.load(open(f))
        runs.setdefault(r["experiment"], []).append(r)
    return runs


def get(r, path):
    """Dotted path lookup that also handles keys containing dots ("0.7", "tpr_at_fpr_0.01")."""
    toks = path.split(".")
    i = 0
    while i < len(toks):
        if not isinstance(r, dict):
            return None
        for j in range(len(toks), i, -1):          # longest key first
            k = ".".join(toks[i:j])
            if k in r:
                r, i = r[k], j
                break
        else:
            return None
    return r


def ppv(prev, tpr, fpr):
    return prev * tpr / (prev * tpr + (1 - prev) * fpr + 1e-12)


def type_rates_at(root, exp, runs, key="test_at_val_fpr"):
    """Flag rate per pair type at the validation-calibrated threshold, from saved scores."""
    out = {}
    for r in runs.get(exp, []):
        f = Path(root) / exp / ("seed%d_scores.npz" % r["seed"])
        if not f.exists():
            continue
        with np.load(f, allow_pickle=True) as z:
            sc, nt = z["score"], z["ntype"].astype(str)
        thr = r[key]["threshold"]
        for t in np.unique(nt):
            out.setdefault(t, []).append(float((sc[nt == t] >= thr).mean()))
    return out


def fmt(s, d=3):
    if not s.get("n"):
        return "n/a"
    txt = "%.*f ± %.*f" % (d, s["mean"], d, s["sd"])
    return txt + (" [%.*f, %.*f]" % (d, s["lo"], d, s["hi"]) if "lo" in s else "")


def col(runs, exp, path, d=3):
    return fmt(S.summarise([get(r, path) for r in runs.get(exp, [])]), d)


def table(title, header, rows):
    out = ["", "### " + title, "", "| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


def seeds_of(runs, exp):
    return sorted(r["seed"] for r in runs.get(exp, []))


def paired(runs, exps, path="test.main.auroc"):
    base = {r["seed"]: get(r, path) for r in runs.get(MAIN, [])}
    raw = {}
    for e in exps:
        if e == MAIN or e not in runs:
            continue
        other = {r["seed"]: get(r, path) for r in runs[e]}
        common = sorted(set(base) & set(other))
        raw[e] = S.paired_t([base[s] for s in common], [other[s] for s in common])
    adj = S.holm({e: v["p"] for e, v in raw.items()})
    return {e: dict(v, p_holm=adj[e]) for e, v in raw.items()}


def eps_txt(runs, e):
    v = [get(r, "privacy.eps_max") for r in runs.get(e, []) if get(r, "privacy.enabled")]
    return "%.2f" % max(v) if v else "none"


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    arg = argv[0] if argv else "results"
    if arg.endswith(".json"):                 # a paths config: use its results_dir
        from sgm import config as C
        arg = C.load(arg)["paths"]["results_dir"]
    root = Path(arg)
    runs = load(root)
    L = ["# SG-MedFusion R1 - generated tables", "",
         "Values: mean ± SD over runs [95% t-interval]; n runs per row shown.  Test split only.",
         "Threshold for accuracy/F1: maximises F1 on the validation split."]

    comp = ["sgmf_nodp", "fedprox_nodp", "concat_nodp", "sgmf_eps16", "sgmf_eps8", "sgmf_eps4", "sgmf_eps1",
            "gbm_both", "lr_both"]
    pt = paired(runs, comp)
    L += table("Table A. Main comparison", ["Method", "runs", "DP eps (max site)", "AUROC", "AUPRC", "Accuracy",
                                            "F1", "FPR@TPR0.95", "PPV @1% prev.", "Δ AUROC vs main (Holm p)"],
               [[LABEL.get(e, e), len(runs[e]), eps_txt(runs, e), col(runs, e, "test.main.auroc"),
                 col(runs, e, "test.main.auprc"), col(runs, e, "test.main.accuracy"), col(runs, e, "test.main.f1"),
                 col(runs, e, "test.main.fpr_at_tpr95"),
                 fmt(S.summarise([r["test"]["prevalence"][0]["ppv"] for r in runs[e]])),
                 "—" if e == MAIN else "%+.3f (p=%.3g)" % (pt[e]["diff"] * -1, pt[e]["p_holm"])]
                for e in comp if e in runs])

    L += table("Table A2. Operating point: threshold set for 5% false alarms on validation hospitals",
               ["Method", "runs", "detection rate (TPR)", "false-alarm rate (FPR)", "precision (50% prev.)",
                "PPV @1% prev.", "PPV @5% prev.", "accuracy"],
               [[LABEL.get(e, e), len(runs[e]), col(runs, e, "test_at_val_fpr.recall"), col(runs, e, "test_at_val_fpr.fpr"),
                 col(runs, e, "test_at_val_fpr.precision"),
                 fmt(S.summarise([ppv(0.01, r["test_at_val_fpr"]["recall"], r["test_at_val_fpr"]["fpr"]) for r in runs[e]])),
                 fmt(S.summarise([ppv(0.05, r["test_at_val_fpr"]["recall"], r["test_at_val_fpr"]["fpr"]) for r in runs[e]])),
                 col(runs, e, "test_at_val_fpr.accuracy")]
                for e in comp if e in runs and all("test_at_val_fpr" in r for r in runs[e])])

    fc = ["sgmf_nodp", "concat_nodp", "sgmf_eps16", "sgmf_eps4"]
    L += table("Table B. Auxiliary forecasting (consistent pairs, +60 min)",
               ["Method", "HR MAE (bpm)", "HR RMSE (bpm)", "MAP MAE (mmHg)", "MAP RMSE (mmHg)"],
               [[LABEL.get(e, e), col(runs, e, "test.forecast.hr.mae", 2), col(runs, e, "test.forecast.hr.rmse", 2),
                 col(runs, e, "test.forecast.map.mae", 2), col(runs, e, "test.forecast.map.rmse", 2)]
                for e in fc if e in runs])

    ab = ["sgmf_nodp", "abl_no_cycle_nodp", "concat_nodp",
          "abl_centralized_dp"]
    pa = paired(runs, ab)
    L += table("Table C. Ablation (all without DP)", ["Variant", "runs", "AUROC", "F1", "Δ AUROC (Holm p)"],
               [[LABEL.get(e, e), len(runs[e]), col(runs, e, "test.main.auroc"), col(runs, e, "test.main.f1"),
                 "—" if e == MAIN else "%+.3f (p=%.3g)" % (-pa[e]["diff"], pa[e]["p_holm"])]
                for e in ab if e in runs])

    pr = ["sgmf_eps1", "sgmf_eps4", "sgmf_eps8", "sgmf_eps16", "sgmf_nodp"]
    L += table("Table D. Privacy-utility and membership inference",
               ["Setting", "achieved eps", "noise multiplier (range over sites)", "AUROC", "MIA AUC", "MIA TPR@FPR=1%"],
               [[LABEL.get(e, e), eps_txt(runs, e),
                 ("%.2f-%.2f" % (min(runs[e][0]["privacy"]["sigma"].values()), max(runs[e][0]["privacy"]["sigma"].values()))
                  if runs[e][0]["privacy"].get("sigma") else "—"),
                 col(runs, e, "test.main.auroc"), col(runs, e, "mia.auc"), col(runs, e, "mia.tpr_at_fpr_0.01")]
                for e in pr if e in runs])

    pb = ["probe_gbm_grid", "probe_gbm_ehr", "abl_grid_only", "abl_ehr_only", "gbm_both"]
    L += table("Table E. Shortcut probes (AUROC near 0.5 = no single-modality shortcut)",
               ["Probe", "AUROC", "AUROC, distinguishable pairs only"],
               [[LABEL.get(e, e), col(runs, e, "test.main.auroc"), col(runs, e, "test.distinguishable_only.auroc")]
                for e in pb if e in runs])

    if MAIN in runs:
        types = sorted(runs[MAIN][0]["test"]["by_type"])
        L += table("Table F. Flag rate by pair type (main model, validation-F1 threshold)",
                   ["Pair type", "n (run 0)", "flag rate"],
                   [[t, runs[MAIN][0]["test"]["by_type"][t]["n"], col(runs, MAIN, "test.by_type.%s.flag_rate" % t)]
                    for t in types]
                   + [["replay (stress set, never trained)", get(runs[MAIN][0], "test.replay.n"),
                       col(runs, MAIN, "test.replay.flag_rate")],
                      ["replay, physically distinguishable only", "", col(runs, MAIN, "test.replay.flag_rate_distinguishable")],
                      ["share of replays that are distinguishable", "", col(runs, MAIN, "test.replay.share_distinguishable")]])
        tr = type_rates_at(root, MAIN, runs)
        if tr:
            L += table("Table F2. Flag rate by pair type at the 5%-false-alarm threshold (main model)",
                       ["Pair type", "flag rate"], [[t, fmt(S.summarise(v))] for t, v in sorted(tr.items())])
            L += ["", "Replay pairs were not stored with scores; their flag rate is reported at the F1 threshold "
                  "in Table F, next to the false-alarm rate of consistent pairs at that threshold."]
        L += table("Table G. Distribution shift (main model)", ["Condition", "AUROC"],
                   [["test, background household %s" % hh, col(runs, MAIN, "test.by_background.%s.auroc" % hh)]
                    for hh in sorted(runs[MAIN][0]["test"].get("by_background", {}))] +
                   [["device power x0.7", col(runs, MAIN, "profile_shift.0.7.auroc")],
                    ["device power x1.3", col(runs, MAIN, "profile_shift.1.3.auroc")],
                    ["all test pairs", col(runs, MAIN, "test.main.auroc")],
                    ["physically distinguishable pairs only", col(runs, MAIN, "test.distinguishable_only.auroc")]])
        rows = []
        for g, vals in runs[MAIN][0]["test"]["subgroups"].items():
            for v in vals:
                rows.append([g, v, vals[v]["n"], col(runs, MAIN, "test.subgroups.%s.%s.auroc" % (g, v))])
        L += table("Table H. Subgroup AUROC (main model)", ["Attribute", "Group", "n (run 0)", "AUROC"], rows)
        r0 = runs[MAIN][0]
        cm = r0["test"]["main"]["confusion"]
        L += table("Table I. Confusion matrix, main model, run seed=%d (rows = truth)" % r0["seed"],
                   ["", "predicted consistent", "predicted inconsistent"],
                   [["consistent", cm["tn"], cm["fp"]], ["inconsistent", cm["fn"], cm["tp"]]])
        L += ["", "Accuracy from this matrix: %.4f; F1: %.4f (identical to Table A run %d)."
              % (r0["test"]["main"]["accuracy"], r0["test"]["main"]["f1"], r0["seed"])]
        cm2 = r0["test_at_val_fpr"]["confusion"]
        L += table("Table I2. Confusion matrix at the 5%%-false-alarm threshold, run seed=%d" % r0["seed"],
                   ["", "predicted consistent", "predicted inconsistent"],
                   [["consistent", cm2["tn"], cm2["fp"]], ["inconsistent", cm2["fn"], cm2["tp"]]])
        L += ["", "Accuracy %.4f, detection rate %.4f, false-alarm rate %.4f (identical to Table A2 run %d)."
              % (r0["test_at_val_fpr"]["accuracy"], r0["test_at_val_fpr"]["recall"], r0["test_at_val_fpr"]["fpr"], r0["seed"])]
        L += table("Table J. Model size and latency (forward pass only)",
                   ["Setting", "parameters", "p50 ms (batch 1)", "p95 ms", "throughput/s (batch 256)", "device"],
                   [[LABEL.get(e, e), runs[e][0].get("n_params"), col(runs, e, "latency.p50_ms", 2),
                     col(runs, e, "latency.p95_ms", 2), col(runs, e, "latency.throughput_per_s_batch256", 0),
                     get(runs[e][0], "latency.device")] for e in (MAIN, "sgmf_eps4", "concat_nodp") if e in runs])

    rb = [("none", "fedavg", MAIN), ("sign-flip x5, 1 of 4 sites", "fedavg", "atk1_signflip_fedavg_nodp"),
          ("sign-flip x5, 1 of 4 sites", "trimmed mean (b=1)", "atk1_signflip_trim_nodp")]
    L += table("Table K. Robustness to a malicious site (no DP)", ["Attack", "Aggregator", "AUROC", "F1"],
               [[a, g, col(runs, e, "test.main.auroc"), col(runs, e, "test.main.f1")] for a, g, e in rb if e in runs])

    envs = {json.dumps(r.get("environment", {}), sort_keys=True) for rs in runs.values() for r in rs}
    L += table("Table M. Computing environment (fills manuscript Table 5, 'Hardware')", ["Field", "Value"],
               [[k, v] for e in sorted(envs) for k, v in json.loads(e).items()])
    if len(envs) > 1:
        L += ["", "WARNING: runs were made in %d different environments; report all." % len(envs)]
    n_runs = {e: len(rs) for e, rs in runs.items()}
    L += ["", "Runs per experiment: " + ", ".join("%s=%d" % kv for kv in sorted(n_runs.items()))]

    audits = sorted(glob.glob(str(root / "leakage_audit_seed*.json")))
    if audits:
        a = json.load(open(audits[0]))
        L += table("Table L. Leakage audit (%s)" % Path(audits[0]).name, ["Check", "Value"],
                   [["patients shared between splits", sum(a["patient_overlap"].values())],
                    ["grid sources drawn from another split", a["cross_split_grid_sources"]],
                    ["wrong-patient pairs using the same patient", a["wrong_patient_same_patient"]],
                    ["households shared between splits", sum(a["household_overlap"].values())],
                    ["overlapping windows within a stay", a["overlapping_windows_within_stay"]],
                    ["hospitals shared between splits", sum(a.get("hospital_overlap", {}).values())
                     if "hospital_overlap" in a else "n/a (temporal split)"],
                    ["hospitals per split", a.get("n_hospitals", "n/a")],
                    ["train anchor-year groups", ", ".join(a["temporal"]["train_groups"])],
                    ["test anchor-year groups", ", ".join(a["temporal"]["test_groups"])],
                    ["zero-filled grid windows (sample)", a.get("zero_filled_grid_windows_in_sample")],
                    ["audit passed", a["passed"]]])
    (root / "tables.md").write_text("\n".join(L), encoding="utf-8")
    figures(runs, root / "figures")
    print("wrote", root / "tables.md")


def figures(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "font.family": "DejaVu Sans",
                         "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name, layout=True):
        if layout:
            fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / ("%s.%s" % (name, ext)), dpi=300)
        plt.close(fig)

    # ROC / PR from saved scores of the lowest seed
    show = [e for e in (MAIN, "concat_nodp", "sgmf_eps16", "sgmf_eps4", "gbm_both") if e in runs]
    if show:
        fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.9))
        for c, e in zip(OKABE, show):
            sd = min(r["seed"] for r in runs[e])
            with np.load(Path(out).parent / e / ("seed%d_scores.npz" % sd), allow_pickle=True) as z:
                y, sc = z["y"], z["score"]
            f, t, _ = roc_curve(y, sc); p, rc, _ = precision_recall_curve(y, sc)
            s = S.summarise([r["test"]["main"]["auroc"] for r in runs[e]])
            ax[0].plot(f, t, color=c, lw=1.4, label="%s (AUROC %.3f ± %.3f)" % (LABEL[e], s["mean"], s.get("sd", 0)))
            ax[1].plot(rc, p, color=c, lw=1.4, label=LABEL[e])
        ax[0].plot([0, 1], [0, 1], ":", color="grey", lw=0.8)
        ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="(a) ROC, run seed=%d" % sd)
        ax[1].set(xlabel="Recall", ylabel="Precision", title="(b) Precision-recall (50% prevalence)", ylim=(0.4, 1.01))
        h, l = ax[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7)
        fig.tight_layout(rect=(0, 0.2, 1, 1))
        save(fig, "fig_roc_pr", layout=False)

    # privacy-utility with CIs
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for c, (name, series) in zip(OKABE, [("SG-MedFusion", ["sgmf_eps1", "sgmf_eps4", "sgmf_eps8", "sgmf_eps16"])]):
        xs, ms, lo, hi = [], [], [], []
        for e in series:
            if e not in runs:
                continue
            s = S.summarise([r["test"]["main"]["auroc"] for r in runs[e]])
            xs.append(max(r["privacy"]["eps_max"] for r in runs[e])); ms.append(s["mean"])
            lo.append(s["mean"] - s.get("lo", s["mean"])); hi.append(s.get("hi", s["mean"]) - s["mean"])
        if xs:
            ax.errorbar(xs, ms, yerr=[lo, hi], color=c, marker="o", ms=4, capsize=3, lw=1.2, label=name)
        ref = {"SG-MedFusion": "sgmf_nodp", "Late fusion": "concat_nodp"}[name]
        if ref in runs:
            ax.axhline(S.summarise([r["test"]["main"]["auroc"] for r in runs[ref]])["mean"], color=c, ls="--", lw=0.9)
    ax.set_xscale("log")
    ax.set(xlabel="Privacy budget ε (δ = 1e-5, patient level)", ylabel="Test AUROC",
           title="Privacy-utility (dashed: no DP)")
    ax.legend(frameon=False)
    save(fig, "fig_privacy_utility")

    # convergence
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for c, e in zip(OKABE, [MAIN, "fedprox_nodp", "concat_nodp", "sgmf_eps4"]):
        if e not in runs:
            continue
        rounds = [h["round"] for h in runs[e][0]["history"]]
        A = np.array([[h["val_auroc"] for h in r["history"]] for r in runs[e]])
        m, s = A.mean(0), A.std(0, ddof=1) if len(A) > 1 else np.zeros(A.shape[1])
        ax.plot(rounds, m, color=c, lw=1.3, label=LABEL[e]); ax.fill_between(rounds, m - s, m + s, color=c, alpha=.2, lw=0)
    ax.set(xlabel="Communication round", ylabel="Validation AUROC", title="Convergence (mean ± SD over runs)")
    ax.set_xticks(sorted({h["round"] for rs in runs.values() for r in rs for h in r.get("history", [])}))
    ax.legend(frameon=False, fontsize=6.5)
    save(fig, "fig_convergence")

    # confusion matrix
    if MAIN in runs:
        r0 = runs[MAIN][0]; cm = r0["test_at_val_fpr"]["confusion"]
        M = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        ax.imshow(M, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, "{:,}".format(M[i, j]), ha="center", va="center",
                        color="white" if M[i, j] > M.max() / 2 else "black", fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["consistent", "inconsistent"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["consistent", "inconsistent"])
        ax.set(xlabel="Predicted", ylabel="Actual",
               title="Threshold: 5%% false alarms (validation)\nTPR %.3f, FPR %.3f, run seed %d"
               % (r0["test_at_val_fpr"]["recall"], r0["test_at_val_fpr"]["fpr"], r0["seed"]))
        save(fig, "fig_confusion")

    # ablation
    ab = [e for e in ["sgmf_nodp", "abl_no_cycle_nodp", "concat_nodp"] if e in runs]
    if ab:
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        ss = [S.summarise([r["test"]["main"]["auroc"] for r in runs[e]]) for e in ab]
        y = np.arange(len(ab))[::-1]
        ax.barh(y, [s["mean"] for s in ss], color=OKABE[0], height=.6,
                xerr=[[s["mean"] - s.get("lo", s["mean"]) for s in ss], [s.get("hi", s["mean"]) - s["mean"] for s in ss]],
                capsize=3)
        ax.set_yticks(y); ax.set_yticklabels([LABEL[e] for e in ab], fontsize=7)
        ax.set(xlabel="Test AUROC (mean, 95% CI)", xlim=(0.4, 1.0), title="Ablation (no DP)")
        save(fig, "fig_ablation")


if __name__ == "__main__":
    main()
