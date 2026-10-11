"""Build the manuscript's results tables (main text: Tables VI-XI) and the
supplementary tables (S1-S8) directly from results/*/seed*.json, in compact
form for Word.  Output: two Markdown fragments included by the manuscript build.

    python scripts/paper_tables.py ../results ../src
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_report import load, get, ppv, type_rates_at  # noqa: E402
from sgm import stats as S  # noqa: E402

root, out = Path(sys.argv[1]), Path(sys.argv[2])
runs = load(root)
N = {
    "sgmf_nodp": "SG-MedFusion, federated, no DP", "fedprox_nodp": "+ FedProx (μ = 0.01), no DP",
    "concat_nodp": "Late fusion (concatenation), no DP", "abl_no_cycle_nodp": "Without match-predict heads, no DP",
    "sgmf_eps16": "SG-MedFusion, DP ε = 16", "sgmf_eps8": "SG-MedFusion, DP ε = 8",
    "sgmf_eps4": "SG-MedFusion, DP ε = 4", "sgmf_eps1": "SG-MedFusion, DP ε = 1",
    "gbm_both": "Gradient boosting, summary features", "lr_both": "Logistic regression, summary features",
    "probe_gbm_grid": "Probe: power features only", "probe_gbm_ehr": "Probe: EHR features only",
}


def m(vals, d=3, ci=False):
    s = S.summarise(vals)
    if not s.get("n"):
        return "—"
    t = "%.*f ± %.*f" % (d, s["mean"], d, s["sd"])
    return t + (" [%.*f, %.*f]" % (d, s["lo"], d, s["hi"]) if ci and "lo" in s else "")


def c(e, path, d=3, ci=False):
    return m([get(r, path) for r in runs.get(e, [])], d, ci)


def tab(title, head, rows, note=None):
    L = ["", "**%s**" % title, "", "| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    L += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    if note:
        L += ["", "*%s*" % note]
    return L


def holm_vs_ref(exps, ref="sgmf_nodp"):
    base = {r["seed"]: r["test"]["main"]["auroc"] for r in runs[ref]}
    raw = {}
    for e in exps:
        if e == ref or e not in runs:
            continue
        o = {r["seed"]: r["test"]["main"]["auroc"] for r in runs[e]}
        k = sorted(set(base) & set(o))
        raw[e] = S.paired_t([o[s] for s in k], [base[s] for s in k])
    adj = S.holm({e: v["p"] for e, v in raw.items()})
    return {e: "%+.3f (%s)" % (v["diff"], ("p < 0.001" if adj[e] < 0.001 else "p = %.3f" % adj[e]))
            for e, v in raw.items()}


main_e = ["sgmf_nodp", "fedprox_nodp", "concat_nodp", "gbm_both", "lr_both",
          "sgmf_eps16", "sgmf_eps8", "sgmf_eps4", "sgmf_eps1"]
dl = holm_vs_ref(main_e)
M = ["## Results tables (main text)"]
M += tab("Table VI. Test performance at 37 held-out hospitals (three runs, mean ± SD; 95% t-interval for AUROC).",
         ["Model", "AUROC", "AUPRC", "FPR at 95% TPR", "Δ AUROC vs. no-DP model (Holm)"],
         [[N[e], c(e, "test.main.auroc", ci=True), c(e, "test.main.auprc"), c(e, "test.main.fpr_at_tpr95"),
           "—" if e == "sgmf_nodp" else dl.get(e, "—")] for e in main_e if e in runs],
         "Paired t-tests over the three runs with Holm correction; with three runs these tests have little power.")
M += tab("Table VII. Operating point: threshold calibrated to a 5% false-alarm rate on the validation hospitals.",
         ["Model", "Detection rate", "False-alarm rate", "Precision (50% prev.)", "PPV, 5% prev.", "PPV, 1% prev."],
         [[N[e], c(e, "test_at_val_fpr.recall"), c(e, "test_at_val_fpr.fpr"), c(e, "test_at_val_fpr.precision"),
           m([ppv(0.05, r["test_at_val_fpr"]["recall"], r["test_at_val_fpr"]["fpr"]) for r in runs[e]]),
           m([ppv(0.01, r["test_at_val_fpr"]["recall"], r["test_at_val_fpr"]["fpr"]) for r in runs[e]])]
          for e in ["sgmf_nodp", "concat_nodp", "gbm_both", "lr_both", "sgmf_eps16", "sgmf_eps4"] if e in runs])
tr = type_rates_at(root, "sgmf_nodp", runs)
lab = {"consistent": "Consistent (false alarms)", "fabricate": "Fabricated entry", "omit": "Omitted entry",
       "shift": "Shifted in time", "wrong_patient": "Wrong patient"}
M += tab("Table VIII. Flag rate by pair type, federated SG-MedFusion without DP, 5% false-alarm threshold.",
         ["Pair type", "Test pairs (run 0)", "Flag rate"],
         [[lab[t], runs["sgmf_nodp"][0]["test"]["by_type"][t]["n"], m(v)] for t, v in sorted(tr.items())]
         + [["All pairs: AUROC", runs["sgmf_nodp"][0]["test"]["n"], c("sgmf_nodp", "test.main.auroc")],
            ["Physically distinguishable pairs only: AUROC", runs["sgmf_nodp"][0]["test"]["distinguishable_only"]["n"],
             c("sgmf_nodp", "test.distinguishable_only.auroc")]],
         "About 21% of shift and wrong-patient pairs were physically distinguishable; fabricated and omitted entries always are.")
M += tab("Table X. Ablation without DP.", ["Variant", "Parameters", "AUROC", "Δ AUROC (Holm)"],
         [[N[e], runs[e][0].get("n_params", "—"), c(e, "test.main.auroc", ci=True),
           "—" if e == "sgmf_nodp" else holm_vs_ref(["abl_no_cycle_nodp", "concat_nodp"])[e]]
          for e in ["sgmf_nodp", "abl_no_cycle_nodp", "concat_nodp"] if e in runs])
M += tab("Table XI. Robustness to one malicious site of four (sign-flipped updates ×5), no DP.",
         ["Setting", "Aggregation", "AUROC", "F1"],
         [["No attack", "FedAvg", c("sgmf_nodp", "test.main.auroc"), c("sgmf_nodp", "test.main.f1")],
          ["Attack", "FedAvg", c("atk1_signflip_fedavg_nodp", "test.main.auroc"), c("atk1_signflip_fedavg_nodp", "test.main.f1")],
          ["Attack", "Trimmed mean (b = 1)", c("atk1_signflip_trim_nodp", "test.main.auroc"), c("atk1_signflip_trim_nodp", "test.main.f1")]])
pr = ["sgmf_eps1", "sgmf_eps4", "sgmf_eps8", "sgmf_eps16", "sgmf_nodp"]
M += tab("Table IX. Patient-level privacy: utility and membership inference (δ = 10⁻⁵).",
         ["Setting", "Achieved ε", "Noise multiplier σ", "AUROC", "Detection at 5% FPR", "MIA AUC", "MIA TPR at 1% FPR"],
         [[N[e], ("%.2f" % max(r["privacy"]["eps_max"] for r in runs[e])) if runs[e][0]["privacy"]["enabled"] else "—",
           ("%.2f–%.2f" % (min(runs[e][0]["privacy"]["sigma"].values()), max(runs[e][0]["privacy"]["sigma"].values())))
           if runs[e][0]["privacy"]["enabled"] else "—",
           c(e, "test.main.auroc"), c(e, "test_at_val_fpr.recall"), c(e, "mia.auc"), c(e, "mia.tpr_at_fpr_0.01")]
          for e in pr if e in runs],
         "Loss-threshold attack on 500 training patients vs. held-out patients of the training hospitals.")

Sx = ["---", "title: \"Supplementary Material\"", "---", "",
      "Supplementary material for *SG-MedFusion: Federated Cross-Modal Consistency Checking of Medical-Device Power "
      "Signatures and EHR Documentation — A Proof-of-Concept Study with a Patient-Level Privacy Analysis*. "
      "All values are generated by `scripts/paper_tables.py` from the released result files (three runs, mean ± SD)."]
Sx += tab("Table S1. Auxiliary forecasting 60 min ahead (consistent pairs).",
          ["Model", "HR MAE (bpm)", "HR RMSE (bpm)", "MAP MAE (mmHg)", "MAP RMSE (mmHg)"],
          [[N[e], c(e, "test.forecast.hr.mae", 2), c(e, "test.forecast.hr.rmse", 2), c(e, "test.forecast.map.mae", 2),
            c(e, "test.forecast.map.rmse", 2)] for e in ["sgmf_nodp", "concat_nodp", "sgmf_eps16", "sgmf_eps4"] if e in runs])
Sx += tab("Table S2. Single-modality shortcut probes (AUROC; 0.5 = no shortcut).",
          ["Probe", "All pairs", "Physically distinguishable pairs"],
          [[N[e], c(e, "test.main.auroc"), c(e, "test.distinguishable_only.auroc")]
           for e in ["probe_gbm_grid", "probe_gbm_ehr", "gbm_both"] if e in runs])
Sx += tab("Table S3. Distribution shift, federated SG-MedFusion without DP (AUROC).", ["Condition", "AUROC"],
          [["Background household %s" % h.replace("ukdale_", "UK-DALE house "), c("sgmf_nodp", "test.by_background.%s.auroc" % h)]
           for h in sorted(runs["sgmf_nodp"][0]["test"]["by_background"])]
          + [["Device power ×0.7", c("sgmf_nodp", "profile_shift.0.7.auroc")],
             ["Device power ×1.3", c("sgmf_nodp", "profile_shift.1.3.auroc")]])
rows = []
for g, vals in runs["sgmf_nodp"][0]["test"]["subgroups"].items():
    for v in vals:
        rows.append([{"gender": "Sex", "age_group": "Age group", "first_careunit": "ICU type"}[g], v, vals[v]["n"],
                     c("sgmf_nodp", "test.subgroups.%s.%s.auroc" % (g, v))])
Sx += tab("Table S4. Subgroup AUROC, federated SG-MedFusion without DP.", ["Attribute", "Group", "Test pairs (run 0)", "AUROC"], rows)
Sx += tab("Table S5. Model size and inference latency (forward pass only, one NVIDIA RTX 4000 Ada GPU).",
          ["Model", "Parameters", "Median latency, batch 1 (ms)", "Throughput, batch 256 (windows/s)"],
          [[N[e], runs[e][0]["n_params"], c(e, "latency.p50_ms", 2), c(e, "latency.throughput_per_s_batch256", 0)]
           for e in ["sgmf_nodp", "concat_nodp"] if e in runs])
a = json.load(open(root / "leakage_audit_seed0.json"))
Sx += tab("Table S6. Automated leakage audit (identical for all three runs).", ["Check", "Result"],
          [["Patients shared between splits", sum(a["patient_overlap"].values())],
           ["Hospitals shared between splits", sum(a["hospital_overlap"].values())],
           ["Hospitals per split (train / validation / test)", "%d / %d / %d" % (a["n_hospitals"]["train"], a["n_hospitals"]["val"], a["n_hospitals"]["test"])],
           ["Grid sources drawn from another split", a["cross_split_grid_sources"]],
           ["Wrong-patient pairs using the same patient", a["wrong_patient_same_patient"]],
           ["Background households shared between splits", sum(a["household_overlap"].values())],
           ["Overlapping windows within a stay", a["overlapping_windows_within_stay"]],
           ["Zero-filled power windows (sample of 200)", a["zero_filled_grid_windows_in_sample"]],
           ["Audit passed (all three runs)", all(json.load(open(f))["passed"] for f in sorted(root.glob("leakage_audit_seed*.json")))]])
Sx += tab("Table S7. Flag rates at the F1-maximising threshold (federated SG-MedFusion, no DP), including the replay stress set.",
          ["Pair type", "Flag rate"],
          [[lab.get(t, t), c("sgmf_nodp", "test.by_type.%s.flag_rate" % t)] for t in sorted(runs["sgmf_nodp"][0]["test"]["by_type"])]
          + [["Replayed telemetry (never trained)", c("sgmf_nodp", "test.replay.flag_rate")],
             ["Share of replays physically distinguishable", c("sgmf_nodp", "test.replay.share_distinguishable")]],
          "At this threshold 74% of consistent pairs are flagged, so the threshold is unsuitable for an audit queue (Section 7.2).")
env = runs["sgmf_nodp"][0]["environment"]
Sx += tab("Table S8. Computing environment.", ["Component", "Version"],
          [["GPU", "%s (%s GB)" % (env.get("gpu"), env.get("gpu_mem_gb"))], ["CUDA", env.get("cuda")],
           ["Python", env["python"]], ["PyTorch", env["torch"]], ["NumPy", env["numpy"]], ["scikit-learn", env["sklearn"]],
           ["Operating system", env["os"]]])
(out / "results_tables.md").write_text("\n".join(M) + "\n", encoding="utf-8")
(out / "supplementary.md").write_text("\n".join(Sx) + "\n", encoding="utf-8")
print("wrote", out / "results_tables.md", "and", out / "supplementary.md")
