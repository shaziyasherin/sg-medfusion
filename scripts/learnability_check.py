"""Sanity check on the synthetic TEST FIXTURE: can the model learn the
consistency task at all, and do the single-modality probes stay near chance?
Output is a pipeline check, never a paper result.

    python scripts/learnability_check.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_all  # noqa: E402
from sgm import config as C, data_mimic, synthetic  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    rng = np.random.default_rng(1)
    mimic = synthetic.write_mimic(Path(tmp) / "mimic", rng, n_patients=int(sys.argv[3]) if len(sys.argv) > 3 else 240)
    redd, ukdale = synthetic.write_nilm(Path(tmp) / "nilm", rng, hours=4)
    cfg = C.load(None, paths={"mimic_dir": mimic, "redd_dir": redd, "ukdale_dir": ukdale,
                              "work_dir": str(Path(tmp) / "w"), "results_dir": str(Path(tmp) / "r")},
                 window_minutes=10, windows_per_stay=6, n_sites=4, min_edit_minutes=3,
                 shift_range_min=[15, 120], replay_min_gap_min=600,
                 model={"d": 32, "heads": 2, "sa_layers": 1},
                 fl={"rounds": int(sys.argv[1]) if len(sys.argv) > 1 else 15, "eval_every": 5,
                     "patients_per_step": 16, "samples_cap_per_patient": 4, "batch_size": 64, "lr": 2e-3},
                 eval={"bootstrap": 100, "latency_reps": 20, "mia_patients": 50})
    data_mimic.extract(cfg, log=lambda *_: None)
    data = run_all.prepare(cfg, 0)
    out = {}
    exps = sys.argv[2].split(",") if len(sys.argv) > 2 else [
        "centralized_nodp", "sgmf_nodp", "sgmf_eps4", "abl_ehr_only",
        "probe_gbm_grid", "probe_gbm_ehr", "gbm_both"]
    for exp in exps:
        run_all.run_one(cfg, exp, 0, data)
        r = json.load(open(Path(cfg["paths"]["results_dir"]) / exp / "seed0.json"))
        out[exp] = {"auroc": round(r["test"]["main"]["auroc"], 3),
                    "auroc_distinguishable": round(r["test"]["distinguishable_only"]["auroc"], 3),
                    "val_curve": [round(h["val_auroc"], 3) for h in r.get("history", [])],
                    "by_type": {k: round(v["flag_rate"], 2) for k, v in r["test"]["by_type"].items()}}
    print(json.dumps(out, indent=1))
