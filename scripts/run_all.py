"""Run every experiment of the revised manuscript for several seeds.

    python scripts/run_all.py --config my_paths.json --stage extract
    python scripts/run_all.py --config my_paths.json --stage run --seeds 0 1 2 3 4
    python scripts/make_report.py --results results

Each (experiment, seed) writes results/<exp>/seed<k>.json (+ scores .npz) and
the resolved config, so every table entry can be traced to a file.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sgm import baselines as BL  # noqa: E402
from sgm import benchmark as BM  # noqa: E402
from sgm import config as C  # noqa: E402
from sgm import data_grid as G  # noqa: E402
from sgm import data_mimic  # noqa: E402
from sgm import evaluate as E  # noqa: E402
from sgm import fed  # noqa: E402

A2 = {"fl": {"n_malicious": 2, "trim_k": 2}}
EXPERIMENTS = {
    # main model and privacy sweep
    "sgmf_eps4": {},
    "sgmf_eps1": {"dp": {"target_eps": 1.0}},
    "sgmf_eps2": {"dp": {"target_eps": 2.0}},
    "sgmf_eps8": {"dp": {"target_eps": 8.0}},
    "sgmf_nodp": {"dp": {"enabled": False}},
    # ablations (all at eps = 4)
    "abl_no_cycle": {"model": {"use_cycle": False}},
    "abl_concat": {"model": {"fusion": "concat"}},
    "abl_grid_only": {"model": {"use_ehr": False}},
    "abl_ehr_only": {"model": {"use_grid": False}},
    "abl_attn_noise": {"model": {"attn_noise_sigma": 1.0}},
    "abl_centralized_dp": {"fl": {"centralized": True}},
    # baselines
    "concat_eps1": {"model": {"fusion": "concat"}, "dp": {"target_eps": 1.0}},
    "concat_eps8": {"model": {"fusion": "concat"}, "dp": {"target_eps": 8.0}},
    "concat_nodp": {"model": {"fusion": "concat"}, "dp": {"enabled": False}},
    "fedprox_eps4": {"fl": {"prox_mu": 0.01}},
    "centralized_nodp": {"fl": {"centralized": True}, "dp": {"enabled": False}},
    "gbm_both": {"_sk": ("gbm", "both")},
    "lr_both": {"_sk": ("lr", "both")},
    "probe_gbm_grid": {"_sk": ("gbm", "grid")},
    "probe_gbm_ehr": {"_sk": ("gbm", "ehr")},
    # Byzantine robustness (2 of K sites malicious)
    "atk_signflip_fedavg": C.merge(A2, {"fl": {"attack": "signflip", "aggregator": "fedavg"}}),
    "atk_signflip_trim": C.merge(A2, {"fl": {"attack": "signflip", "aggregator": "trimmed_mean"}}),
    "atk_signflip_median": C.merge(A2, {"fl": {"attack": "signflip", "aggregator": "median"}}),
    "atk_labelflip_fedavg": C.merge(A2, {"fl": {"attack": "labelflip", "aggregator": "fedavg"}}),
    "atk_labelflip_trim": C.merge(A2, {"fl": {"attack": "labelflip", "aggregator": "trimmed_mean"}}),
    "atk_labelflip_median": C.merge(A2, {"fl": {"attack": "labelflip", "aggregator": "median"}}),
    "noatk_trim": {"fl": {"aggregator": "trimmed_mean", "trim_k": 2}},
    # eICU / 4-site plan: privacy sweep extended; architecture and robustness without DP
    "sgmf_eps16": {"dp": {"target_eps": 16.0}},
    "abl_no_cycle_nodp": {"model": {"use_cycle": False}, "dp": {"enabled": False}},
    "fedprox_nodp": {"fl": {"prox_mu": 0.01}, "dp": {"enabled": False}},
    "atk1_signflip_fedavg_nodp": {"fl": {"n_malicious": 1, "attack": "signflip", "aggregator": "fedavg"},
                                  "dp": {"enabled": False}},
    "atk1_signflip_trim_nodp": {"fl": {"n_malicious": 1, "attack": "signflip", "aggregator": "trimmed_mean",
                                       "trim_k": 1}, "dp": {"enabled": False}},
    "noatk_median": {"fl": {"aggregator": "median"}},
}
TIER1 = ["sgmf_eps4", "sgmf_eps1", "sgmf_eps8", "sgmf_nodp", "abl_concat", "concat_nodp",
         "abl_no_cycle", "abl_grid_only", "abl_ehr_only", "abl_centralized_dp", "centralized_nodp",
         "gbm_both", "lr_both", "probe_gbm_grid", "probe_gbm_ehr",
         "atk_signflip_fedavg", "atk_signflip_trim", "noatk_trim"]
# Minimal plan (GPU-light): the configurations the reviewers' points require.
MINIMAL = ["gbm_both", "lr_both", "probe_gbm_grid", "probe_gbm_ehr",          # CPU only, minutes
           "sgmf_nodp", "sgmf_eps1", "sgmf_eps4", "sgmf_eps8", "sgmf_eps16",  # privacy sweep (DP runs ~4 min)
           "concat_nodp", "abl_no_cycle_nodp", "fedprox_nodp",                # architecture / optimiser
           "atk1_signflip_fedavg_nodp", "atk1_signflip_trim_nodp"]             # robustness
MAIN = "sgmf_eps4"
MIA = {"sgmf_eps1", "sgmf_eps4", "sgmf_eps8", "sgmf_eps16", "sgmf_nodp"}
SHIFT = {"sgmf_nodp", "sgmf_eps4", "sgmf_eps16"}   # also evaluated under distribution shift


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def environment():
    import platform
    import sklearn
    env = {"python": platform.python_version(), "os": platform.platform(), "torch": torch.__version__,
           "numpy": np.__version__, "sklearn": sklearn.__version__, "cuda": torch.version.cuda,
           "cpu": platform.processor()}
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1)
    return env


def estimate(cfg, seeds, experiments):
    """Time a few real steps and extrapolate the wall time of the whole plan."""
    from sgm import dp as DP
    from sgm.model import SGMedFusion, loss_terms, sample_loss_fn
    tl, bgs, rec, norms = prepare(cfg, seeds[0])
    dev = torch.device(cfg["device"])
    ds = BM.PairDataset(rec, tl, bgs, cfg, norms)
    sites = fed.build_sites(rec, False)
    n_tr = int((rec.split == "train").sum())
    log("train pairs %d, patients %d, sites %s" % (n_tr, sum(s.n for s in sites), [s.n for s in sites]))
    model = SGMedFusion(cfg).to(dev).train()
    fl = cfg["fl"]
    s0 = max(sites, key=lambda s: s.n)
    rng = np.random.default_rng(0)

    def dp_batch():
        pick = rng.choice(s0.n, min(fl["patients_per_step"], s0.n), replace=False)
        idx, pidx = [], []
        for j, p in enumerate(pick):
            smp = s0.patients[p][:fl["samples_cap_per_patient"]]
            idx += list(smp); pidx += [j] * len(smp)
        b = {k: v.to(dev) for k, v in BM.collate([ds[int(i)] for i in idx]).items()}
        b.pop("idx")
        return b, torch.tensor(pidx, device=dev), len(pick)

    t0 = time.time()
    for _ in range(3):
        dp_batch()
    t_load = (time.time() - t0) / 3
    t0 = time.time()
    for _ in range(3):
        b, pidx, n = dp_batch()
        DP.dp_patient_step(model, sample_loss_fn(cfg), b, pidx, n, 1.0, 1.0, float(n))
        if dev.type == "cuda":
            torch.cuda.synchronize()
    t_dp = (time.time() - t0) / 3
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    t0 = time.time()
    for _ in range(3):
        b = {k: v.to(dev) for k, v in BM.collate([ds[int(i)] for i in s0.idx[:fl["batch_size"]]]).items()}
        loss = loss_terms(model(b), b, cfg)[0].mean(); opt.zero_grad(); loss.backward(); opt.step()
    t_plain = (time.time() - t0) / 3
    dp_steps = fl["rounds"] * fl["local_epochs"] * sum(max(1, round(s.n / fl["patients_per_step"])) for s in sites)
    plain_steps = fl["rounds"] * fl["local_epochs"] * int(np.ceil(n_tr / fl["batch_size"]))
    h_dp, h_plain = dp_steps * t_dp / 3600, plain_steps * t_plain / 3600
    total = 0.0
    for e in experiments:
        over = EXPERIMENTS[e]
        if "_sk" in over:
            h = 0.3
        elif over.get("dp", {}).get("enabled") is False:
            h = h_plain
        else:
            h = h_dp
        total += h * len(seeds)
    log("per step: load %.2fs, DP step %.2fs, plain step %.2fs" % (t_load, t_dp, t_plain))
    log("one DP run ~ %.1f h, one non-DP run ~ %.1f h (+ evaluation)" % (h_dp, h_plain))
    log("PLAN: %d experiments x %d seeds ~ %.0f hours" % (len(experiments), len(seeds), total))
    log("If too long: set \"max_stays\" in paths.json (time scales ~linearly with stays), "
        "or run fewer seeds for Tier 2 experiments.")


def prepare(cfg, seed):
    work = Path(cfg["paths"]["work_dir"])
    cache = work / ("bench_seed%d.pkl" % seed)
    tl = data_mimic.load(cfg)
    names = sorted(set(sum(cfg["households"].values(), [])))
    bgs = G.load_backgrounds(cfg, names)
    if cache.exists():
        with open(cache, "rb") as f:
            rec, norms, audit = pickle.load(f)
    else:
        rec, _ = BM.build_recipes(tl, cfg, bgs, seed)
        norms = BM.fit_norms(tl, rec, bgs, cfg, seed=seed)
        test = rec[rec.split == "test"]
        audit = BM.leakage_audit(rec, cfg, bgs, BM.PairDataset(test.sample(min(200, len(test)), random_state=0),
                                                              tl, bgs, cfg, norms))
        with open(cache, "wb") as f:
            pickle.dump((rec, norms, audit), f)
    BM.save_json(audit, Path(cfg["paths"]["results_dir"]) / ("leakage_audit_seed%d.json" % seed))
    if not audit["passed"]:
        raise RuntimeError("leakage audit failed, see leakage_audit_seed%d.json" % seed)
    return tl, bgs, rec.reset_index(drop=True), norms


def run_one(base, exp, seed, data):
    tl, bgs, rec, norms = data
    over = dict(EXPERIMENTS[exp])
    sk = over.pop("_sk", None)
    cfg = C.merge(base, over)
    cfg["seed"] = seed
    cfg["_norms"] = norms
    out_dir = Path(cfg["paths"]["results_dir"]) / exp
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(cfg["device"])
    sub = lambda sp: rec[rec.split == sp].reset_index(drop=True)
    mk = lambda r, scale=1.0: BM.PairDataset(r, tl, bgs, cfg, norms, scale)
    r_val, r_test, r_rep = sub("val"), sub("test"), sub("test_replay")
    ds_val, ds_test, ds_rep = mk(r_val), mk(r_test), mk(r_rep)
    res = {"experiment": exp, "seed": seed, "config": {k: v for k, v in cfg.items() if k != "_norms"},
           "environment": environment(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    t0 = time.time()
    if sk:
        kind, which = sk
        r_tr = sub("train")
        r_tr = r_tr.sample(min(len(r_tr), 50_000), random_state=seed).reset_index(drop=True)
        X = [BL.featurise(mk(r), which, cfg["grid_hz"]) for r in (r_tr, r_val, r_test, r_rep)]
        s_val, s_test, s_rep = BL.fit_predict(kind, X[0], r_tr.y.values, X[1:], seed)
        pv_val = {"y": r_val.y.values.astype(float), "score": s_val}
        pv_test = {"y": r_test.y.values.astype(float), "score": s_test,
                   "yhat": np.zeros((len(r_test), 2)), "target": np.zeros((len(r_test), 2)),
                   "tmask": np.zeros((len(r_test), 2))}
        pv_rep = {"score": s_rep}
        res["privacy"] = {"enabled": False, "note": "centralised, non-private reference"}
    else:
        ds_all = mk(rec)
        ckpt = out_dir / ("seed%d.ckpt" % seed)
        model, info = fed.train(cfg, ds_all, rec, ds_val, log, ckpt=ckpt)
        res.update(info)
        pv_val, pv_test, pv_rep = (E.predict(model, d, dev) for d in (ds_val, ds_test, ds_rep))
        res["latency"] = E.latency(model, ds_test, dev, cfg["eval"]["latency_reps"])
        res["n_params"] = int(sum(p.numel() for p in model.parameters()))
        if exp in SHIFT:
            res["profile_shift"] = {}
            for sc in (0.7, 1.3):
                pv = E.predict(model, mk(r_test, sc), dev)
                res["profile_shift"][str(sc)] = {"auroc": E.auroc(pv["y"], pv["score"])}
        if exp in MIA:
            rng = np.random.default_rng(seed)
            tr = sub("train")
            pick = rng.choice(tr.subject_id.unique(), min(cfg["eval"]["mia_patients"], tr.subject_id.nunique()), replace=False)
            r_mem = tr[tr.subject_id.isin(pick)].reset_index(drop=True)
            r_non = sub("mia_holdout")
            res["mia"] = E.membership_inference(r_mem, E.predict(model, mk(r_mem), dev),
                                                r_non, E.predict(model, mk(r_non), dev))
    thr = E.thr_max_f1(pv_val["y"], pv_val["score"])
    thr_f = E.thr_at_fpr(pv_val["y"], pv_val["score"], cfg["eval"]["target_fpr"])
    res["val_auroc"] = E.auroc(pv_val["y"], pv_val["score"])
    res["test"] = E.full_report(r_test, pv_test, thr, cfg, pv_rep, r_rep)
    res["test_at_val_fpr"] = E.metrics(pv_test["y"], pv_test["score"], thr_f)
    res["test"]["by_background"] = {          # one entry per test household
        hh: {"auroc": E.auroc(pv_test["y"][m], pv_test["score"][m]), "n": int(m.sum())}
        for hh in sorted(r_test.hh.unique()) for m in [(r_test.hh == hh).values]}
    res["wall_time_s"] = time.time() - t0
    with open(out_dir / ("seed%d.json" % seed), "w") as f:
        json.dump(res, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    np.savez_compressed(out_dir / ("seed%d_scores.npz" % seed), y=pv_test["y"], score=pv_test["score"],
                        subject_id=r_test.subject_id.values, ntype=r_test.ntype.values.astype(str))
    for f in out_dir.glob("seed%d.ckpt*" % seed):     # run complete: progress file no longer needed
        f.unlink()
    log("%s seed %d: test AUROC %.4f acc %.4f (%.0f s)" % (exp, seed, res["test"]["main"]["auroc"],
                                                           res["test"]["main"]["accuracy"], res["wall_time_s"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--stage", choices=["extract", "estimate", "run"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))
    ap.add_argument("--tier", type=int, choices=[1, 2], help="run only Tier 1 or Tier 2 experiments")
    ap.add_argument("--plan", choices=["minimal", "full"], help="minimal = 12 configurations (8 on GPU)")
    a = ap.parse_args()
    cfg = C.load(a.config)
    if a.plan == "minimal":
        a.experiments = list(MINIMAL)
    if a.tier:
        a.experiments = [e for e in EXPERIMENTS if (e in TIER1) == (a.tier == 1)]
    if a.stage == "extract":
        if cfg["dataset"] == "eicu":
            from sgm import data_eicu
            data_eicu.extract(cfg, log)
        else:
            data_mimic.extract(cfg, log)
        return
    if a.stage == "estimate":
        log("environment: %s" % environment())
        estimate(cfg, a.seeds, a.experiments)
        return
    for seed in a.seeds:
        data = prepare(cfg, seed)
        for exp in a.experiments:
            f = Path(cfg["paths"]["results_dir"]) / exp / ("seed%d.json" % seed)
            if f.exists() and exp in SHIFT and "profile_shift" not in json.load(open(f)):
                log("re-running %s seed %d to add the distribution-shift evaluation" % (exp, seed))
            elif f.exists():
                log("skip %s seed %d (done)" % (exp, seed))
                continue
            run_one(cfg, exp, seed, data)


if __name__ == "__main__":
    main()
