"""Metrics, patient-clustered bootstrap, subgroup analysis, latency and a
loss-threshold membership-inference audit.

Every accuracy/F1 number is derived from the confusion matrix returned with
it, so tables and confusion-matrix figures cannot disagree.
"""
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .benchmark import collate


@torch.no_grad()
def predict(model, ds, dev, bs=256, cfg=None):
    model.eval()
    ys, ss, yh, tg, tm, ls = [], [], [], [], [], []
    for a in range(0, len(ds), bs):
        b = collate([ds[i] for i in range(a, min(a + bs, len(ds)))])
        b = {k: v.to(dev) for k, v in b.items()}
        out = model(b)
        ys.append(b["y"].cpu()); ss.append(torch.sigmoid(out["logit"]).cpu())
        yh.append(out["yhat"].cpu()); tg.append(b["ypred"].cpu()); tm.append(b["ypred_mask"].cpu())
        ls.append(torch.nn.functional.binary_cross_entropy_with_logits(
            out["logit"], b["y"], reduction="none").cpu())
    model.train()
    cat = lambda x: torch.cat(x).numpy()
    loss = cat(ls)
    return {"y": cat(ys), "score": cat(ss), "yhat": cat(yh), "target": cat(tg),
            "tmask": cat(tm), "sample_loss": loss, "loss": float(loss.mean())}


def auroc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def confusion(y, s, thr):
    p = s >= thr
    tp = int(np.sum(p & (y == 1))); fp = int(np.sum(p & (y == 0)))
    fn = int(np.sum(~p & (y == 1))); tn = int(np.sum(~p & (y == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def from_cm(cm):
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": (tp + tn) / n, "precision": prec, "recall": rec,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0}


def thr_max_f1(y, s):
    cands = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 197)))
    f = [from_cm(confusion(y, s, t))["f1"] for t in cands]
    return float(cands[int(np.argmax(f))])


def thr_at_fpr(y, s, fpr):
    neg = np.sort(s[y == 0])
    return float(neg[min(len(neg) - 1, int(np.ceil((1 - fpr) * len(neg))))])


def metrics(y, s, thr):
    cm = confusion(y, s, thr)
    out = {"auroc": auroc(y, s), "auprc": float(average_precision_score(y, s)),
           "threshold": thr, "confusion": cm, **from_cm(cm)}
    fpr, tpr, _ = roc_curve(y, s)
    out["fpr_at_tpr95"] = float(fpr[np.searchsorted(tpr, 0.95)])
    return out


def at_prevalence(y, s, thr, prev):
    """PPV and AUPRC re-weighted to a realistic prevalence of inconsistencies."""
    m = from_cm(confusion(y, s, thr))
    ppv = prev * m["recall"] / (prev * m["recall"] + (1 - prev) * m["fpr"] + 1e-12)
    w = np.where(y == 1, prev / max(y.mean(), 1e-12), (1 - prev) / max(1 - y.mean(), 1e-12))
    return {"prevalence": prev, "ppv": float(ppv),
            "auprc": float(average_precision_score(y, s, sample_weight=w))}


def cluster_bootstrap(groups, fn, n=1000, seed=0, *arrays):
    """Percentile CI resampling patients (clusters) with replacement."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        try:
            vals.append(fn(*[a[idx] for a in arrays]))
        except ValueError:
            continue
    vals = np.array(vals)
    return {"lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)),
            "n": int(len(vals))}


def paired_bootstrap_auroc(groups, y, s1, s2, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    d = []
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(roc_auc_score(y[idx], s1[idx]) - roc_auc_score(y[idx], s2[idx]))
    d = np.array(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"diff": float(roc_auc_score(y, s1) - roc_auc_score(y, s2)),
            "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "p": float(min(1.0, p))}


def full_report(rec, pv, thr, cfg, pv_replay=None, rec_replay=None):
    y, s = pv["y"], pv["score"]
    g = rec.subject_id.values
    ev = cfg["eval"]
    out = {"n": int(len(y)), "n_patients": int(len(np.unique(g))), "main": metrics(y, s, thr)}
    out["ci"] = {
        "auroc": cluster_bootstrap(g, auroc, ev["bootstrap"], 0, y, s),
        "accuracy": cluster_bootstrap(g, lambda a, b: from_cm(confusion(a, b, thr))["accuracy"],
                                      ev["bootstrap"], 1, y, s),
        "f1": cluster_bootstrap(g, lambda a, b: from_cm(confusion(a, b, thr))["f1"],
                                ev["bootstrap"], 2, y, s)}
    out["prevalence"] = [at_prevalence(y, s, thr, p) for p in ev["prevalences"]]
    # detection rate per inconsistency type, false-alert rate on consistent pairs
    out["by_type"] = {}
    for t, gi in rec.reset_index(drop=True).groupby("ntype").groups.items():
        gi = np.asarray(gi)
        out["by_type"][t] = {"n": int(len(gi)), "flag_rate": float((s[gi] >= thr).mean())}
    dist = rec.distinguishable.values
    keep = dist | (y == 0)
    out["distinguishable_only"] = {"n": int(keep.sum()), "auroc": auroc(y[keep], s[keep])}
    # subgroups
    out["subgroups"] = {}
    for col in ("gender", "age_group", "first_careunit"):
        out["subgroups"][col] = {}
        for val, gi in rec.reset_index(drop=True).groupby(col).groups.items():
            gi = np.asarray(gi)
            if min((y[gi] == 1).sum(), (y[gi] == 0).sum()) < 20:
                continue
            out["subgroups"][col][str(val)] = {
                "n": int(len(gi)), "auroc": auroc(y[gi], s[gi]),
                "ci": cluster_bootstrap(g[gi], auroc, 200, 3, y[gi], s[gi])}
    # forecasting (consistent pairs only, original units)
    c = y == 0
    units = {}
    for j, v in enumerate(("hr", "map")):
        m = c & (pv["tmask"][:, j] > 0)
        sd = cfg["_norms"]["vitals"][v][1]
        err = (pv["yhat"][m, j] - pv["target"][m, j]) * sd
        units[v] = {"n": int(m.sum()), "mae": float(np.abs(err).mean()) if m.any() else None,
                    "rmse": float(np.sqrt((err ** 2).mean())) if m.any() else None}
    out["forecast"] = units
    if pv_replay is not None:
        sr = pv_replay["score"]
        out["replay"] = {"n": int(len(sr)), "flag_rate": float((sr >= thr).mean()),
                         "flag_rate_distinguishable": float((sr[rec_replay.distinguishable.values] >= thr).mean())
                         if rec_replay.distinguishable.any() else None,
                         "share_distinguishable": float(rec_replay.distinguishable.mean())}
    return out


@torch.no_grad()
def latency(model, ds, dev, reps=200):
    model.eval()
    one = {k: v.to(dev) for k, v in collate([ds[0]]).items()}
    for _ in range(10):
        model(one)
    t = []
    for _ in range(reps):
        a = time.perf_counter(); model(one)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t.append((time.perf_counter() - a) * 1000)
    b = {k: v.to(dev) for k, v in collate([ds[i % len(ds)] for i in range(256)]).items()}
    a = time.perf_counter(); model(b)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    thr = 256 / (time.perf_counter() - a)
    model.train()
    return {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)),
            "throughput_per_s_batch256": float(thr), "device": str(dev),
            "note": "model forward only; excludes telemetry retrieval and signature checks"}


def membership_inference(member_rec, member_pv, non_rec, non_pv):
    """Yeom et al. (2018) loss-threshold attack at patient level: score =
    -mean loss of the patient's windows.  AUC 0.5 = no leakage detected."""
    def per_patient(rec, pv):
        df = rec[["subject_id"]].assign(l=pv["sample_loss"])
        return -df.groupby("subject_id").l.mean().values
    a, b = per_patient(member_rec, member_pv), per_patient(non_rec, non_pv)
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    s = np.r_[a, b]
    fpr, tpr, _ = roc_curve(y, s)
    return {"n_members": int(len(a)), "n_nonmembers": int(len(b)), "auc": auroc(y, s),
            "tpr_at_fpr_0.01": float(tpr[np.searchsorted(fpr, 0.01, side="right") - 1])}
