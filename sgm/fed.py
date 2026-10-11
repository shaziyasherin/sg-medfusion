"""Federated training: FedAvg / FedProx with patient-level DP-SGD at each site,
optional Byzantine-robust aggregation (coordinate-wise trimmed mean / median)
and simulated malicious sites.

Secure aggregation (additive masking, Bonawitz et al. 2017) is a deployment
component and is not simulated cryptographically here.  It is only compatible
with the FedAvg aggregator; the robust aggregators need to see individual
updates, which remain patient-level DP because every update is itself the
output of DP-SGD (post-processing).
"""
import copy
import os
from pathlib import Path

import numpy as np
import torch

from . import dp as DP
from .benchmark import collate
from .evaluate import predict, auroc
from .model import SGMedFusion, loss_terms, sample_loss_fn


def _params(model):
    return {k: v for k, v in model.named_parameters() if v.requires_grad}


def aggregate(deltas, weights, how, trim_k):
    if how == "trimmed_mean" and 2 * trim_k >= len(deltas):
        raise ValueError("trimmed mean needs 2*trim_k < number of sites (%d)" % len(deltas))
    out = {}
    for k in deltas[0]:
        s = torch.stack([d[k] for d in deltas])
        if how == "fedavg":
            w = torch.tensor(weights, dtype=s.dtype, device=s.device).view(-1, *[1] * (s.dim() - 1))
            out[k] = (w * s).sum(0)
        elif how == "trimmed_mean":
            srt = s.sort(0).values
            out[k] = srt[trim_k: len(deltas) - trim_k].mean(0)
        elif how == "median":
            out[k] = s.median(0).values
        else:
            raise ValueError(how)
    return out


class Site:
    def __init__(self, sid, rec_idx, subjects):
        self.sid = sid
        self.idx = np.asarray(rec_idx)
        by = {}
        for i, s in zip(self.idx, subjects):
            by.setdefault(s, []).append(i)
        self.patients = list(by.values())
        self.n = len(self.patients)


def build_sites(rec, centralized):
    tr = rec.reset_index(drop=True)
    tr = tr[tr.split == "train"]
    if centralized:
        return [Site(0, tr.index.values, tr.subject_id.values)]
    return [Site(int(s), g.index.values, g.subject_id.values) for s, g in tr.groupby("site")]


def train(cfg, ds, rec, val_ds, log=print, ckpt=None):
    fl, dpc = cfg["fl"], cfg["dp"]
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    dev = torch.device(cfg["device"])
    model = SGMedFusion(cfg).to(dev)
    sites = build_sites(rec, fl["centralized"])
    N = sum(s.n for s in sites)
    malicious = set(rng.choice([s.sid for s in sites], fl["n_malicious"], replace=False).tolist()) \
        if fl["n_malicious"] else set()
    acct = DP.Accountant(dpc["delta"])
    sigma = {}
    for s in sites:
        q = min(1.0, fl["patients_per_step"] / s.n)
        steps = fl["rounds"] * fl["local_epochs"] * max(1, int(round(1 / q)))
        s.q, s.steps_per_epoch = q, max(1, int(round(1 / q)))
        if dpc["enabled"]:
            sigma[s.sid] = dpc["noise_multiplier"] or DP.calibrate_sigma(q, steps, dpc["target_eps"], dpc["delta"])
    if dpc["enabled"]:
        log("DP: " + ", ".join("site %d: n=%d q=%.4f sigma=%.3f" % (s.sid, s.n, s.q, sigma[s.sid]) for s in sites))
    history, best, start = [], (-1.0, None, -1), 0
    import json as _json
    sig = _json.dumps({k: cfg[k] for k in ("fl", "dp", "model", "loss", "seed")}, sort_keys=True, default=str)
    if ckpt is not None and Path(ckpt).exists():          # resume an interrupted run
        st = torch.load(ckpt, map_location="cpu", weights_only=False)  # RNG states must stay on CPU
        if st.get("sig") != sig:
            log("checkpoint was made with different settings - starting this run fresh")
            Path(ckpt).unlink()
            st = None
    else:
        st = None
    if st is not None:
        model.load_state_dict(st["model"])
        history, best, start = st["history"], tuple(st["best"]), st["round"]
        acct.rdp = st["rdp"]
        rng.bit_generator.state = st["np_rng"]
        torch.set_rng_state(st["torch_rng"])
        if st.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        log("resumed from round %d" % start)
    to = lambda b: {k: v.to(dev) for k, v in b.items()}
    for r in range(start, fl["rounds"]):
        g0 = {k: v.detach().clone() for k, v in _params(model).items()}
        deltas, weights, tl = [], [], []
        for s in sites:
            local = copy.deepcopy(model).train()
            opt = torch.optim.AdamW(local.parameters(), lr=fl["lr"], weight_decay=fl["weight_decay"])
            bad = s.sid in malicious
            lf = sample_loss_fn(cfg, flip_labels=bad and fl["attack"] == "labelflip")
            for _ in range(fl["local_epochs"]):
                if dpc["enabled"]:
                    for _ in range(s.steps_per_epoch):
                        pick = np.where(rng.random(s.n) < s.q)[0]
                        idx, pidx = [], []
                        for j, p in enumerate(pick):
                            smp = s.patients[p]
                            if len(smp) > fl["samples_cap_per_patient"]:
                                smp = rng.choice(smp, fl["samples_cap_per_patient"], replace=False)
                            idx += list(smp)
                            pidx += [j] * len(smp)
                        if idx:
                            b = to(collate([ds[int(i)] for i in idx]))
                            b.pop("idx")
                        else:
                            b = None
                        g = DP.dp_patient_step(local, lf, b, torch.tensor(pidx, dtype=torch.long, device=dev),
                                               len(pick), dpc["clip"], sigma[s.sid], s.q * s.n)
                        for k, p in _params(local).items():
                            p.grad = g[k]
                            if fl["prox_mu"] > 0:      # data-independent term, added after noise
                                p.grad = p.grad + fl["prox_mu"] * (p.detach() - g0[k])
                        opt.step()
                        acct.step(s.sid, s.q, sigma[s.sid])
                else:
                    order = rng.permutation(s.idx)
                    for a in range(0, len(order), fl["batch_size"]):
                        b = to(collate([ds[int(i)] for i in order[a:a + fl["batch_size"]]]))
                        out = local(b)
                        if bad and fl["attack"] == "labelflip":
                            b = dict(b, y=1 - b["y"])
                        loss = loss_terms(out, b, cfg)[0].mean()
                        if fl["prox_mu"] > 0:
                            loss = loss + fl["prox_mu"] / 2 * sum(
                                (p - g0[k]).pow(2).sum() for k, p in _params(local).items())
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        tl.append(float(loss))
            d = {k: v.detach() - g0[k] for k, v in _params(local).items()}
            if bad and fl["attack"] == "signflip":
                d = {k: -fl["attack_scale"] * v for k, v in d.items()}
            deltas.append(d)
            weights.append(s.n / N)
        agg = aggregate(deltas, weights, fl["aggregator"], fl["trim_k"])
        with torch.no_grad():
            for k, p in _params(model).items():
                p.add_(agg[k])
        if (r + 1) % fl["eval_every"] == 0 or r == fl["rounds"] - 1:
            pv = predict(model, val_ds, dev)
            a = auroc(pv["y"], pv["score"])
            eps = acct.spent()[0] if dpc["enabled"] else float("inf")
            history.append({"round": r + 1, "val_auroc": a, "val_loss": pv["loss"],
                            "train_loss": float(np.mean(tl)) if tl else None, "eps": eps})
            log("round %3d  val AUROC %.4f  val loss %.4f  eps %.3f" % (r + 1, a, pv["loss"], eps))
            if a > best[0]:
                best = (a, copy.deepcopy(model.state_dict()), r + 1)
            if ckpt is not None:                           # progress is never lost beyond eval_every rounds
                tmp = str(ckpt) + ".tmp"
                torch.save({"model": model.state_dict(), "history": history, "best": list(best),
                            "round": r + 1, "rdp": acct.rdp, "sig": sig, "np_rng": rng.bit_generator.state,
                            "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                           tmp)
                os.replace(tmp, ckpt)
            if cfg.get("_stop_after_round") == r + 1:     # test hook: simulate an interruption
                break
    model.load_state_dict(best[1])
    priv = {"enabled": dpc["enabled"], "delta": dpc["delta"], "clip": dpc["clip"],
            "sigma": sigma, "q": {s.sid: s.q for s in sites},
            "steps": {s.sid: fl["rounds"] * fl["local_epochs"] * s.steps_per_epoch for s in sites}}
    if dpc["enabled"]:
        priv["eps_max"], priv["eps_per_site"] = acct.spent()
    return model, {"history": history, "best_round": best[2], "privacy": priv,
                   "malicious_sites": sorted(malicious),
                   "sites": {s.sid: s.n for s in sites}}
