"""Semi-synthetic paired benchmark with leakage-controlled splits.

A sample is a (grid window, EHR window) pair for one ICU stay at request time
tau.  Label y = 1 means the documented device use in the EHR window is NOT
consistent with the physical device activity implied by the grid window.

Consistent pair (y=0): grid synthesised from the stay's own documented device
intervals at tau.
Inconsistent pairs (y=1), one per consistent pair, type drawn uniformly among
those applicable:
  shift          grid from the same stay at tau +/- Delta (mistimed record)
  wrong_patient  grid from another patient's bed, same split (wrong-patient record)
  fabricate      EHR gains a device interval with no physical counterpart
  omit           EHR loses a documented device interval that is physically present
Stress set (test only, never trained on):
  replay         grid is an older genuine window of the same stay (>= 24 h earlier)

No zero-filled or empty grid windows are ever generated.
Splits are patient-disjoint; the test split is temporally later (MIMIC-IV
anchor_year_group); background households are disjoint across splits.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data_grid as G

VITALS = ["hr", "map", "spo2", "rr"]
DEVS = ["vent", "niv", "crrt", "inf"]
DIFF_SCALE_W = 5.0
TYPE_ID = {"consistent": 0, "shift": 1, "wrong_patient": 2, "fabricate": 3, "omit": 4, "replay": 5}


# --------------------------------------------------------------------------- splits
def make_splits(meta, cfg, rng):
    if cfg.get("split_mode") == "hospital":
        return make_hospital_splits(meta, cfg, rng)
    first = meta.sort_values("stay_id").groupby("subject_id").first()
    split = pd.Series("train", index=first.index)
    split[first.anchor_year_group.isin(cfg["test_year_groups"])] = "test"
    rest = np.array(split.index[split == "train"])
    rng.shuffle(rest)
    n_val = int(round(cfg["val_frac"] * len(rest)))
    n_mia = int(round(cfg["mia_frac"] * len(rest)))
    split[rest[:n_val]] = "val"
    split[rest[n_val:n_val + n_mia]] = "mia_holdout"
    # sites = ICU type of the patient's first stay (non-IID by construction)
    tr = first.loc[split == "train", "first_careunit"]
    units = tr.value_counts().index.tolist()
    K = min(cfg["n_sites"], len(units))
    unit_site = {u: min(i, K - 1) for i, u in enumerate(units)}
    site = first.first_careunit.map(unit_site).fillna(-1).astype(int)
    site[split != "train"] = -1
    return split.to_dict(), site.to_dict(), unit_site


def make_hospital_splits(meta, cfg, rng):
    """Multi-hospital data (eICU): whole hospitals are held out for test and
    validation; each patient follows the hospital of their first stay.  The
    remaining hospitals are dealt round-robin (largest first) to K federated
    sites, so a site is a group of hospitals, like a health system."""
    first = meta.sort_values("stay_id").groupby("subject_id").first()
    hosp = np.array(sorted(first.hospitalid.unique()))
    rng.shuffle(hosp)
    n_test = int(round(cfg["test_hospital_frac"] * len(hosp)))
    n_val = int(round(cfg["val_hospital_frac"] * len(hosp)))
    h_split = {h: "test" for h in hosp[:n_test]}
    h_split.update({h: "val" for h in hosp[n_test:n_test + n_val]})
    split = first.hospitalid.map(lambda h: h_split.get(h, "train"))
    rest = np.array(split.index[split == "train"])
    rng.shuffle(rest)
    split[rest[:int(round(cfg["mia_frac"] * len(rest)))]] = "mia_holdout"
    train_h = first.loc[split == "train"].hospitalid.value_counts().index.tolist()
    K = min(cfg["n_sites"], len(train_h))
    h_site = {h: i % K for i, h in enumerate(train_h)}
    site = first.hospitalid.map(h_site).fillna(-1).astype(int)
    site[split != "train"] = -1
    return split.to_dict(), site.to_dict(), {"site_of_hospital": h_site, "split_of_hospital": h_split}


# --------------------------------------------------------------------------- helpers
def doc_states(intervals, tau, W):
    return G.activity(intervals, tau - W, W)


def diff_minutes(a, b):
    return int(sum(np.sum(a[d] != b[d]) for d in DEVS))


def _device_window(intervals, tau, W):
    st = doc_states(intervals, tau, W)
    return any(st[d].any() for d in DEVS)


# --------------------------------------------------------------------------- recipes
def build_recipes(tl, cfg, backgrounds, seed):
    rng = np.random.default_rng(seed)
    W, hz = cfg["window_minutes"], cfg["grid_hz"]
    L = W * 60 * hz
    meta = tl["meta"]
    # patient splits are fixed across runs; negatives, noise and init vary with `seed`
    split, site, unit_site = make_splits(meta, cfg, np.random.default_rng(cfg.get("split_seed", 0)))
    meta = meta.assign(split=meta.subject_id.map(split), site=meta.subject_id.map(site))
    if cfg.get("split_mode") == "hospital":
        # a stay counts only if its hospital belongs to the patient's split
        hs = meta.hospitalid.map(lambda h: unit_site["split_of_hospital"].get(h, "train"))
        ok = (hs == meta.split) | ((hs == "train") & (meta.split == "mia_holdout"))
        meta = meta[ok].reset_index(drop=True)
    starts = {h: G.valid_starts(p, L) for h, p in backgrounds.items()}
    for h, s in starts.items():
        if len(s) == 0:
            raise RuntimeError("household %s has no complete %d-s window" % (h, L))
    hh_of = {}
    for sp in ("train", "val", "test"):
        for h in cfg["households"][sp]:
            hh_of.setdefault(sp, []).append(h)
    hh_of["mia_holdout"] = hh_of["train"]

    def bg(sp):
        h = hh_of[sp][rng.integers(len(hh_of[sp]))]
        return h, int(starts[h][rng.integers(len(starts[h]))])

    stays_by_split = {sp: g.stay_id.values for sp, g in meta.groupby("split")}
    from types import SimpleNamespace
    info = {k: SimpleNamespace(**v) for k, v in meta.set_index("stay_id").to_dict("index").items()}
    rows = []

    def row(sid, tau, y, ntype, src, src_tau, edit=(None, 0), dist=True):
        r = info[sid]
        h, off = bg(r.split)
        rows.append(dict(stay_id=int(sid), subject_id=int(r.subject_id), split=r.split,
                         site=int(r.site), tau=int(tau), y=int(y), ntype=ntype,
                         src_stay=int(src), src_tau=int(src_tau), edit_dev=edit[0],
                         edit_sign=int(edit[1]), hh=h, bg_off=off,
                         noise_seed=int(rng.integers(2 ** 31)), distinguishable=bool(dist)))

    lo, hi = cfg["shift_range_min"]
    for sid, r in info.items():
        iv = tl["intervals"][sid]
        cands = np.arange(W, r.los_min + 1, W)       # non-overlapping windows
        if len(cands) == 0:
            continue
        segs = np.array([s for d in DEVS for s in iv.get(d, [])], dtype=np.int64).reshape(-1, 2)
        # a device is on somewhere in [t-W, t)  <=>  start < t and end > t-W
        on = (((segs[:, 0][None, :] < cands[:, None]) & (segs[:, 1][None, :] > cands[:, None] - W)).any(1)
              if len(segs) else np.zeros(len(cands), bool))
        k = min(cfg["windows_per_stay"], len(cands))
        k_on = min(int(round(cfg["device_window_frac"] * k)), on.sum())
        pick = list(rng.choice(cands[on], k_on, replace=False)) if k_on else []
        rest = np.setdiff1d(cands, pick)
        pick += list(rng.choice(rest, min(k - k_on, len(rest)), replace=False))
        for tau in pick:
            tau = int(tau)
            own = doc_states(iv, tau, W)
            row(sid, tau, 0, "consistent", sid, tau)
            options = []
            for _ in range(10):
                d = int(rng.integers(lo, hi + 1)) * (1 if rng.random() < .5 else -1)
                if W <= tau + d <= r.los_min:
                    options.append(("shift", sid, tau + d))
                    break
            others = stays_by_split[r.split]
            for _ in range(10):
                o = int(others[rng.integers(len(others))])
                if info[o].subject_id != r.subject_id and info[o].los_min >= W:
                    options.append(("wrong_patient", o, int(rng.integers(W, info[o].los_min + 1))))
                    break
            options.append(("fabricate", sid, tau))
            if any(own[d].sum() >= cfg["min_edit_minutes"] for d in DEVS):
                options.append(("omit", sid, tau))
            nt, src, st = options[rng.integers(len(options))]
            edit = (None, 0)
            if nt == "fabricate":
                edit = (DEVS[rng.integers(len(DEVS))], +1)
            elif nt == "omit":
                ds = [d for d in DEVS if own[d].sum() >= cfg["min_edit_minutes"]]
                edit = (ds[rng.integers(len(ds))], -1)
            dist = True
            if nt in ("shift", "wrong_patient"):
                dist = diff_minutes(own, doc_states(tl["intervals"][src], st, W)) >= cfg["min_edit_minutes"]
            row(sid, tau, 1, nt, src, st, edit, dist)
            if r.split == "test" and tau - cfg["replay_min_gap_min"] >= W:
                g = int(rng.integers(cfg["replay_min_gap_min"], tau - W + 1))
                rp = doc_states(iv, tau - g, W)
                row(sid, tau, 1, "replay", sid, tau - g, dist=diff_minutes(own, rp) >= cfg["min_edit_minutes"])
    rec = pd.DataFrame(rows)
    rec.loc[rec.ntype == "replay", "split"] = "test_replay"
    keep_cols = ["stay_id", "gender", "age", "first_careunit", "anchor_year_group"] +         (["hospitalid"] if "hospitalid" in meta.columns else [])
    rec = rec.merge(meta[keep_cols],
                    on="stay_id", how="left")
    rec["age_group"] = pd.cut(rec.age, [17, 44, 64, 79, 200],
                              labels=["18-44", "45-64", "65-79", "80+"]).astype(str)
    return rec, unit_site


# --------------------------------------------------------------------------- fabricate / omit
def apply_edit(states, dev, sign, rng, min_len):
    st = {d: v.copy() for d, v in states.items()}
    W = len(st[dev])
    if sign > 0:
        if dev == "inf":
            a = int(rng.integers(0, W - min_len + 1)); b = int(rng.integers(a + min_len, W + 1))
            st[dev][a:b] = np.minimum(st[dev][a:b] + 1, G.MAX_PUMPS)
        else:
            off = np.where(st[dev] == 0)[0]
            if len(off) < min_len:           # device on all window: fabricate a pump instead
                return apply_edit(states, "inf", +1, rng, min_len)
            a = int(off[rng.integers(0, len(off) - min_len + 1)])
            b = min(W, a + int(rng.integers(min_len, W + 1)))
            seg = st[dev][a:b]
            seg[seg == 0] = 1
    else:
        st[dev] = np.maximum(st[dev] - 1, 0) if dev == "inf" else np.zeros_like(st[dev])
    return st


# --------------------------------------------------------------------------- normalisation
def fit_norms(tl, rec, backgrounds, cfg, n=2000, seed=0):
    """Normalisation constants are fitted on the VALIDATION split so that no
    non-private statistic of training patients enters the model (the
    patient-level DP guarantee then covers everything derived from training
    data)."""
    tr = rec[rec.split == "val"]
    stays = tr.stay_id.unique()
    vit = {}
    for v in VITALS:
        vals = np.concatenate([tl["vitals"][s][v][1] for s in stays if v in tl["vitals"][s]] or [np.zeros(1)])
        vit[v] = (float(np.mean(vals)), float(np.std(vals) + 1e-6))
    ds = PairDataset(tr.sample(min(n, len(tr)), random_state=seed),
                     tl, backgrounds, cfg, {"vitals": vit, "grid": (0.0, 1.0)})
    lp = np.concatenate([ds.raw_grid(i) for i in range(len(ds))])
    return {"vitals": vit, "grid": (float(lp.mean()), float(lp.std() + 1e-6))}


# --------------------------------------------------------------------------- dataset
class PairDataset(torch.utils.data.Dataset):
    def __init__(self, rec, tl, backgrounds, cfg, norms, profile_scale=1.0):
        self.rec = rec.reset_index(drop=True)
        self.tl, self.bg, self.cfg, self.norms = tl, backgrounds, cfg, norms
        self.W, self.hz = cfg["window_minutes"], cfg["grid_hz"]
        self.scale = profile_scale
        self.cols = {c: self.rec[c].values for c in self.rec.columns}

    def __len__(self):
        return len(self.rec)

    def _c(self, c, i):
        return self.cols[c][i]

    def raw_power(self, i):
        W, hz = self.W, self.hz
        L = W * 60 * hz
        src, st = self._c("src_stay", i), self._c("src_tau", i)
        rng = np.random.default_rng(self._c("noise_seed", i))
        states = doc_states(self.tl["intervals"][src], st, W)
        p = self.bg[self._c("hh", i)][self._c("bg_off", i): self._c("bg_off", i) + L]
        return np.maximum(p + G.device_power(src, states, hz, rng, self.scale), 0.0)

    def raw_grid(self, i):
        return np.log1p(self.raw_power(i))

    def _vitals(self, sid, tau):
        W, lb = self.W, self.cfg["lookback_minutes"]
        mins = np.arange(tau - W, tau)
        val = np.zeros((W, 4), np.float32)
        msk = np.zeros((W, 4), np.float32)
        tgt = np.zeros(2, np.float32)
        tmask = np.zeros(2, np.float32)
        for j, v in enumerate(VITALS):
            s = self.tl["vitals"][sid].get(v)
            if s is None:
                continue
            t, x = s
            mu, sd = self.norms["vitals"][v]
            idx = np.searchsorted(t, mins, side="right") - 1
            ok = (idx >= 0) & (mins - t[np.maximum(idx, 0)] <= lb)
            val[ok, j] = (x[idx[ok]] - mu) / sd
            msk[ok, j] = 1
            if v in ("hr", "map"):
                k = VITALS.index(v) if v == "hr" else 1
                fut = np.where((t > tau) & (t <= tau + self.cfg["horizon_minutes"]))[0]
                if len(fut):
                    tgt[k] = (x[fut[-1]] - mu) / sd
                    tmask[k] = 1
        return val, msk, tgt, tmask

    def __getitem__(self, i):
        sid, tau = self._c("stay_id", i), self._c("tau", i)
        mu, sd = self.norms["grid"]
        p = self.raw_power(i)
        # channel 0: standardised log power (level); channel 1: compressed first
        # difference in watts (on/off steps and breath/pump modulation, which the
        # slow background dominates in channel 0)
        diff = np.arcsinh(np.diff(p, prepend=p[0]) / DIFF_SCALE_W)
        xg = np.stack([(np.log1p(p) - mu) / sd, diff], 1).astype(np.float32)
        states = doc_states(self.tl["intervals"][sid], tau, self.W)
        if self._c("edit_sign", i) != 0:
            rng = np.random.default_rng(self._c("noise_seed", i) + 1)
            states = apply_edit(states, self._c("edit_dev", i), self._c("edit_sign", i),
                                rng, self.cfg["min_edit_minutes"])
        flags = np.stack([states["vent"], states["niv"], states["crrt"],
                          states["inf"] / G.MAX_PUMPS], 1).astype(np.float32)
        val, msk, tgt, tmask = self._vitals(sid, tau)
        xm = np.concatenate([val, msk, flags], 1)
        cg = xg[:, 0].reshape(self.W, -1).mean(1)
        return {"xg": torch.from_numpy(xg), "xm": torch.from_numpy(xm),
                "cg": torch.from_numpy(cg.astype(np.float32)), "cm": torch.from_numpy(flags),
                "ypred": torch.from_numpy(tgt), "ypred_mask": torch.from_numpy(tmask),
                "y": torch.tensor(float(self._c("y", i))),
                "idx": torch.tensor(i)}


def collate(items):
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


# --------------------------------------------------------------------------- leakage audit
def leakage_audit(rec, cfg, backgrounds=None, sample_ds=None):
    out = {}
    pat = {sp: set(g.subject_id) for sp, g in rec.groupby("split")}
    pat["test"] = pat.get("test", set()) | pat.pop("test_replay", set())
    names = sorted(pat)
    out["patient_overlap"] = {"%s|%s" % (a, b): len(pat[a] & pat[b])
                              for i, a in enumerate(names) for b in names[i + 1:]}
    split_of = rec.drop_duplicates("stay_id").set_index("stay_id").split.replace(
        {"test_replay": "test"})
    src_split = rec.src_stay.map(split_of)
    out["cross_split_grid_sources"] = int((src_split != rec.split.replace({"test_replay": "test"})).sum())
    wp = rec[rec.ntype == "wrong_patient"]
    subj = rec.drop_duplicates("stay_id").set_index("stay_id").subject_id
    out["wrong_patient_same_patient"] = int((wp.src_stay.map(subj) == wp.subject_id).sum())
    hh = {sp: set(v) for sp, v in cfg["households"].items()}
    out["household_overlap"] = {"train|val": len(hh["train"] & hh["val"]),
                                "train|test": len(hh["train"] & hh["test"]),
                                "val|test": len(hh["val"] & hh["test"])}
    used = {sp: set(g.hh) for sp, g in rec.groupby("split")}
    out["households_used_outside_pool"] = int(sum(
        len(used.get(sp, set()) - hh.get("train" if sp == "mia_holdout" else
                                           "test" if sp == "test_replay" else sp, set()))
        for sp in used))
    cons = rec[rec.y == 0].sort_values(["stay_id", "tau"])
    gaps = cons.groupby("stay_id").tau.diff().dropna()
    out["overlapping_windows_within_stay"] = int((gaps < cfg["window_minutes"]).sum())
    test_groups = set(rec[rec.split == "test"].anchor_year_group)
    train_groups = set(rec[rec.split == "train"].anchor_year_group)
    out["temporal"] = {"train_groups": sorted(train_groups), "test_groups": sorted(test_groups),
                       "overlap": sorted(train_groups & test_groups)}
    hospital_mode = cfg.get("split_mode") == "hospital"
    if hospital_mode:
        hs = {sp: set(g.hospitalid) for sp, g in rec.replace({"split": {"test_replay": "test",
                                                                          "mia_holdout": "train"}}).groupby("split")}
        out["hospital_overlap"] = {"%s|%s" % (a, b): len(hs.get(a, set()) & hs.get(b, set()))
                                   for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
        out["n_hospitals"] = {sp: len(v) for sp, v in hs.items()}
        out["temporal"]["note"] = "split by held-out hospitals; temporal split not used"
    if sample_ds is not None:
        mins = [float(np.expm1(sample_ds.raw_grid(i)).min()) for i in range(min(200, len(sample_ds)))]
        out["zero_filled_grid_windows_in_sample"] = int(sum(m <= 0 for m in mins))
        out["min_grid_power_W_in_sample"] = min(mins)
    out["counts"] = rec.groupby(["split", "ntype"]).size().unstack(fill_value=0).to_dict()
    out["passed"] = (all(v == 0 for v in out["patient_overlap"].values())
                     and out["cross_split_grid_sources"] == 0
                     and out["wrong_patient_same_patient"] == 0
                     and all(v == 0 for v in out["household_overlap"].values())
                     and out["households_used_outside_pool"] == 0
                     and out["overlapping_windows_within_stay"] == 0
                     and (all(v == 0 for v in out["hospital_overlap"].values()) if hospital_mode
                          else not out["temporal"]["overlap"])
                     and out.get("zero_filled_grid_windows_in_sample", 0) == 0)
    return out


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
