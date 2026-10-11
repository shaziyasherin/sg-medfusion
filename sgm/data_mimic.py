"""Extract compact ICU timelines from MIMIC-IV (v2.2 or later).

For every eligible ICU stay we keep
  * vitals (HR, MAP, SpO2, RR) as sparse (minute, value) series,
  * documented device intervals (invasive ventilation, NIV, CRRT) from
    procedureevents and continuous-infusion intervals from inputevents,
  * patient attributes for splitting and subgroup analysis.
Times are minutes since ICU admission (intime).  Nothing here leaves the
credentialed environment: the output pickle is derived MIMIC data and must not
be redistributed (PhysioNet Credentialed Health Data Use Agreement).
"""
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

VITAL_ITEMS = {"hr": [220045], "map": [220052, 220181], "spo2": [220277], "rr": [220210]}
DEVICE_LABELS = {
    "vent": r"^invasive ventilation$",
    "niv": r"^non-invasive ventilation$",
    "crrt": r"^dialysis\s*-\s*crrt$",
}
CONTINUOUS_INFUSION = {"Continuous Med", "Continuous IV"}
DEVICES = ["vent", "niv", "crrt", "inf"]


def _path(root, rel):
    p = Path(root) / rel
    for c in (p, Path(str(p) + ".gz")):
        if c.exists():
            return c
    raise FileNotFoundError(p)


def _minutes(ts, t0):
    return ((pd.to_datetime(ts) - t0).dt.total_seconds() // 60).astype("int64")


def resolve_device_items(mimic_dir, log):
    d = pd.read_csv(_path(mimic_dir, "icu/d_items.csv"), usecols=["itemid", "label"])
    out = {}
    for dev, pat in DEVICE_LABELS.items():
        ids = d.loc[d["label"].str.strip().str.lower().str.match(pat), "itemid"].tolist()
        if not ids:
            raise RuntimeError("no d_items label matches %r for %s" % (pat, dev))
        out[dev] = ids
        log("device %-5s -> itemids %s" % (dev, ids))
    return out


def extract(cfg, log=print):
    root = cfg["paths"]["mimic_dir"]
    stays = pd.read_csv(_path(root, "icu/icustays.csv"),
                        parse_dates=["intime", "outtime"])
    pats = pd.read_csv(_path(root, "hosp/patients.csv"))
    stays = stays.merge(pats[["subject_id", "gender", "anchor_age", "anchor_year",
                              "anchor_year_group"]], on="subject_id")
    stays["los_min"] = ((stays.outtime - stays.intime).dt.total_seconds() // 60).astype(int)
    stays = stays[stays.los_min >= cfg["min_stay_hours"] * 60].copy()
    stays["los_min"] = stays.los_min.clip(upper=cfg["max_stay_days"] * 1440)
    stays["age"] = stays.anchor_age + (stays.intime.dt.year - stays.anchor_year)
    stays = stays[stays.age >= 18]
    if cfg["max_stays"]:
        stays = stays.sample(n=min(cfg["max_stays"], len(stays)),
                             random_state=cfg["seed"])
    stays = stays.set_index("stay_id")
    log("eligible stays: %d (patients: %d)" % (len(stays), stays.subject_id.nunique()))
    intime = stays["intime"]

    # --- device intervals ---------------------------------------------------
    items = resolve_device_items(root, log)
    intervals = {sid: {d: [] for d in DEVICES} for sid in stays.index}
    pe = pd.read_csv(_path(root, "icu/procedureevents.csv"),
                     usecols=["stay_id", "itemid", "starttime", "endtime"])
    pe = pe[pe.stay_id.isin(stays.index)]
    for dev, ids in items.items():
        sub = pe[pe.itemid.isin(ids)]
        t0 = intime.loc[sub.stay_id].values
        s = ((pd.to_datetime(sub.starttime).values - t0) / np.timedelta64(1, "m")).astype(int)
        e = ((pd.to_datetime(sub.endtime).values - t0) / np.timedelta64(1, "m")).astype(int)
        for sid, a, b in zip(sub.stay_id.values, s, e):
            if b > a:
                intervals[sid][dev].append((int(a), int(b)))

    ie = pd.read_csv(_path(root, "icu/inputevents.csv"),
                     usecols=["stay_id", "starttime", "endtime", "linkorderid",
                              "ordercategorydescription"])
    ie = ie[ie.stay_id.isin(stays.index) & ie.ordercategorydescription.isin(CONTINUOUS_INFUSION)]
    t0 = intime.loc[ie.stay_id].values
    ie = ie.assign(s=((pd.to_datetime(ie.starttime).values - t0) / np.timedelta64(1, "m")).astype(int),
                   e=((pd.to_datetime(ie.endtime).values - t0) / np.timedelta64(1, "m")).astype(int))
    # one pump per linkorderid: union of its rate segments
    for (sid, _), g in ie.groupby(["stay_id", "linkorderid"]):
        for a, b in _union(zip(g.s.values, g.e.values)):
            intervals[sid]["inf"].append((a, b))

    # --- vitals ---------------------------------------------------------------
    wanted = {i: k for k, ids in VITAL_ITEMS.items() for i in ids}
    parts = []
    reader = pd.read_csv(_path(root, "icu/chartevents.csv"),
                         usecols=["stay_id", "itemid", "charttime", "valuenum"],
                         chunksize=5_000_000)
    for n, ch in enumerate(reader):
        ch = ch[ch.itemid.isin(wanted.keys()) & ch.stay_id.isin(stays.index)].dropna(subset=["valuenum"])
        parts.append(ch)
        if n % 10 == 0:
            log("chartevents chunk %d" % n)
    ce = pd.concat(parts)
    ce["var"] = ce.itemid.map(wanted)
    ce = _plausible(ce)
    t0 = intime.loc[ce.stay_id].values
    ce["m"] = ((pd.to_datetime(ce.charttime).values - t0) / np.timedelta64(1, "m")).astype(int)
    # MAP: prefer arterial (220052) over non-invasive (220181) within the same minute
    ce["pri"] = np.where(ce.itemid == 220181, 1, 0)
    ce = ce.sort_values(["stay_id", "var", "m", "pri"]).drop_duplicates(["stay_id", "var", "m"])
    vitals = {sid: {} for sid in stays.index}
    for (sid, var), g in ce.groupby(["stay_id", "var"]):
        vitals[sid][var] = (g.m.values.astype(np.int32), g.valuenum.values.astype(np.float32))

    meta = stays.reset_index()[["stay_id", "subject_id", "first_careunit", "los_min",
                                "gender", "age", "anchor_year_group"]]
    out = {"meta": meta, "intervals": intervals, "vitals": vitals,
           "device_items": items}
    work = Path(cfg["paths"]["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "timelines.pkl", "wb") as f:
        pickle.dump(out, f)
    log("wrote %s" % (work / "timelines.pkl"))
    return out


def _union(segs):
    segs = sorted((int(a), int(b)) for a, b in segs if b > a)
    out = []
    for a, b in segs:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


PLAUSIBLE = {"hr": (20, 250), "map": (20, 200), "spo2": (50, 100), "rr": (3, 60)}


def _plausible(ce):
    keep = np.ones(len(ce), bool)
    for var, (lo, hi) in PLAUSIBLE.items():
        m = ce["var"].values == var
        v = ce["valuenum"].values
        keep &= ~m | ((v >= lo) & (v <= hi))
    return ce[keep]


def load(cfg):
    with open(Path(cfg["paths"]["work_dir"]) / "timelines.pkl", "rb") as f:
        return pickle.load(f)
