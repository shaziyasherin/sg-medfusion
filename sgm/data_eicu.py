"""Extract compact ICU timelines from the eICU Collaborative Research Database
Demo v2.0.1 (open access, ODbL v1.0; https://doi.org/10.13026/4mxk-na84).

Output has the same structure as data_mimic.extract, plus `hospitalid`, so the
rest of the pipeline is unchanged.  All eICU times are minutes from unit
admission, which is exactly the timeline used here.

Device-interval rules (documented in the manuscript, Section 6.2):
  vent  invasive ventilation: respiratoryCharting rows with ventilator-setting
        labels or O2 device = Ventilator / Mechanical ventilator, merged when
        consecutive charts are <= 240 min apart (interval = first .. last+60);
        union with respiratoryCare ventstart/ventend where end > start.
  niv   non-invasive ventilation: O2 device Bi-PAP/CPAP, NIV settings, BiPAP
        delivery mode; same merging rule.
  crrt  dialysis machine: treatment strings starting 'renal|dialysis'; each
        record = 240 min of machine activity (offsets only are charted).
  inf   one pump per drug name in infusiondrug; running between consecutive
        charts <= 120 min apart with rate > 0; last chart extends 60 min.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

VENT_LABELS = {"Tidal Volume (set)", "Vent Rate", "PEEP", "Pressure Support", "Mean Airway Pressure",
               "Peak Insp. Pressure", "Plateau Pressure", "Exhaled MV", "Set Vt (Servo,LTV)",
               "Pressure Control", "Ventilator Support Mode", "Mechanical Ventilator Mode"}
VENT_DEVICE_VALUES = {"Ventilator", "Mechanical ventilator"}
NIV_LABELS = {"NIV Setting EPAP", "Bipap Delivery Mode", "Non-invasive Ventilation Mode"}
NIV_DEVICE_VALUES = {"Bi-PAP", "CPAP", "BiPAP"}
DEVICE_LABELS = {"O2 Device", "Oxygen Delivery Method"}
VITALS = {"heartrate": "hr", "systemicmean": "map", "sao2": "spo2", "respiration": "rr"}
PLAUSIBLE = {"hr": (20, 250), "map": (20, 200), "spo2": (50, 100), "rr": (3, 60)}
DEVICES = ["vent", "niv", "crrt", "inf"]


def _read(root, name, **kw):
    for c in (Path(root) / (name + ".csv.gz"), Path(root) / (name + ".csv")):
        if c.exists():
            return pd.read_csv(c, **kw)
    raise FileNotFoundError(Path(root) / name)


def _merge_times(times, gap, tail):
    """Chart times -> intervals: consecutive charts <= gap apart are one episode."""
    t = np.sort(np.unique(np.asarray(times, dtype=np.int64)))
    out = []
    for x in t:
        if out and x - out[-1][1] <= gap:
            out[-1][1] = x
        else:
            out.append([x, x])
    return [(int(a), int(b + tail)) for a, b in out]


def _union(segs):
    segs = sorted((int(a), int(b)) for a, b in segs if b > a)
    out = []
    for a, b in segs:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def extract(cfg, log=print):
    root = cfg["paths"]["eicu_dir"]
    p = _read(root, "patient")
    p["age_num"] = pd.to_numeric(p.age.replace({"> 89": "90"}), errors="coerce")
    p = p[(p.unitdischargeoffset >= cfg["min_stay_hours"] * 60) & (p.age_num >= 18)].copy()
    p["los_min"] = p.unitdischargeoffset.clip(upper=cfg["max_stay_days"] * 1440).astype(int)
    if cfg.get("max_stays"):
        p = p.sample(n=min(cfg["max_stays"], len(p)), random_state=cfg["seed"])
    p["subject_id"] = pd.factorize(p.uniquepid)[0] + 1
    stays = set(p.patientunitstayid)
    log("eligible stays: %d (patients: %d, hospitals: %d)" % (len(p), p.uniquepid.nunique(), p.hospitalid.nunique()))
    intervals = {s: {d: [] for d in DEVICES} for s in stays}

    rc = _read(root, "respiratoryCharting", usecols=["patientunitstayid", "respchartoffset",
                                                     "respchartvaluelabel", "respchartvalue"])
    rc = rc[rc.patientunitstayid.isin(stays)]
    lab, val = rc.respchartvaluelabel.astype(str), rc.respchartvalue.astype(str).str.strip()
    is_dev = lab.isin(DEVICE_LABELS)
    vent = rc[lab.isin(VENT_LABELS) | (is_dev & val.isin(VENT_DEVICE_VALUES))]
    niv = rc[lab.isin(NIV_LABELS) | (is_dev & val.isin(NIV_DEVICE_VALUES))]
    for df, dev in ((vent, "vent"), (niv, "niv")):
        for sid, g in df.groupby("patientunitstayid"):
            intervals[sid][dev] += _merge_times(g.respchartoffset.values, 240, 60)
    care = _read(root, "respiratoryCare", usecols=["patientunitstayid", "ventstartoffset", "ventendoffset"])
    care = care[care.patientunitstayid.isin(stays) & (care.ventendoffset > care.ventstartoffset)]
    for sid, a, b in care[["patientunitstayid", "ventstartoffset", "ventendoffset"]].itertuples(index=False):
        intervals[sid]["vent"].append((int(a), int(b)))

    tr = _read(root, "treatment", usecols=["patientunitstayid", "treatmentoffset", "treatmentstring"])
    tr = tr[tr.patientunitstayid.isin(stays) & tr.treatmentstring.str.startswith("renal|dialysis", na=False)]
    for sid, t in tr[["patientunitstayid", "treatmentoffset"]].itertuples(index=False):
        intervals[sid]["crrt"].append((int(t), int(t) + 240))

    inf = _read(root, "infusiondrug", usecols=["patientunitstayid", "infusionoffset", "drugname", "drugrate"])
    inf = inf[inf.patientunitstayid.isin(stays)].copy()
    inf["rate"] = pd.to_numeric(inf.drugrate, errors="coerce")
    inf = inf[inf.rate > 0]
    for (sid, _), g in inf.groupby(["patientunitstayid", "drugname"]):
        intervals[sid]["inf"] += _merge_times(g.infusionoffset.values, 120, 60)

    for sid in intervals:
        for dev in ("vent", "niv", "crrt"):
            intervals[sid][dev] = _union(intervals[sid][dev])

    v = _read(root, "vitalPeriodic", usecols=["patientunitstayid", "observationoffset"] + list(VITALS))
    v = v[v.patientunitstayid.isin(stays)]
    vitals = {s: {} for s in stays}
    for col, var in VITALS.items():
        lo, hi = PLAUSIBLE[var]
        d = v[["patientunitstayid", "observationoffset", col]].dropna()
        d = d[(d[col] >= lo) & (d[col] <= hi)].sort_values(["patientunitstayid", "observationoffset"])
        d = d.drop_duplicates(["patientunitstayid", "observationoffset"])
        for sid, g in d.groupby("patientunitstayid"):
            vitals[sid][var] = (g.observationoffset.values.astype(np.int32), g[col].values.astype(np.float32))

    meta = pd.DataFrame({"stay_id": p.patientunitstayid.values, "subject_id": p.subject_id.values,
                         "first_careunit": p.unittype.values, "los_min": p.los_min.values,
                         "gender": p.gender.map({"Male": "M", "Female": "F"}).fillna("U").values,
                         "age": p.age_num.values, "anchor_year_group": p.hospitaldischargeyear.astype(str).values,
                         "hospitalid": p.hospitalid.values})
    n_dev = {d: sum(bool(intervals[s][d]) for s in stays) for d in DEVICES}
    log("stays with documented device use: %s" % n_dev)
    out = {"meta": meta, "intervals": intervals, "vitals": vitals, "device_items": {"source": "eICU-CRD demo 2.0.1"}}
    work = Path(cfg["paths"]["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "timelines.pkl", "wb") as f:
        pickle.dump(out, f)
    log("wrote %s" % (work / "timelines.pkl"))
    return out
