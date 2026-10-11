"""TEST FIXTURE ONLY.  Writes tiny fake files with the same schema as MIMIC-IV,
REDD (low_freq) and UK-DALE so the real loaders and the whole pipeline can be
smoke-tested without credentialed data.  Nothing produced from these files may
be reported as a result.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def _bg(rng, seconds, step):
    t = np.arange(0, seconds, step)
    p = np.full(len(t), rng.uniform(80, 200))
    for _ in range(int(seconds / 900)):          # appliance switching events
        a = rng.integers(0, len(t)); b = min(len(t), a + rng.integers(20, 600) // step)
        p[a:b] += rng.uniform(50, 2000)
    return t + 1_300_000_000, p + rng.normal(0, 5, len(t))


def write_nilm(root, rng, hours=4):
    root = Path(root)
    for h in range(1, 7):
        d = root / "redd" / ("house_%d" % h); d.mkdir(parents=True, exist_ok=True)
        for ch in (1, 2):
            t, p = _bg(rng, hours * 3600, 1)
            np.savetxt(d / ("channel_%d.dat" % ch), np.c_[t, p / 2], fmt="%d %.2f")
    for h in range(1, 6):
        d = root / "ukdale" / ("house_%d" % h); d.mkdir(parents=True, exist_ok=True)
        t, p = _bg(rng, hours * 3600, 6)
        np.savetxt(d / "channel_1.dat", np.c_[t, p], fmt="%d %.2f")
    return str(root / "redd"), str(root / "ukdale")


def write_mimic(root, rng, n_patients=60):
    root = Path(root)
    (root / "icu").mkdir(parents=True, exist_ok=True)
    (root / "hosp").mkdir(parents=True, exist_ok=True)
    groups = ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019"]
    units = ["MICU", "SICU", "CVICU", "CCU", "TSICU"]
    pats, stays, pe, ie, ce = [], [], [], [], []
    for i in range(n_patients):
        sid, stay = 10_000_000 + i, 30_000_000 + i
        pats.append(dict(subject_id=sid, gender=rng.choice(["F", "M"]),
                         anchor_age=int(rng.integers(20, 90)), anchor_year=2150,
                         anchor_year_group=groups[i % 4], dod=None))
        intime = pd.Timestamp("2150-01-01") + pd.Timedelta(hours=int(rng.integers(0, 1000)))
        los_h = int(rng.integers(30, 72))
        stays.append(dict(subject_id=sid, hadm_id=20_000_000 + i, stay_id=stay,
                          first_careunit=units[i % len(units)], last_careunit=units[i % len(units)],
                          intime=intime, outtime=intime + pd.Timedelta(hours=los_h), los=los_h / 24))
        for item in (225792, 225794, 225802):
            if rng.random() < 0.5:
                a = int(rng.integers(0, los_h * 60 - 120)); b = a + int(rng.integers(60, 600))
                pe.append(dict(subject_id=sid, stay_id=stay, itemid=item,
                               starttime=intime + pd.Timedelta(minutes=a),
                               endtime=intime + pd.Timedelta(minutes=b)))
        for k in range(int(rng.integers(0, 4))):
            a = int(rng.integers(0, los_h * 60 - 60))
            for seg in range(2):
                ie.append(dict(subject_id=sid, stay_id=stay, linkorderid=stay * 10 + k,
                               starttime=intime + pd.Timedelta(minutes=a + seg * 60),
                               endtime=intime + pd.Timedelta(minutes=a + seg * 60 + 60),
                               ordercategorydescription="Continuous Med"))
        for item, mu in ((220045, 85), (220052, 75), (220181, 78), (220277, 96), (220210, 18)):
            for m in range(0, los_h * 60, int(rng.integers(15, 60))):
                ce.append(dict(subject_id=sid, stay_id=stay, itemid=item,
                               charttime=intime + pd.Timedelta(minutes=m),
                               valuenum=float(mu + rng.normal(0, 5))))
    pd.DataFrame(pats).to_csv(root / "hosp/patients.csv", index=False)
    pd.DataFrame(stays).to_csv(root / "icu/icustays.csv", index=False)
    pd.DataFrame(pe).to_csv(root / "icu/procedureevents.csv", index=False)
    pd.DataFrame(ie).to_csv(root / "icu/inputevents.csv", index=False)
    pd.DataFrame(ce).to_csv(root / "icu/chartevents.csv", index=False)
    pd.DataFrame(dict(itemid=[225792, 225794, 225802, 220045],
                      label=["Invasive Ventilation", "Non-invasive Ventilation",
                             "Dialysis - CRRT", "Heart Rate"])).to_csv(root / "icu/d_items.csv", index=False)
    return str(root)
