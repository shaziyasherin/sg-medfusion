"""Background mains (REDD, UK-DALE) and parametric medical-device power models.

The public NILM corpora are residential, so they are used ONLY as realistic
background load (real appliance switching acts as a confounder).  Medical
device activity is synthesised from parametric models whose parameter ranges
are explicit assumptions (DEVICE_PARAMS) and are varied in a sensitivity test.
No appliance is "relabelled" as a medical device.
"""
from pathlib import Path

import numpy as np

# Assumed ranges (watts, Hz, relative modulation).  These are modelling
# assumptions, not measurements; see the manuscript, Section 4.3 and Table 3.
DEVICE_PARAMS = {
    "monitor": {"p": (30, 60)},
    "vent": {"p": (60, 150), "f": (0.20, 0.50), "a": (0.05, 0.20)},
    "niv": {"p": (40, 90), "f": (0.20, 0.50), "a": (0.05, 0.25)},
    "crrt": {"p": (150, 350), "f": (0.30, 0.90), "a": (0.05, 0.15)},
    "inf": {"p": (4, 12), "period": (2, 20)},
}
MAX_PUMPS = 4


def _read_two_col(path, max_days=None):
    """Read 'timestamp power' files fast; keep only the first `max_days` days
    (UK-DALE house 1 alone spans ~4 years, far more background than needed)."""
    import pandas as pd
    df = pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, 1], names=["t", "p"],
                     dtype={"t": np.float64, "p": np.float32}, engine="c")
    t = df.t.values.astype(np.int64)
    p = df.p.values
    order = np.argsort(t, kind="stable")
    t, p = t[order], p[order]
    if max_days:
        keep = t < t[0] + int(max_days * 86400)
        t, p = t[keep], p[keep]
    return t, p


def _to_1hz(ts, p, max_gap=10):
    t0, t1 = int(ts[0]), int(ts[-1])
    grid = np.arange(t0, t1 + 1)
    idx = np.searchsorted(ts, grid, side="right") - 1
    gap = grid - ts[idx]
    out = p[idx].copy()
    out[gap > max_gap] = np.nan
    return out


def load_redd(root, house, max_days=None):
    d = Path(root) / ("house_%d" % house)
    t1, p1 = _read_two_col(d / "channel_1.dat", max_days)
    t2, p2 = _read_two_col(d / "channel_2.dat", max_days)
    a, b = _to_1hz(t1, p1), _to_1hz(t2, p2)
    off = int(t2[0] - t1[0])
    if off >= 0:
        a = a[off:]
    else:
        b = b[-off:]
    n = min(len(a), len(b))
    return a[:n] + b[:n]


def load_ukdale(root, house, max_days=None):
    t, p = _read_two_col(Path(root) / ("house_%d" % house) / "channel_1.dat", max_days)
    return _to_1hz(t, p)   # 6 s aggregate held for up to 10 s


def load_backgrounds(cfg, names):
    out = {}
    days = cfg.get("background_max_days")
    for name in names:
        src, h = name.split("_")
        if src == "redd":
            out[name] = load_redd(cfg["paths"]["redd_dir"], int(h), days)
        elif src == "ukdale":
            out[name] = load_ukdale(cfg["paths"]["ukdale_dir"], int(h), days)
        else:
            raise ValueError(name)
    return out


def valid_starts(power, length):
    """Start indices of windows with no missing samples."""
    bad = np.isnan(power).astype(np.int32)
    c = np.concatenate([[0], np.cumsum(bad)])
    s = np.arange(0, len(power) - length + 1)
    return s[(c[s + length] - c[s]) == 0]


def device_params(stay_id, device, scale=1.0):
    """Deterministic per-(stay, device) parameters: the same bed keeps the same
    physical device across windows."""
    rng = np.random.default_rng([int(stay_id) % (2 ** 32), list(DEVICE_PARAMS).index(device)])
    pr = DEVICE_PARAMS[device]
    out = {k: rng.uniform(*v) for k, v in pr.items()}
    out["p"] *= scale
    out["phase"] = rng.uniform(0, 2 * np.pi)
    return out


def activity(intervals, start_min, W):
    """Per-minute device states for minutes [start_min, start_min+W)."""
    m = np.arange(start_min, start_min + W)
    st = {}
    for dev in ("vent", "niv", "crrt"):
        on = np.zeros(W, np.float32)
        for a, b in intervals.get(dev, []):
            on[(m >= a) & (m < b)] = 1
        st[dev] = on
    n = np.zeros(W, np.float32)
    for a, b in intervals.get("inf", []):
        n += ((m >= a) & (m < b))
    st["inf"] = np.minimum(n, MAX_PUMPS)
    return st


def device_power(stay_id, states, hz, rng, scale=1.0):
    W = len(states["vent"])
    t = np.arange(W * 60 * hz) / hz
    mi = (np.arange(W * 60 * hz) // (60 * hz)).astype(int)
    p = np.full(len(t), device_params(stay_id, "monitor", scale)["p"], np.float32)
    for dev in ("vent", "niv"):
        q = device_params(stay_id, dev, scale)
        breath = np.maximum(0.0, np.sin(2 * np.pi * q["f"] * t + q["phase"])) ** 2
        p += states[dev][mi] * q["p"] * (1 + q["a"] * breath)
    q = device_params(stay_id, "crrt", scale)
    p += states["crrt"][mi] * q["p"] * (1 + q["a"] * np.sin(2 * np.pi * q["f"] * t + q["phase"]))
    q = device_params(stay_id, "inf", scale)
    pulse = ((t / q["period"]) % 1.0) < 0.1
    p += states["inf"][mi] * q["p"] * (1 + 0.5 * pulse)
    p += rng.normal(0, 0.5, len(t)) + rng.normal(0, 0.002, len(t)) * p
    return p.astype(np.float32)
