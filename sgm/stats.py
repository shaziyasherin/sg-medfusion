"""Across-run statistics: mean, SD, t-based 95% CI, paired t-test, Holm."""
import numpy as np
from scipy import stats


def summarise(values):
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], float)
    if len(v) == 0:
        return {"n": 0}
    out = {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    if len(v) > 1:
        h = stats.t.ppf(0.975, len(v) - 1) * out["sd"] / np.sqrt(len(v))
        out["lo"], out["hi"] = out["mean"] - h, out["mean"] + h
    return out


def paired_t(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2:
        return {"diff": float(np.mean(a - b)), "p": float("nan")}
    t = stats.ttest_rel(a, b)
    return {"diff": float(np.mean(a - b)), "t": float(t.statistic), "p": float(t.pvalue)}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (dict name -> p)."""
    items = sorted(((p, k) for k, p in pvals.items() if not np.isnan(p)))
    m, out, run = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    for k, p in pvals.items():
        out.setdefault(k, float("nan"))
    return out
