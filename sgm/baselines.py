"""Non-neural baselines and shortcut probes on hand-crafted summary statistics.

If a probe that sees only ONE modality scores far above AUROC 0.5, the
benchmark contains a shortcut (the label is decidable without cross-modal
matching).  These probes are reported alongside the main results.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BANDS = [(0.02, 0.1), (0.1, 0.2), (0.2, 0.5)]


def grid_features(xg, hz):
    x, dx = xg[:, 0], xg[:, 1]
    f = np.fft.rfftfreq(len(x), 1.0 / hz)
    p = np.abs(np.fft.rfft(x - x.mean())) ** 2
    bands = [np.log1p(p[(f >= a) & (f < b)].sum()) for a, b in BANDS]
    return [x.mean(), x.std(), np.percentile(x, 5), np.percentile(x, 95), x.max() - x.min(),
            np.abs(dx).mean(), dx.std()] + bands


def ehr_features(xm):
    return list(xm.mean(0)) + list(xm[:, 8:].max(0))


def featurise(ds, which, hz):
    rows = []
    for i in range(len(ds)):
        it = ds[i]
        r = []
        if which in ("both", "grid"):
            r += grid_features(it["xg"].numpy(), hz)
        if which in ("both", "ehr"):
            r += ehr_features(it["xm"].numpy())
        rows.append(r)
    return np.asarray(rows, np.float32)


def fit_predict(kind, Xtr, ytr, Xs, seed):
    if kind == "gbm":
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)
    else:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    m.fit(Xtr, ytr)
    return [m.predict_proba(X)[:, 1] for X in Xs]
