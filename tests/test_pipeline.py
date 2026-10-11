"""Checks run before any real experiment:  python -m pytest -q tests
(or  python tests/test_pipeline.py).  Uses the synthetic TEST FIXTURE only."""
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sgm import benchmark as BM, config as C, data_mimic, dp, evaluate as E, synthetic  # noqa: E402
from sgm.model import SGMedFusion, sample_loss_fn  # noqa: E402


def test_rdp_matches_numerical_integration():
    """Closed-form integer-order RDP of the sampled Gaussian vs direct
    integration of E_{mu0}[(mu/mu0)^alpha], mu = (1-q) N(0,s^2) + q N(1,s^2)."""
    q, s = 0.05, 1.2
    for a in (2, 8, 16):
        f = lambda z: math.exp(stats.norm.logpdf(z, 0, s)
                               + a * math.log((1 - q) + q * math.exp((2 * z - 1) / (2 * s * s))))
        num = math.log(integrate.quad(f, -30, 60, limit=1000, points=[0, 5, 10, 20])[0]) / (a - 1)
        ana = dp.rdp_sampled_gaussian(q, s, [a])[0]
        assert abs(num - ana) < 1e-6 * max(1, abs(ana)), (a, num, ana)


def test_rdp_full_batch_and_calibration():
    assert np.allclose(dp.rdp_sampled_gaussian(1.0, 2.0, [2, 10]), [2 / 8, 10 / 8])
    e1, e2 = dp.epsilon(0.01, 1.0, 1000, 1e-5), dp.epsilon(0.01, 1.0, 2000, 1e-5)
    assert 0 < e1 < e2
    sig = dp.calibrate_sigma(0.02, 2000, 4.0, 1e-5)
    assert abs(dp.epsilon(0.02, sig, 2000, 1e-5) - 4.0) < 0.02


def _tiny_cfg(tmp):
    rng = np.random.default_rng(0)
    mimic = synthetic.write_mimic(Path(tmp) / "mimic", rng, n_patients=60)
    redd, ukdale = synthetic.write_nilm(Path(tmp) / "nilm", rng, hours=3)
    return C.load(None, paths={"mimic_dir": mimic, "redd_dir": redd, "ukdale_dir": ukdale,
                               "work_dir": str(Path(tmp) / "work"),
                               "results_dir": str(Path(tmp) / "results")},
                  window_minutes=10, windows_per_stay=3, n_sites=3, min_edit_minutes=3,
                  shift_range_min=[15, 120], replay_min_gap_min=600, min_stay_hours=6,
                  model={"d": 32, "heads": 2, "sa_layers": 1},
                  fl={"rounds": 2, "eval_every": 1, "patients_per_step": 4, "samples_cap_per_patient": 2,
                      "batch_size": 32},
                  eval={"bootstrap": 50, "latency_reps": 5, "mia_patients": 10})


def test_per_patient_clipping():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _tiny_cfg(tmp)
        data_mimic.extract(cfg, log=lambda *_: None)
        import run_all
        tl, bgs, rec, norms = run_all.prepare(cfg, 0)
        ds = BM.PairDataset(rec[rec.split == "train"].head(6), tl, bgs, cfg, norms)
        b = BM.collate([ds[i] for i in range(6)]); b.pop("idx")
        model = SGMedFusion(cfg)
        pidx = torch.tensor([0, 0, 1, 1, 2, 2])
        g = dp.dp_patient_step(model, sample_loss_fn(cfg), b, pidx, 3, clip=0.1, sigma=0.0, expected_batch=1.0)
        total = math.sqrt(sum(float(v.pow(2).sum()) for v in g.values()))
        assert total <= 3 * 0.1 + 1e-5, total


def test_end_to_end_and_consistency():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _tiny_cfg(tmp)
        data_mimic.extract(cfg, log=lambda *_: None)
        import run_all
        data = run_all.prepare(cfg, 0)
        audit = json.load(open(Path(cfg["paths"]["results_dir"]) / "leakage_audit_seed0.json"))
        assert audit["passed"], audit
        assert "zero" not in set(data[2].ntype)
        for exp in ("sgmf_eps4", "abl_ehr_only", "gbm_both", "probe_gbm_grid", "atk_signflip_median"):
            run_all.run_one(cfg, exp, 0, data)
            r = json.load(open(Path(cfg["paths"]["results_dir"]) / exp / "seed0.json"))
            m = r["test"]["main"]
            assert abs(m["accuracy"] - E.from_cm(m["confusion"])["accuracy"]) < 1e-12
            assert sum(m["confusion"].values()) == r["test"]["n"]
            if r["privacy"]["enabled"]:
                assert r["privacy"]["eps_max"] <= cfg["dp"]["target_eps"] + 0.05
        import make_report
        make_report.main([cfg["paths"]["results_dir"]])
        assert (Path(cfg["paths"]["results_dir"]) / "tables.md").exists()


def test_checkpoint_resume():
    """Stopping after round 1 and restarting must continue at round 2."""
    from sgm import fed
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _tiny_cfg(tmp)
        data_mimic.extract(cfg, log=lambda *_: None)
        import run_all
        tl, bgs, rec, norms = run_all.prepare(cfg, 0)
        cfg["_norms"] = norms
        ds = BM.PairDataset(rec, tl, bgs, cfg, norms)
        val = BM.PairDataset(rec[rec.split == "val"].reset_index(drop=True), tl, bgs, cfg, norms)
        ck = Path(tmp) / "run.ckpt"
        fed.train(dict(cfg, _stop_after_round=1), ds, rec, val, log=lambda *_: None, ckpt=ck)
        assert ck.exists()
        msgs = []
        _, info = fed.train(cfg, ds, rec, val, log=msgs.append, ckpt=ck)
        assert any("resumed from round 1" in m for m in msgs), msgs
        assert [h["round"] for h in info["history"]] == [1, 2]
        assert info["privacy"]["eps_max"] <= cfg["dp"]["target_eps"] + 0.05


def test_estimate_stage():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _tiny_cfg(tmp)
        data_mimic.extract(cfg, log=lambda *_: None)
        import run_all
        run_all.estimate(cfg, [0], list(run_all.EXPERIMENTS))
        assert set(run_all.TIER1) <= set(run_all.EXPERIMENTS)
        env = run_all.environment()
        assert "torch" in env and "python" in env


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
