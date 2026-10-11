"""Default configuration for the SG-MedFusion R1 experiments.

Every number reported in the revised manuscript must come from a run of this
code with a config that is saved next to the results (see scripts/run_all.py).
"""
import copy
import json

DEFAULT = {
    "paths": {
        "mimic_dir": None,        # root of MIMIC-IV (contains hosp/ and icu/)
        "redd_dir": None,         # REDD low_freq/ root
        "ukdale_dir": None,       # UK-DALE root (house_1/, house_2/, ...)
        "eicu_dir": None,         # eICU-CRD demo 2.0.1 (patient.csv.gz, ...)
        "work_dir": "work",
        "results_dir": "results",
    },
    "dataset": "mimic",           # "mimic" (credentialed) | "eicu" (open-access demo)
    "split_mode": "temporal",     # "temporal" (MIMIC anchor years) | "hospital" (held-out hospitals)
    "test_hospital_frac": 0.2,
    "val_hospital_frac": 0.1,
    # --- windows -----------------------------------------------------------
    "window_minutes": 30,         # W: length of the grid and EHR windows
    "grid_hz": 1,                 # grid sampling rate after resampling
    "horizon_minutes": 60,        # forecasting horizon for the auxiliary task
    "lookback_minutes": 240,      # LOCF limit for vitals
    "windows_per_stay": 8,
    "min_stay_hours": 6,
    "max_stay_days": 14,
    "max_stays": None,            # cap for quick runs (None = all)
    "device_window_frac": 0.5,    # share of windows drawn where a device is on
    # --- inconsistency construction -----------------------------------------
    "neg_types": ["shift", "wrong_patient", "fabricate", "omit"],
    "shift_range_min": [30, 360],
    "replay_min_gap_min": 1440,
    "min_edit_minutes": 10,
    # --- splits --------------------------------------------------------------
    # latest anchor-year groups form the test split ("2020 - 2022" exists from MIMIC-IV v3)
    "test_year_groups": ["2017 - 2019", "2020 - 2022"],
    "val_frac": 0.15,
    "mia_frac": 0.05,
    "n_sites": 8,
    "households": {               # background mains, disjoint across splits
        "train": ["redd_1", "redd_2", "redd_3", "redd_4", "ukdale_1", "ukdale_2"],
        "val": ["redd_5", "ukdale_3"],
        "test": ["redd_6", "ukdale_4", "ukdale_5"],
    },
    "device_profile_scale_test": 1.0,   # >1 or <1 = device-profile shift test
    "background_max_days": 60,          # days of mains kept per household
    # --- model ---------------------------------------------------------------
    "model": {
        "d": 64,
        "heads": 4,
        "sa_layers": 2,
        "ffn_mult": 2,
        "dropout": 0.1,
        "fusion": "xattn",        # "xattn" | "concat"
        "use_grid": True,
        "use_ehr": True,
        "use_cycle": True,
        "attn_noise_sigma": 0.0,  # logit-noise attention (no formal DP claim)
    },
    "loss": {"alpha_pred": 0.5, "lambda_cyc": 0.3},
    # --- federated training --------------------------------------------------
    "fl": {
        "rounds": 40,
        "local_epochs": 1,
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "aggregator": "fedavg",   # "fedavg" | "trimmed_mean" | "median"
        "trim_k": 1,              # coordinates trimmed per side
        "prox_mu": 0.0,           # >0 turns FedAvg into FedProx
        "patients_per_step": 32,  # expected patients per DP step (Poisson)
        "samples_cap_per_patient": 8,
        "batch_size": 128,        # non-private training minibatch
        "attack": None,           # None | "signflip" | "labelflip"
        "n_malicious": 0,
        "attack_scale": 5.0,
        "eval_every": 5,
        "centralized": False,     # True = pool all sites (upper bound)
    },
    "dp": {
        "enabled": True,
        "target_eps": 4.0,
        "delta": 1e-5,
        "clip": 1.0,
        "noise_multiplier": None, # None = calibrate to target_eps
    },
    "eval": {
        "bootstrap": 1000,
        "prevalences": [0.01, 0.05],
        "target_fpr": 0.05,
        "latency_reps": 200,
        "mia_patients": 500,
    },
    "seed": 0,
    "device": "cpu",
}


def merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path=None, **override):
    cfg = copy.deepcopy(DEFAULT)
    if path:
        with open(path) as f:
            cfg = merge(cfg, json.load(f))
    return merge(cfg, override)
