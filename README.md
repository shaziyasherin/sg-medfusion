# SG-MedFusion: reproducibility code

Code for the article:

> Johar S, Ganavi M, Banu S S. *SG-MedFusion: Federated Cross-Modal Consistency Checking of Medical-Device Power Signatures and EHR Documentation — A Proof-of-Concept Study with a Patient-Level Privacy Analysis.* Intelligence-Based Medicine (revised manuscript IBMED-D-26-00504).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23186964.svg)](https://doi.org/10.5281/zenodo.23186964)

Archived code and all result files: https://doi.org/10.5281/zenodo.23186964

It regenerates the benchmark from **open data**, trains every model of the paper, and writes every table and figure from saved outputs. No number in the paper is typed by hand.

## 1. What the code does

1. Reads ICU device documentation and vital signs from the open-access **eICU Collaborative Research Database Demo v2.0.1** (2,309 adult stays, 186 hospitals).
2. Simulates bedside device power from the documented device intervals using parametric device models, added to real household mains from **UK-DALE**, which is used only as background load.
3. Builds consistent and inconsistent power–EHR pairs: shift, wrong patient, fabricated entry and omitted entry. It also builds a replay stress set.
4. Splits the data by **held-out hospitals**, patients and households, and checks for leakage automatically. A run stops if the audit fails.
5. Trains SG-MedFusion federatedly across 4 sites (FedAvg, FedProx, trimmed mean, median), with or without patient-level DP-SGD. A Rényi-DP accountant covers every step, epoch and round.
6. Evaluates AUROC and other metrics, operating points, per-type detection, shortcut probes, membership inference, robustness, distribution shift, subgroups and latency.

## 2. Data (downloaded by the scripts, no login needed)

| Source | Licence | Script |
|---|---|---|
| eICU-CRD Demo v2.0.1, https://doi.org/10.13026/4mxk-na84 | ODbL v1.0 | `scripts/download_eicu.py` (about 21 MB) |
| UK-DALE (2017 release), https://doi.org/10.5286/UKERC.EDC.000004 | CC BY 4.0 | `scripts/download_ukdale.py` (about 100 MB; fetches only `house_1..5/channel_1.dat`) |

## 3. Installation

Python 3.10, PyTorch ≥ 2.0 (uses `torch.func`), numpy, pandas, scipy, scikit-learn, matplotlib.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python tests/test_pipeline.py        # 6 checks on a synthetic fixture, about 10 min on CPU
```

## 4. Reproduce the paper

Edit the paths in `configs/paths_aimlserver.json` (or copy it), then:

```bash
python scripts/download_eicu.py   --out data/eicu_demo
python scripts/download_ukdale.py --out data/ukdale
python scripts/run_all.py --config configs/paths_aimlserver.json --stage extract
python scripts/run_all.py --config configs/paths_aimlserver.json --stage run --plan minimal --seeds 0 1 2
python scripts/make_report.py configs/paths_aimlserver.json       # tables.md + figures/
python scripts/paper_tables.py <results_dir> <out_dir>            # manuscript Tables 6-11, S1-S8
```

On Windows, the batch files in `aimlserver/` (STEP1 to STEP6) run the same steps.

- Runs are resumable: a progress file is saved every 5 rounds, and finished (experiment, seed) pairs are skipped.
- The paper's 46 runs took 9.5 hours in total (measured) on one NVIDIA RTX 4000 Ada (20 GB).
- Each run writes `results/<experiment>/seed<k>.json`, which contains the configuration, environment, privacy accounting, training history, all metrics and confusion matrices. It also writes `seed<k>_scores.npz` with the test scores.

## 5. Experiments (`--plan minimal`)

| Name | What |
|---|---|
| `sgmf_nodp` | federated SG-MedFusion without DP (reference) |
| `sgmf_eps1/4/8/16` | the same with patient-level DP at ε = 1, 4, 8, 16 |
| `concat_nodp`, `abl_no_cycle_nodp`, `fedprox_nodp` | ablations / optimiser |
| `atk1_signflip_fedavg_nodp`, `atk1_signflip_trim_nodp` | one malicious site of four |
| `gbm_both`, `lr_both` | gradient boosting / logistic regression on summary features |
| `probe_gbm_grid`, `probe_gbm_ehr` | single-modality shortcut probes |

## 6. Layout

| File | Content |
|---|---|
| `sgm/config.py` | defaults (manuscript Table 5) |
| `sgm/data_eicu.py` | eICU extraction and device-interval rules (manuscript Section 6.2) |
| `sgm/data_mimic.py` | optional MIMIC-IV extraction (credentialed users only; not used in the paper) |
| `sgm/data_grid.py` | UK-DALE loader, parametric device power models (Table 4) |
| `sgm/benchmark.py` | splits, sites, pairs, dataset, leakage audit |
| `sgm/model.py` | SG-MedFusion (Eqs. 1–10) |
| `sgm/dp.py` | per-patient DP-SGD and RDP accountant (Eqs. 11–15) |
| `sgm/fed.py` | federated training, robust aggregation, attacks, checkpoints |
| `sgm/evaluate.py`, `sgm/stats.py` | metrics, clustered bootstrap, MIA, latency, statistics |
| `sgm/baselines.py` | summary-feature baselines and probes |
| `sgm/synthetic.py` | **test fixture only**: fake files in the real formats |
| `scripts/` | runner, downloads, report, paper tables, progress |
| `tests/test_pipeline.py` | accountant vs. numerical integration, per-patient clipping, end-to-end consistency, checkpoint resume |

## 7. Licence and citation

Code: MIT (see `LICENSE`). The result files archived with the paper are derived from eICU data and are licensed under ODbL v1.0. Please cite the article and the data sources (eICU-CRD Demo, UK-DALE, PhysioNet); see `CITATION.cff`.
