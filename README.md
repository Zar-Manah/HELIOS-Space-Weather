# ☀️ HELIOS v3.4 — Where Remote CME Forecasting Stops

> **An empirical information-ceiling study of CME arrival-time forecasting across Solar Cycles 23–25, with a fully automatic five-stage pipeline as the test vehicle.**

[![License: MIT / CC BY 4.0](https://img.shields.io/badge/License-MIT%20%2F%20CC%20BY%204.0-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24215019.svg)](https://doi.org/10.5281/zenodo.24215019)

---

## 🖥️ Zero-Install Interactive Cockpit

`Zar-Manah_HELIOS.html` is a single self-contained file: no build step, no server, no
external library, no analytics. Open it in any modern browser (macOS, Windows, Linux,
Android, iOS) and it runs.

It has two modes, and they are not the same thing:

| Mode | What runs | What it is |
|---|---|---|
| **Standalone** (default) | exact v3.4 physics backbone + live NOAA SWPC telemetry | physics-only: exact DBM transit (70-step bisection, same constants), Burton–O'Brien–McPherron ODE rollout, launch geometry and the deterministic `bz_risk_tag`. It does **not** run the 14 frozen `.joblib` artefacts. |
| **FULL ML v3.4** (optional) | `predict.py` + the 14 frozen `.joblib` artefacts through a localhost bridge | the calibrated five-stage pipeline, same code path as the CLI: impact_blend, transit CQR, apex_v3, MO-Net + 3DCORE, tactical GBDT + ACI |

To enable FULL ML on macOS, double-click `Zar-Manah_HELIOS.command`; it starts the bridge
(`python3 server.py`, loopback only, `127.0.0.1:8787`), waits for it and opens the cockpit
on `http://127.0.0.1:8787/`. On other systems run `python3 server.py` and open that URL.
Without the bridge the cockpit stays in standalone mode and says so.

What the standalone mode does **not** do: it does not reproduce the four-cell table, the
CQR/ACI coverage or any benchmark number in this README. Those come from the CLI and the
frozen artefacts. The cockpit states this in its header badge (`PHYSICS-ONLY`), in the
banner and in the provenance panel.

Telemetry: nine SWPC feeds (RTSW 1-min magnetometer and plasma, ACE 1-hour fallback,
Kyoto Dst, planetary Kp, F10.7, GOES X-ray and the 7-day flare list) with explicit
RTSW/ACE schema normalisation, sentinel filtering, a 60 s poll and a `localStorage`
cache that degrades gracefully when storage is isolated.

CME kinematics are **manual input only**. No CORS-enabled feed of `v_lin`, width and
Stonyhurst coordinates was available (DONKI CME timed out; LMSAL HEK returned `403`
without `Access-Control-Allow-Origin`), and the legacy `derivedV0`-style mapping from
X-ray flux is prohibited by the v3 audit. The panel is labelled `MANUAL · NOT TELEMETRY`.

The infrastructure matrix (power grids, LEO drag, polar aviation) is an operational
heuristic over NOAA scales, not a NERC/ICAO-certified product and not a calibrated v3.4
output.

---

## What this package is

This is **not** a state-of-the-art claim. It is a measurement of where the wall is.

Operational CME arrival-time forecasting has sat near ten hours of mean absolute error
for a decade, across more than thirty methods including 3D MHD simulations on
supercomputers. This work separates three effects that are usually reported as one:

- the **model**,
- the **cohort**,
- the **information available at launch**.

The answer, on frozen artefacts and frozen cohorts, is that the cohort effect is about
five times larger than the model effect, and that two of the original design targets are
unreachable with white-light coronagraphy and no interplanetary magnetic field
measurement.

## The four cells

Both models are frozen. Neither was retrained for this study. All four combinations are
reported, including the one that had never been measured: the benchmark model on the
full cohort.

| Cohort | Model | n | MAE (h) | Median AE (h) | RMSE (h) | Bias (h) | Coverage@90 | Mean width (h) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| full (Cycle 25) | served | 126 | 15.4713 | 12.2499 | 20.7469 | −1.4744 | 0.9206 | 74.0419 |
| full (Cycle 25) | bench | 126 | 15.4275 | 11.0394 | 20.8825 | −1.0402 | 0.7857 | 49.2887 |
| frozen 1:1 | served | 91 | 11.5792 | 8.8489 | 16.1284 | −1.4598 | 0.9560 | 71.4393 |
| frozen 1:1 | bench | 91 | 10.5336 | 6.5440 | 14.7272 | −1.4055 | 0.8791 | 49.1864 |

Paired bootstrap, 4,000 resamples, same events for both models:

| Cohort | ΔMAE (served − bench) | 95 % CI | p |
|---|---:|---|---:|
| full (n=126) | +0.0438 h | [−0.8367, +0.9117] | 0.94 |
| frozen 1:1 (n=91) | +1.0456 h | [+0.0967, +1.9837] | 0.027 |

**Reading:** on the full cohort the two models are statistically indistinguishable. On the
clean subset the benchmark model is better by about one hour, with a confidence interval
that barely excludes zero. Changing the cohort moves both models by four to five hours.

The widely quoted 10.5 h and 15.5 h figures are therefore not evidence about model
quality. They are evidence about cohort difficulty. The narrow benchmark interval is also
not transportable: its coverage collapses from 0.8791 on its own cohort to 0.7857 on the
full cohort.

## The two information ceilings

Both were design targets. Neither is reachable, and neither has been redefined after the
fact.

| Gate | Measured | Target | Status |
|---|---:|---:|---|
| 90 % conformal interval width | 74.0419 h at 0.9206 coverage | 35 h at 0.88 | information-infeasible |
| Apex Brier skill score, Dst ≤ −100 nT | 0.0768 (ROC-AUC 0.6980) | 0.25 | information-infeasible |

The tightest *static* band that can reach 88 % coverage on this cohort is 69.3126 h if
constrained to be symmetric and 64.6250 h if allowed to be asymmetric. The
oracle-recalibrated BSS — the best score obtainable by recalibrating on the test labels
themselves, which is not a legitimate forecast — is 0.1500, still below 0.25.

With a residual dispersion of σ ≈ 15–21 h, an 88 % interval needs roughly 3.1 σ, which is
47–65 h before any model error is added. A 35 h interval at 88 % coverage would require
σ ≈ 11 h. The cohort does not have it.

## Tactical stage

Dst forecasting at L1 is a different problem, and it is where the skill actually lives:

| Horizon | RMSE (nT) | Reference |
|---|---:|---|
| +1 h | 3.5014 | persistence 4.6172, raw ODE 4.0099 |
| +6 h | 10.6057 | persistence 13.2073, raw ODE 11.9636 |
| +12 h | 15.1372 | — |

Adaptive conformal inference holds marginal coverage near 0.90 at every horizon
(0.8999 at +1 h, 0.8996 at +6 h, 0.8986 at +12 h), but coverage on storm hours falls
with lead time: 0.8983 at +1 h, 0.8004 at +6 h, 0.6625 at +12 h. The reliable horizon
for L1-driven Dst is about three hours; beyond that only the recovery phase remains
predictable. The strategic/tactical split is what the data supports, not a design
preference.

## Operational context, stated correctly

The CCMC Earth CME Scoreboard for 2023–2024, recomputed for this study directly from the
public API (164 closed CMEs; earliest pre-arrival submission per method):

| Method | MAE (h) | CMEs |
|---|---:|---:|
| NOAA/SWPC ENLIL | 9.72 | 47 |
| Ensemble ENLIL (M2M) | 10.18 | 51 |
| Cone + HAF | 10.20 | 42 |
| Met Office ENLIL | 10.59 | 91 |
| M2M ENLIL | 10.74 | 153 |
| ELEvo | 11.28 | 26 |
| CMEFM | 11.37 | 74 |
| BoM | 11.45 | 40 |
| EAM | 12.04 | 58 |
| SARM | 12.05 | 71 |
| SIDC | 15.23 | 80 |

HELIOS sits mid-table: 11.5792 h on the clean subset, 15.4713 h on the full cohort. Any
claim that it tops this table is false.

The ranking is also not like-for-like. The best entry, 9.72 h, is computed on 47 CMEs
that NOAA chose to forecast, with human adjustment of the cone parameters before
submission. The largest-sample entry, 10.74 h, covers 153 CMEs and is still a
human-supervised operational product. HELIOS has no human in the loop, selects its
features by pre-2020 out-of-fold error, and runs on one consumer CPU core.

## Architecture

Five stages, all local, none of them requiring a cluster:

| Stage | Role | Horizon | Method |
|---|---|---|---|
| 1. Impact | probability that the CME reaches Earth | launch | five-member ensemble (LightGBM, XGBoost, logistic regression, a LambdaRank ranker and a physics-interaction logistic model), source-aware switch, beta calibration |
| 2. Transit | shock arrival time | 12–150 h | LightGBM + XGBoost + ExtraTrees + ridge, blended by inverse out-of-fold MAE, fitted drag-based prior, CQR band |
| 3. Apex | storm severity, P(Dst_min ≤ t) for t ∈ {−50, −100, −200} nT | 24–72 h | heteroscedastic gradient-boosted regressor; the probability map is chosen per threshold by pre-2020 out-of-fold Brier skill, then Platt-aligned; conformal bands |
| 4. Magnetic obstacle | detection of an ICME/MC at L1 and extrapolation of the remaining Bz profile | 0–6 h after MO arrival | MO-Net LightGBM on 5-min OMNI windows, 3DCORE-lite constant-α Lundquist fit, Bz_min calibrator fitted out of fold |
| 5. Tactical | Dst nowcasting at L1 | 1–12 h | Burton–O'Brien–McPherron ODE plus a gradient-boosted residual ensemble, adaptive conformal intervals |

No synthetic magnetic proxy is used anywhere in the v3 pipeline. The legacy v2 proxies
are explicitly banned by an automated test. Every merged feature carries a timestamp at
or before the forecast time; the audit reports zero violations across 41,756 merged rows.

Every stage runs on one CPU core (`n_jobs=1`). No latency figure is claimed here: this
study measures accuracy limits, not runtime, and no timing artefact is shipped.

## Quick start

```bash
pip install -r requirements.txt

# Four-cell limits table (read-only, loads frozen artefacts)
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/v3_15_limits_table.py

# Gate audit: 16/20 pass, 4 expected failures (the targets the paper reports as unmet)
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/v3_10_verify_all.py

# Tests
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 -m unittest discover -s tests -p "test_helios_v3.py"
```

Test result: `Ran 21 tests ... OK (expected failures=2)`. The two expected failures are
exactly the two information-infeasible gates, encoded as tests so that they cannot be
silently "fixed" by loosening a threshold. The suite loads the frozen artefacts in
`models/v3/`, which are included.

## What is not in this package

- **Raw observations** — the raw archive is not shipped. The six processed
  inputs needed to rebuild the reported tables are included in `data/processed/`
  (111.8 MB).
- **Legacy v2 artefacts** — the superseded models, scripts, results and figures are not
  shipped.

The frozen v3 artefacts are included in `models/v3/` (14 files, 72.4 MB).

The limits study lives in `paper/limits_study/`:

| File | Contents |
|---|---|
| `manuscript_limits.md` | the study (raw markdown) |
| `manuscript_limits.pdf` | academic preprint (typeset PDF with figures) |
| `RESULTS_TABLE.md` | generated four-cell table |
| `figures/` | scientific figures (four-cell scatter, conformal bands, tactical horizons) |

## What this work does not claim

- It does not claim state-of-the-art arrival-time accuracy.
- It does not claim that the benchmark model is served. It is not.
- It does not claim that the southward field can be predicted from white light.
- It does not claim that the 35 h interval or the 0.25 BSS target are achievable.
- It does not use synthetic data, synthetic proxies or imputed labels.
- It does not tune any threshold against the Cycle 25 cohort.

## Status

Independent research study deposited on Zenodo under Concept DOI [`10.5281/zenodo.24215019`](https://doi.org/10.5281/zenodo.24215019) (Version 3.4.1: [`10.5281/zenodo.24421551`](https://doi.org/10.5281/zenodo.24421551)). Typeset academic preprint with figures is available at [`paper/limits_study/manuscript_limits.pdf`](paper/limits_study/manuscript_limits.pdf).

## Data attributions

- **NASA SOHO/LASCO CDAW CME Catalog** — NASA GSFC and The Catholic University of
  America, in cooperation with ESA.
- **Richardson & Cane ICME Table** — hosted by Caltech ASC.
- **NASA OMNIWeb** — NASA GSFC Space Physics Data Facility.
- **Kyoto WDC for Geomagnetism** — Dst index.
- **NOAA SWPC** — operational RTSW plasma and magnetic field feeds.
- **NASA SDO/HMI and JSOC SHARP** — source-region magnetism.
- **CCMC Earth CME Scoreboard** — operational context table.

## License

- **Code:** MIT ([`LICENSE`](LICENSE)).
- **Text and data compilations:** CC BY 4.0 ([`LICENSE-DATA.md`](LICENSE-DATA.md)).
