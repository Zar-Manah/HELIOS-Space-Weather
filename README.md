# ☀️ Zar-Manah HELIOS: Full-Spectrum Space Weather Intelligence Suite

> **Zar-Manah Heliospheric Environment & Learning for In-situ & Operational Storms**  
> An open-source, dual-scale space weather architecture covering:
> 1. **Zar-Manah HELIOS-Transit (24–72h):** Long-range CME travel time & shock severity from solar launch kinematics (NASA SOHO/LASCO).
> 2. **Zar-Manah HELIOS-Tactical (1–6h):** Multi-horizon geomagnetic disturbance ($Dst$) nowcasting with distribution-free Conformal Uncertainty intervals (NASA OMNI2 & NOAA SWPC).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch MPS](https://img.shields.io/badge/PyTorch-Apple%20Silicon%20MPS-orange.svg)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-CQR-green.svg)](https://lightgbm.readthedocs.io/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24061452-blue.svg)](https://doi.org/10.5281/zenodo.24061452)
[![Made with Passion in Spain](https://img.shields.io/badge/Made%20with%20Passion%20in-Spain%20%F0%9F%87%AA%F0%9F%87%B8-red.svg)](https://github.com/Zar-Manah)

---

## 🌟 The Two Operational Wings of Zar-Manah HELIOS

### 🚀 1. Zar-Manah HELIOS-Transit (24–72 Hours: Sun $\rightarrow$ Earth Journey)
- **Dataset:** 338 validated Earth-impacting CMEs cross-matched between **NASA SOHO/LASCO CDAW** (42,500+ CMEs) and **Richardson & Cane** (1996–2026).
- **Architecture:** Mixture-of-Kinematics (`MoKTransitNN`) with bifurcated hypersonic bow-shock and subsonic ambient wind branches + analytical weight surgery.
- **Evaluated Out-of-Distribution on Solar Cycle 25 (2020–2026):**
  - **Fast CME Arrival Error ($\ge 1000$ km/s):** **$\pm 7.90$ hours** (Directly outperforming NASA's operational **WSA-ENLIL MHD supercomputer** benchmark of $\pm 9.20$ hours by **$+1.30$ hours**).
  - **Global Arrival Time Error:** **$\pm 11.29$ hours** (Massively outperforms classical analytical Drag-Based Models at $\pm 29.19$ hours by $17.9$h).
  - **90% Conformal Coverage:** 85.9% empirical coverage window across the entire solar cycle.

### 🧲 2. Zar-Manah HELIOS-Tactical (1–6 Hours: L1 Bow Shock $\rightarrow$ Magnetosphere)
- **Dataset:** 245,963 hourly observations from **NASA OMNI2** (Solar Cycles 23–25).
- **Physical Couplings:** Newell coupling ($d\Phi/dt$), Akasofu epsilon ($\epsilon$), clock angle ($\theta$), dynamic pressure ($P_{dyn}$), half-wave rectified $E_y$.
- **Performance on Unseen Cycle 25 (2020–2026):**
  - **+1h Lead:** RMSE 3.57 nT | $R^2 = 0.969$ | TSS = **0.9579** (via Helios Weight Surgery)
  - **+2h Lead:** RMSE 5.83 nT | $R^2 = 0.918$ | TSS = 0.830
  - **+3h Lead:** RMSE 7.52 nT | $R^2 = 0.864$ | TSS = 0.765
  - **+6h Lead:** RMSE 10.62 nT | $R^2 = 0.729$ | TSS = 0.642
  - **Conformal 90% Calibration:** Exactly **89.94% (+1h), 90.18% (+2h), 89.76% (+3h), 88.88% (+6h)**.

---

## 🚀 Unified CLI Usage

### 1. Long-Range CME Transit Forecast (24–72h)
When a CME is observed exploding from the Sun:
```bash
python predict.py --cme_speed 1600 --halo
```
*Output: Predicted travel duration (hours), exact expected arrival timestamp at Earth, 90% conformal window, and estimated Dst shock severity.*

### 2. Tactical Live Satellite Nowcasting (1–6h)
Connects live to NOAA Space Weather Prediction Center (SWPC) DSCOVR telemetry:
```bash
python predict.py --live
```

### 3. Universal Mission Cockpit (Zero Configuration)
Launch the interactive mission-control cockpit across any desktop, tablet, or mobile browser:
```bash
python predict.py --dashboard
# Or simply double-click Zar-Manah_HELIOS.command on macOS
```

---

## 📁 Repository Structure

```
SOLAR-HELIOS-01/
├── Zar-Manah_HELIOS.html             # Universal Mission Cockpit (Auto NOAA L1 Streaming)
├── Zar-Manah_HELIOS.command          # Native macOS double-click launcher
├── predict.py                        # Unified CLI (supports --live, --cme_speed, --dashboard)
├── requirements.txt                  # Python dependencies
├── scripts/
│   ├── 01_build_features.py          # NASA OMNI2 feature extraction (1-6h)
│   ├── 02_train_helios_engine.py     # Multi-horizon GBDT + CQR calibration (1-6h)
│   ├── 03_analyze_may2024_storm.py   # Publication figures 1, 2, 3 (300 DPI)
│   ├── 04_train_neural_lstm.py       # Deep Temporal Bi-LSTM (Apple Silicon MPS)
│   ├── 05_build_paper_pdf.py         # ReportLab PDF compiler
│   ├── 06_crossmatch_cme_dataset.py  # Cross-matches SOHO/LASCO CDAW & Richardson-Cane
│   └── 07_train_helios_transit.py    # Trains 24-72h CME Transit & Geoeffectiveness engine
├── data/
│   ├── raw/                          # OMNI2 (175MB), CDAW (5.2MB), ICME catalog
│   └── processed/                    # Clean parquet datasets
├── models/                           # Checkpoints (Point, CQR quantiles, Bi-LSTM, Transit)
├── results/
│   ├── figures/                      # High-resolution figures 1, 2, 3, 4
│   └── helios_transit_metrics.json
└── paper/
    ├── manuscript.md                 # Formal preprint text
    └── HELIOS_Solar_Storm_Conformal_Forecasting.pdf # Complete 1.8 MB PDF
```

---

## 📜 Citation

```bibtex
@article{manah2026zarmanahhelios,
  title={Zar-Manah HELIOS: Dual-Scale Space Weather Intelligence Coupling Tactical Conformal Nowcasting (1–6h) and Planetary CME Shock Transit (24–72h) across Solar Cycles 23–25},
  author={Manah, Zar},
  journal={Zenodo Preprints},
  year={2026},
  publisher={Zenodo / CERN},
  doi={10.5281/zenodo.24061452},
  url={https://doi.org/10.5281/zenodo.24061452}
}
```
