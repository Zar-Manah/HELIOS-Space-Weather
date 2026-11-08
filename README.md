# ☀️ HELIOS: Open-Source Artificial Intelligence (AI) Models for Real-Time Solar Storm Prediction

> **A Dual-Scale Ensemble of Artificial Intelligence (AI) Models Forecasting Severe Solar Storms, Coronal Mass Ejection (CME) Arrival Times, and Geomagnetic Blackouts from Live Satellite Telemetry.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch MPS](https://img.shields.io/badge/PyTorch-Apple%20Silicon%20MPS-orange.svg)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-CQR-green.svg)](https://lightgbm.readthedocs.io/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24061452-blue.svg)](https://doi.org/10.5281/zenodo.24061452)
[![Made with Passion in Spain](https://img.shields.io/badge/Made%20with%20Passion%20in-Spain%20%F0%9F%87%AA%F0%9F%87%B8-red.svg)](https://github.com/Zar-Manah)

---

## ⚡ What is HELIOS?

**HELIOS** (*Heliospheric Environment & Learning for In-situ Operational Storms*) is an open-source suite of **Artificial Intelligence (AI) models** engineered to predict dangerous solar storms before they trigger catastrophic Geomagnetically Induced Currents (GICs) in terrestrial electric grids, degrade GPS accuracy, or induce high-altitude radiation blackouts for commercial aviation.

⚡ **Outperforming NASA Supercomputers on a Standard Consumer Device:**  
Evaluated strictly out-of-distribution across **Solar Cycle 25 (2020–2026)**, HELIOS achieves an arrival accuracy of **$\pm 7.90$ hours** on fast, geoeffective Coronal Mass Ejections ($\ge 1000$ km/s)—**outperforming NASA's operational WSA-ENLIL 3D MHD supercomputer standard ($\pm 9.20$h) by $+1.30$ hours**. Breaking the barrier of institutional computing, HELIOS runs entirely on a **standard consumer device / personal computer** in milliseconds ($<0.05$s) with zero cloud dependencies or high-performance computing (HPC) clusters.

🤖 **Eliminating Human Bottlenecks & Full Operational Automation:**  
Legacy space weather forecasting relies heavily on slow manual labor—requiring human operators to hand-fit geometric wireframes to coronagraph images, configure simulation grids, and wait hours for supercomputing cluster queues. HELIOS eliminates these operational bottlenecks through **100% autonomous end-to-end execution**: from live multi-satellite telemetry ingestion (NOAA SWPC & NASA at L1) and physical feature derivation to real-time kinematic inference, conformal uncertainty quantification, and automated NASA CCMC Scoreboard export.

Connecting directly to real-time satellite telemetry from **NASA** and **NOAA Space Weather Prediction Center (SWPC)** at the Sun-Earth Lagrangian point L1 (DSCOVR & ACE spacecraft), HELIOS solves the space weather prediction challenge through two synchronized forecasting engines:

1. **🚀 Planetary CME Transit (24–72 Hours Ahead):**  
   When a massive Coronal Mass Ejection (CME) erupts from the solar corona, HELIOS calculates its exact interplanetary travel time and impact severity. Evaluated out-of-distribution across **Solar Cycle 25 (2020–2026)**, HELIOS achieves an arrival accuracy of **$\pm 7.90$ hours on fast geoeffective CMEs ($\ge 1000$ km/s)**, outperforming NASA's operational **WSA-ENLIL numerical MHD supercomputer** standard ($\pm 9.20$h) by **$+1.30$ hours** with sub-second execution ($<0.05$s) on an Apple Silicon SoC.

2. **🧲 Tactical Magnetospheric Nowcasting (1–6 Hours Ahead):**  
   As solar plasma impinges on Earth's bow shock, HELIOS predicts ring current collapse (the Disturbance Storm Time, $Dst$ index) across multi-hour horizons (+1h, +2h, +3h, +6h) using **Inductive Conformal Prediction (CQR)**. This provides electric utility operators with **finite-sample mathematically guaranteed 90% confidence bands**, dynamically expanding during extreme events (such as the historic **May 10–12, 2024 G5 Superstorm**, $Dst = -406$ nT).

---

## 🌟 Key Scientific Innovations

| Innovation | Legacy Approach | HELIOS Solution | Operational Benefit |
| :--- | :--- | :--- | :--- |
| **CME Arrival Accuracy** | Numerical 3D MHD (WSA-ENLIL) $\pm 9.20$h on fast CMEs | Dynamic Mixture-of-Kinematics (MoK) **$\pm 7.90$h** | **$+1.30$ hours earlier notice**; runs in $<0.05$s vs hours on HPC cluster |
| **Uncertainty Bounds** | Deterministic point estimate (no guarantees) | Inductive Conformal Quantile Regression (CQR) | **Mathematically guaranteed 90% coverage** without Gaussian assumptions |
| **Cross-Cycle Validation** | Random shuffle splits (severe temporal leakage) | **Strict Out-of-Distribution split (Solar Cycle 25)** | Tested on 57,998 unseen future hours (2020–2026) with $R^2 = 0.969$ |
| **Extreme Storm Peak** | $L_2$ regression-to-the-mean underestimation | **Helios Weight Surgery (HWS)** projection scaling | Accurately tracks May 2024 G5 peak ($Dst = -406$ nT, TSS = **0.9579**) |
| **Operational Automation** | Manual human wireframe fitting & multi-hour HPC queues | **100% Autonomous End-to-End AI Pipeline** | Eliminates human bottlenecks; sub-second live alerts on a consumer PC |
| **Mission Cockpit** | Heavy command-line Fortran / manual wireframes | **Universal HTML5/Canvas Cockpit (60 FPS)** | Real-time NOAA RTSW sync & 1-click NASA CCMC Scoreboard JSON export |

---

## 🛰️ Live In-Situ Telemetry & Operational Mission Cockpit

HELIOS includes an autonomous, cross-platform mission cockpit (`Zar-Manah_HELIOS.html`) designed for utility command centers, satellite ground stations, and field tablets:
- **Zero Configuration:** Open CORS ingestion directly streaming 1-minute magnetic ($B_x, B_y, B_z, B_t$) and plasma flow ($V_{sw}, N_p, P_{dyn}$) telemetry from NOAA SWPC every 60 seconds.
- **60 FPS Interplanetary Trajectory Canvas:** Visualizes CME propagation through the inner solar system from Sun to Earth with dynamic conformal arrival brackets.
- **NASA CCMC Interoperability:** Generates and exports official standardized JSON payloads for the NASA CCMC CME Scoreboard.

```bash
# Launch interactive Mission Cockpit in default browser
python predict.py --dashboard
# Or simply double-click Zar-Manah_HELIOS.command on macOS
```

---

## 💻 Quick Start & Real-Time Predictions

### Installation
```bash
git clone https://github.com/Zar-Manah/HELIOS-Space-Weather.git
cd HELIOS-Space-Weather
pip install -r requirements.txt
```

### 1. Real-Time Solar Wind Ingestion (1–6h Nowcast)
Fetches live telemetry concurrently from NOAA SWPC and Kyoto WDC:
```bash
python predict.py --live
```
```text
✅ Ingested Multi-Station Live Telemetry (NOAA SWPC + Kyoto WDC):
   • Telemetry Epoch:    2026-11-08 21:06:00 UTC
   • Solar Wind Flow V:  722.8 km/s (1h lag: 719.9 km/s, Δ1h: +2.9 km/s)
   • Proton Density Np:  1.9 cm⁻³ | Pdyn: 1.68 nPa
   • IMF Field (GSM):    Bz = +2.44 nT (3h min: -0.67 nT, 6h min: -1.03 nT)
   • Newell Coupling:    2526 (3h sum: 15896, 6h sum: 35952)
   • Geomagnetic Kyoto:  Dst = -29.0 nT (24h min: -39.0 nT)
------------------------------------------------------------------------------
⚡ ZAR-MANAH HELIOS-TACTICAL: MULTI-HORIZON RING CURRENT (Dst) FORECAST
+1h Lead   | -27.5 nT | [-32.3 nT, -23.0 nT] (±4.6) | 🟢 QUIET / MINOR
+2h Lead   | -25.9 nT | [-33.9 nT, -18.2 nT] (±7.9) | 🟢 QUIET / MINOR
+3h Lead   | -25.3 nT | [-38.2 nT, -16.2 nT] (±11.0) | 🟢 QUIET / MINOR
+6h Lead   | -24.5 nT | [-39.9 nT, -12.1 nT] (±13.9) | 🟢 QUIET / MINOR
```

### 2. Long-Range CME Transit Arrival (24–72h)
When a fast coronal mass ejection is detected by coronagraphs (SOHO/LASCO):
```bash
python predict.py --cme_speed 1800 --halo
```
```text
🚀 ZAR-MANAH HELIOS-TRANSIT v2.5: 24–72H DYNAMIC REGIME ENSEMBLE FORECAST
• Dynamic Hybrid Transit Time: 26.8 hours (1.1 days)
• Kinematic Decomposition:    MoK Neural: 25.7h (90%) | GBDT: 35.9h (10%)
• 90% Conformal Window:       [14.0h, 57.0h] (Margin: ±21.5h)
• Expected Earth Impact:      2026-11-09 23:59 UTC
• Expected Storm Shock (Dst): -152.5 nT
• Threat Level:               🟠 G3/G4 SEVERE GEOMAGNETIC STORM
```

---

## 📁 Repository Structure

```
HELIOS-Space-Weather/
├── Zar-Manah_HELIOS.html             # Universal Mission Cockpit (Auto NOAA L1 Streaming)
├── Zar-Manah_HELIOS.command          # Native macOS double-click launcher
├── predict.py                        # Unified CLI (supports --live, --cme_speed, --dashboard)
├── requirements.txt                  # Python dependencies
├── scripts/
│   ├── 01_build_features.py          # NASA OMNI2 feature extraction (1-6h)
│   ├── 02_train_helios_engine.py     # Multi-horizon GBDT + CQR calibration (1-6h)
│   ├── 03_analyze_may2024_storm.py   # Publication figures 1, 2, 3 (300 DPI)
│   ├── 04_train_neural_lstm.py       # Deep Temporal Bi-LSTM (Apple Silicon MPS)
│   ├── 05_build_paper_pdf.py         # ReportLab 6-Page publication PDF compiler
│   ├── 06_crossmatch_cme_dataset.py  # Cross-matches SOHO/LASCO CDAW & Richardson-Cane
│   ├── 07_train_helios_transit.py    # GBDT CME Transit & Geoeffectiveness engine
│   ├── 08_train_neural_transit.py    # MoKTransitNN PyTorch neural kinematics
│   └── 09_ensemble_transit.py        # Dynamic Regime Ensemble calibration
├── models/                           # 16 GBDT (.joblib) & PyTorch (.pt) weights + configs
├── results/
│   ├── figures/                      # High-resolution benchmark figures
│   └── helios_evaluation_metrics.json# Unseen Cycle 25 OOD quantitative metrics
└── paper/
    ├── manuscript.md                 # Formal peer-reviewed preprint text
    └── HELIOS_Solar_Storm_Conformal_Forecasting.pdf # Complete 6-Page Publication PDF
```

---

## 📜 Citation & Permanent DOI

If you utilize HELIOS in space weather research, electric grid protection, or satellite operations, please cite the official Zenodo scientific preprint:

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

---

*Crafted with passion in Spain 🇪🇸. Distributed under the MIT License.*
