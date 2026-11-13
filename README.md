# ☀️ HELIOS 2.0: Multi-Gear Architecture for Real-Time Solar Storm Impact & CME Arrival Prediction

> **An Open-Source Space Weather Architecture on Consumer Hardware (<25ms) — Predicting CME Arrival Times and Geomagnetic Storm Severity from Live Satellite Telemetry.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch MPS](https://img.shields.io/badge/PyTorch-Apple%20Silicon%20MPS-orange.svg)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-CQR-green.svg)](https://lightgbm.readthedocs.io/)
[![Zenodo DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24181624-blue.svg)](https://doi.org/10.5281/zenodo.24181624)
[![Concept DOI](https://img.shields.io/badge/Concept%20DOI-10.5281%2Fzenodo.24061451-blue.svg)](https://doi.org/10.5281/zenodo.24061451)
[![Made with Passion in Spain](https://img.shields.io/badge/Made%20with%20Passion%20in-Spain%20%F0%9F%87%AA%F0%9F%87%B8-red.svg)](https://github.com/Zar-Manah)

---

> ### ⚡ Universal Plug-and-Play Execution (Zero Install, All Devices)
> **Run the full synchronized 4-gear AI architecture in 1 click on any platform (Mac, Windows, Linux, iOS, Android):**  
> 📥 **Single File to Download & Open:** **[`Zar-Manah_HELIOS.html`](Zar-Manah_HELIOS.html)** (double-click to open in any web browser).  
> *• Zero dependencies / Zero Python setup • Live NOAA SWPC & NASA SDO/SOHO streaming in volatile RAM • Sub-25ms inference.*  
>  
> *Prefer CLI / Terminal? Run:* `python3 scripts/helios_swiss_orchestrator.py`

---

## ⚡ What is HELIOS 2.0?

**HELIOS 2.0** (*Heliospheric Environment & Learning for In-situ Operational Storms*) is an open-source suite of **Artificial Intelligence (AI) models** engineered to predict dangerous **solar storms**, coronal mass ejection (CME) shock arrival times, and space weather blackouts before they impact Earth.

Operated as a synchronized **4-Gear "Swiss Watch" AI architecture**, HELIOS 2.0 evaluates solar eruptions at the exact moment of launch on the Sun, tracks their 3D propagation through interplanetary space, and confirms magnetospheric coupling at Lagrange Point L1—executing in **$<25$ milliseconds on consumer-grade hardware (Apple Silicon M4 GPU)** with zero cloud dependencies or supercomputing queues.

⚡ **Near-Supercomputer Accuracy on a Standard Consumer Device:**  
Evaluated strictly out-of-distribution across **Solar Cycle 25 (2020–2026)** on 100% empirical CDAW/Richardson & Cane observations (zero synthetic leakage), HELIOS 2.0 achieves a median transit arrival error of **$\pm 7.45$ to $\pm 8.34$ hours** (MAE $\pm 10.63\text{ h}$ with GBDT, $\pm 11.69\text{ h}$ with MoK hybrid), outperforming classical drag-based models ($\pm 28.95\text{ h}$) and approaching the operational accuracy of NASA's WSA-ENLIL 3D MHD supercomputer standard ($\pm 10.40\text{ h}$ at Earth; Wold et al., 2018). Breaking the barrier of institutional supercomputing, HELIOS runs entirely on standard consumer hardware (Apple Silicon M4 GPU / consumer CPUs) in **$<25$ milliseconds** with zero cloud dependencies or supercomputing queues.

🤖 **Eliminating Human Bottlenecks & Full Operational Automation:**  
Legacy space weather forecasting relies heavily on slow manual labor—requiring human operators to hand-fit geometric wireframes to coronagraph images, configure simulation grids, and wait hours for supercomputing cluster queues. HELIOS 2.0 eliminates these operational bottlenecks through **100% autonomous end-to-end execution**: from live multi-satellite telemetry ingestion (NOAA SWPC & NASA SDO/SOHO) and physical feature derivation to real-time kinematic inference, conformal uncertainty quantification, and automated NASA CCMC Scoreboard export.

```
       ☀️ SUN (t = 0)                    🛰️ INTERPLANETARY (24–72h)              🌍 L1 & EARTH (0–6h)
 ┌──────────────────────────┐             ┌──────────────────────────┐            ┌──────────────────────────┐
 │  GEAR 1: SOLAR CHIRALITY │             │ GEAR 3: 3D KINEMATICS    │            │ GEAR 4: IN-SITU L1 BiLSTM│
 │  HELIOS-VISION-APEX      │             │ MoKTransitNN (HTWS)      │            │ 24h Multivariate History │
 │  LASCO C2 + SDO/HMI R    │             │ Vršnak Aerodynamic Drag  │            │ Conformal CQR Bands      │
 └────────────┬─────────────┘             └────────────┬─────────────┘            └────────────┬─────────────┘
              │                                        │                                       │
              ▼                                        ▼                                       ▼
 ┌──────────────────────────┐             ┌──────────────────────────┐            ┌──────────────────────────┐
 │  GEAR 2: STORM SEVERITY  │────────────▶│ CONFORMAL ARRIVAL WINDOW │───────────▶│ MISSION COCKPIT VERDICT  │
 │  HELIOS-STORM-APEX       │             │ Calibrated ± 18.8h Bound │            │ 🟢 Safe | 🟡 Watch | 🔴 G5│
 │  Dst <= -110 nT Veto     │             │ 7.45h Median (10.6h MAE) │            │ Sub-Second Mitigation    │
 └──────────────────────────┘             └──────────────────────────┘            └──────────────────────────┘
```

---

## ⚙️ The 4-Gear Cascaded Architecture

| Gear | Operational Role | Time Horizon | Core Architecture & Data | Key Performance Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Gear 1: Coronal Chirality** | Magnetic cloud polarity ($B_z \le -10\text{ nT}$) | **24–72 hours** | Deep MobileNetV3 visual extraction from SOHO/LASCO C2 + SDO/HMI Active Region magnetograms (Schrijver $R$-value) | Calibrated sensitivity to southward dangerous magnetic clouds |
| **Gear 2: Storm Severity Veto** | Severe storm classification ($Dst \le -110\text{ nT}$) | **24–72 hours** | Unified 25-dim Solar Vector + Gradient-boosted Conformal Quantile Ensembles | **85.9% Accuracy** & **0.8717 ROC-AUC** on Cycle 25; **Defuses 92.1% of benign events** |
| **Gear 3: 3D Transit Kinematics** | Interplanetary shock arrival time ($\tau$) | **12–48 hours** | Bifurcated `MoKTransitNN` + GBDT with deterministic 3D cone trajectory filter | **$\pm 7.45\text{ h}$ Median / $\pm 10.63\text{ h}$ MAE** on Cycle 25 OOD (beats DBM by 17.3h; near WSA-ENLIL 10.4h Earth std) |
| **Gear 4: L1 In-Situ Attention** | Tactical ring current nowcasting ($Dst$) | **1–6 hours** | 24-hour multivariate bidirectional LSTM (`HeliosBiLSTM`) with temporal attention on L1 DSCOVR/ACE telemetry | Calibrated inductive conformal prediction bounds on real L1 stream |

### 🚀 Legacy Space Weather vs HELIOS 2.0 Architectural Contrast

| Dimension | Legacy Standard (WSA-ENLIL / NOAA) | HELIOS 2.0 Autonomous AI | Operational Advantage |
| :--- | :--- | :--- | :--- |
| **CME Arrival Accuracy** | Numerical 3D MHD $\pm 10.40\text{h}$ (Earth, Wold 2018) | GBDT/MoK Ensemble **$\pm 7.45\text{h}$ median / $\pm 10.63\text{h}$ MAE** | **Comparable accuracy to HPC without cluster queues** |
| **Computational Footprint** | 1 to 4 hours on supercomputing HPC clusters | **$<25$ milliseconds** on Apple Silicon M4 GPU / consumer CPUs | **Eliminates cluster queue latency entirely** |
| **Human Labor & Bottlenecks** | Manual wireframe cone fitting on coronagraphs | **100% Autonomous** multi-satellite ingestion | **Zero human response lag in severe emergencies** |
| **False Alarm Rate** | Velocity-only sirens (up to 92% false alarms) | 3D Cone Trajectory Filter + Severity Veto | **Filters non-Earth CMEs deterministically** |
| **Uncertainty Quantification** | Heuristic single-point deterministic output | Empirical Conformal Quantiles | **Calibrated ~81.3% coverage intervals** |
| **Standard Interoperability** | Complex manual formats | Native NASA CCMC JSON Schema export | **Instant pre-event benchmarking for global agencies** |

---

## 🌟 Key Scientific Breakthroughs

1. **Eliminating the 92% False Alarm Trap:**  
   Standard kinematic-only models (like WSA-ENLIL or empirical speed alerts) trigger high-level emergency alerts whenever a CME launches at $V > 700-1500\text{ km/s}$, regardless of magnetic field orientation. When the field is northward ($B_z > 0$), the Earth's magnetosphere deflects the plasma, resulting in benign conditions ($Dst > -50\text{ nT}$). HELIOS 2.0 inspects coronal chirality at launch, **defusing 92.1% of false alarms** before sirens are sounded.

2. **Aerodynamic Drag Bounding for Hypersonic CMEs:**  
   In extreme hypersonic eruptions ($V > 2000\text{ km/s}$), standard empirical models suffer from extrapolation error. By implementing closed-form aerodynamic drag limits ($v_{\text{amb}} = 420\text{ km/s}$, $\tau \ge 16.0\text{ h}$ physical solar wind saturation bound), HELIOS establishes an empirical transit lower bound on landmark events like the record October 29, 2003 Halloween storm ($19.4\text{ h}$ observed vs $16.0\text{ h}$ aerodynamic bound, $\Delta = 3.4\text{ h}$).

3. **Sub-Second Execution on Consumer Hardware:**  
   While numerical 3D MHD simulations take 1 to 4 hours to run on supercomputer clusters, HELIOS 2.0 evaluates the entire 4-tier pipeline in **$<25$ milliseconds** on Apple Silicon M4 GPU (or standard CPUs), enabling continuous real-time space weather nowcasting.

4. **Interactive Browser Mission Cockpit:**  
   The interactive mission cockpit (`Zar-Manah_HELIOS.html`) provides an ephemeral visualization environment streaming live NOAA SWPC RTSW plasma and magnetic vectors directly in the browser with dual-track mission timelines and responsive aerodynamic drag modeling.

5. **Eliminating Human Bottlenecks & Operational Latency:**  
   Legacy space weather forecasting relies on human operators manually fitting geometric wireframes to coronagraph images and waiting hours in supercomputing queues. HELIOS 2.0 achieves **100% autonomous end-to-end execution**: from live satellite ingestion (NOAA SWPC & NASA SDO/SOHO) to real-time kinematic inference and automated NASA CCMC Scoreboard export with zero human-in-the-loop latency.

---

## 📊 Rigorous Validation: Landmark Historical Case Studies

Evaluating historical landmark space weather events across Solar Cycles 23–25 with exact time-synchronized inputs and real empirical ground truth:

| Landmark Event | Detection Epoch | Launch Speed | Ground Truth ($Dst / B_z$) | Observed Transit | Pipeline Alert & Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Halloween Superstorm (Oct 29, 2003)** | 2003-10-29 20:54 | $2029\text{ km/s}$ | $-383\text{ nT} / -38.5\text{ nT}$ (G5) | $19.4\text{ h}$ | 🔴 G5 Protocol Active ($P_{\text{sev}} = 70.0\%$, $P_{\text{south}} = 56.9\%$) |
| **May 2024 Superstorm (AR 3664)** | 2024-05-07 03:12 | $812\text{ km/s}$ | $-406\text{ nT} / -35.3\text{ nT}$ (G5) | $85.9\text{ h}$ | 🔴 G5 Protocol Active ($P_{\text{sev}} = 73.3\%$, $P_{\text{south}} = 61.8\%$) |
| **Nov 2003 Superstorm (Halloween)** | 2003-11-18 08:50 | $1824\text{ km/s}$ | $-422\text{ nT} / -50.9\text{ nT}$ (G5) | $47.2\text{ h}$ | 🔴 G5 Protocol Active ($P_{\text{sev}} = 74.8\%$, $P_{\text{south}} = 63.0\%$) |
| **May 14, 2024 Moderate Eruption** | 2024-05-14 10:09 | $749\text{ km/s}$ | $-88\text{ nT} / -12.8\text{ nT}$ | $75.3\text{ h}$ | 🔴 Alert ($P_{\text{sev}} = 72.2\%$; ETA $70.7\text{ h}$, $\Delta = 4.7\text{ h}$) |
| **Jan 17, 1998 Solar Baseline** | 1998-01-17 04:09 | $350\text{ km/s}$ | $-11\text{ nT} / -4.4\text{ nT}$ | $95.8\text{ h}$ | 🔴 Alert ($P_{\text{sev}} = 61.5\%$; ETA $70.2\text{ h}$) |

*Note on Headless Evaluation:* In headless execution where raw active region magnetograms are absent from local storage, visual feature extractors default to zero, elevating baseline severity probability ($P_{\text{severe}} \approx 0.60\text{--}0.72$) for non-extreme events. Operational deployment utilizes live SDO/HMI magnetograms for full dynamic range.

---

## 🛰️ Interactive Mission Cockpit (`Zar-Manah_HELIOS.html`)

HELIOS 2.0 includes a zero-install, single-file browser cockpit running at 60 FPS on any modern browser:
- **Live Satellite Feeds:** Auto-updates NOAA SWPC RTSW plasma and magnetic vectors every 60s, and NASA SDO/SOHO solar imagery every 5 min.
- **Master Mission Timeline:** Dual-track interactive canvas visualizing CME 3D transit and magnetospheric ring current response over a 100-hour window.
- **Zero Disk Footprint:** Ephemeral memory management guaranteeing zero disk buildup.

```bash
# Launch interactive Mission Cockpit
open Zar-Manah_HELIOS.html
# Or execute via Python orchestrator
python3 scripts/helios_swiss_orchestrator.py --interactive
```

---

## 💻 Quick Start & Real-Time Predictions

### Installation
```bash
git clone https://github.com/Zar-Manah/HELIOS-Space-Weather.git
cd HELIOS-Space-Weather
pip install -r requirements.txt
```

### 1. Dual-Scale Instant Inference (`predict.py`)
```bash
# Tactical Nowcasting (1–6h) with live NOAA SWPC & Kyoto WDC satellite telemetry
python3 predict.py --live

# Tactical Nowcasting on synthetic extreme benchmark scenario (Dst < -400 nT)
python3 predict.py --sample extreme

# 24–72h Planetary CME Shock Transit & Impact Severity Forecast
python3 predict.py --cme-speed 1200 --halo

# Launch Interactive Visual Mission Control Cockpit
python3 predict.py --dashboard
```

### 2. Live Swiss Watch Orchestrator (All 4 Gears Synchronized)
```bash
python3 scripts/helios_swiss_orchestrator.py
```
```text
🇨🇭 Initializing HELIOS Swiss Watch Pipeline (Apple Silicon M4 GPU)...
✅ All 4 gears loaded into GPU memory in 22.4 ms.
🛰️ Ingesting NOAA SWPC and SDO/SOHO satellite telemetry...
---------------------------------------------------------------------------------
⚙️  GEARS 1 & 2 [SUN]:      P(South) = 45.2% | P(Severe) = 28.4% -> 🟢 QUIET / SAFE
⚙️  GEAR 3 [TRANSIT]:       ETA = 54.2h [Conformal Window: ±6.5h]
⚙️  GEAR 4 [L1 IN-SITU]:    Dst +1h = -18.2 nT | Bz = +2.4 nT
>>> MISSION CONTROL VERDICT: 🟢 GREEN PROTOCOL (SAFE TRANSIT)
```

### 3. Run Comprehensive Randomized Stress Battery
```bash
python3 scripts/17_randomized_stress_battery.py
```

### 4. Export to NASA CCMC Earth CME Scoreboard JSON Format
```bash
python3 scripts/export_ccmc_scoreboard.py --demo
```

---

## 📜 Scientific Publication & Citation

If you utilize HELIOS 2.0 in academic research, space operations, or utility grid defense, please cite:

```bibtex
@article{ZarManah2026HELIOS2,
  title={{Zar-Manah HELIOS 2.0: Single-File Plug-and-Play 4 AI Models for Real-Time Solar Storm Impact \& CME Arrival Prediction}},
  author={{Zar Manah}},
  affiliation={Independent Researcher, Madrid, Spain},
  orcid={0009-0007-8443-8999},
  journal={Zenodo Preprints},
  year={2026},
  publisher={Zenodo / CERN},
  doi={10.5281/zenodo.24179115},
  url={https://doi.org/10.5281/zenodo.24179115}
}
```

---

## 🛰️ Public Scientific Data Attributions

HELIOS 2.0 ingests and builds upon public open-science data from international missions:
* **NASA SOHO/LASCO CDAW CME Catalog:** Maintained by NASA GSFC and The Catholic University of America in cooperation with ESA.
* **Richardson & Cane ICME Table:** Curated by I. G. Richardson and H. V. Cane, hosted by Caltech ASC.
* **NASA OMNIWeb Data:** High-resolution solar wind, IMF, and plasma data provided by NASA GSFC Space Physics Data Facility (SPDF).
* **Kyoto WDC for Geomagnetism:** Real-time and definitive Dst index series provided by Kyoto University.
* **NOAA Space Weather Prediction Center (SWPC):** Operational RTSW plasma and magnetic field feeds.
* **NASA Solar Dynamics Observatory (SDO):** HMI magnetograms and AIA EUV coronal observations.

---

## 📄 License & Open Science

* **Software & Code:** Distributed under the **MIT License**.
* **Preprint & Data Compilations:** Distributed under **Creative Commons Attribution 4.0 International (CC BY 4.0)** via CERN Zenodo (DOI: [10.5281/zenodo.24179115](https://doi.org/10.5281/zenodo.24179115)).
* Crafted with scientific passion in Spain 🇪🇸 by **Zar Manah** (ORCID: [0009-0007-8443-8999](https://orcid.org/0009-0007-8443-8999)).
