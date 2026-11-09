#!/usr/bin/env python3
"""
HELIOS - Solar Storm Predictor (Pipeline Phase 1)
Extracts, cleans, and engineers physical features from NASA OMNI2 hourly dataset (1998-2026).
Computes Newell coupling, Akasofu epsilon, clock angle, dynamic pressure, and multi-scale temporal lags.
"""

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RAW_FILE = os.path.join(BASE_DIR, "data/raw/omni2_all_years.dat")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/processed")
OUTPUT_PARQUET = os.path.join(OUTPUT_DIR, "helios_omni_features_1998_2026.parquet")
OUTPUT_CSV_SAMPLE = os.path.join(OUTPUT_DIR, "helios_omni_sample.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("☀️  HELIOS SPACE WEATHER PIPELINE: FEATURE ENGINEERING  ☀️")
print("=" * 70)

# Parsing OMNI2 format
print(f"Reading raw data from: {RAW_FILE}")

records = []
total_lines = 0

with open(RAW_FILE, "r") as f:
    for line in f:
        parts = line.split()
        if len(parts) < 42:
            continue
        
        year = int(parts[0])
        if year < 1998:  # Focus on continuous L1 satellite era (Cycle 23, 24, 25)
            continue
            
        doy = int(parts[1])
        hour = int(parts[2])
        
        # Word 9: B magnitude
        b_mag = float(parts[8]) if float(parts[8]) < 999.0 else np.nan
        # Word 16: By GSM, Word 17: Bz GSM
        by_gsm = float(parts[15]) if float(parts[15]) < 999.0 else np.nan
        bz_gsm = float(parts[16]) if float(parts[16]) < 999.0 else np.nan
        # Word 24: Proton Density Np
        np_dens = float(parts[23]) if float(parts[23]) < 999.0 else np.nan
        # Word 25: Flow speed V
        v_flow = float(parts[24]) if float(parts[24]) < 9999.0 else np.nan
        # Word 23: Temperature
        temp = float(parts[22]) if float(parts[22]) < 9999990.0 else np.nan
        # Word 29: Flow pressure Pdyn
        pdyn = float(parts[28]) if float(parts[28]) < 99.0 else np.nan
        # Word 36: Electric field
        e_field = float(parts[35]) if float(parts[35]) < 999.0 else np.nan
        # Word 39: Kp index * 10
        kp = float(parts[38]) / 10.0 if int(parts[38]) < 99 else np.nan
        # Word 40: Sunspot number R
        r_sunspot = float(parts[39]) if int(parts[39]) < 999 else np.nan
        # Word 41: Dst Index (nT)
        dst = float(parts[40]) if int(parts[40]) != 99999 else np.nan
        # Word 51: F10.7 flux
        f107 = float(parts[50]) if float(parts[50]) < 999.0 else np.nan
        
        records.append({
            "year": year,
            "doy": doy,
            "hour": hour,
            "b_mag": b_mag,
            "by_gsm": by_gsm,
            "bz_gsm": bz_gsm,
            "np_dens": np_dens,
            "v_flow": v_flow,
            "temp": temp,
            "pdyn": pdyn,
            "e_field": e_field,
            "kp": kp,
            "r_sunspot": r_sunspot,
            "f107": f107,
            "dst": dst
        })

df = pd.DataFrame(records)
print(f"Loaded {len(df):,} hourly records from 1998 to 2026.")

# Construct continuous datetime index
df["datetime"] = pd.to_datetime(df["year"].astype(str) + "-" + df["doy"].astype(str) + " " + df["hour"].astype(str) + ":00:00", format="%Y-%j %H:%M:%S")
df = df.sort_values("datetime").reset_index(drop=True)

# Interpolate small gaps (up to 3 consecutive missing hours) to maintain continuous time-series
cols_to_interp = ["b_mag", "by_gsm", "bz_gsm", "np_dens", "v_flow", "temp", "pdyn", "dst"]
for c in cols_to_interp:
    df[c] = df[c].interpolate(method="linear", limit=3)

# Filter out rows where essential targets or drivers are still missing
df = df.dropna(subset=["dst", "bz_gsm", "v_flow"]).copy()
print(f"Valid contiguous records after 3h gap interpolation: {len(df):,}")

# -------------------------------------------------------------
# ASTROPHYSICAL FEATURE ENGINEERING
# -------------------------------------------------------------
print("Computing magnetospheric physics features...")

# 1. Transverse Magnetic Field
df["b_transverse"] = np.sqrt(df["by_gsm"]**2 + df["bz_gsm"]**2)

# 2. Clock angle theta (radians from north towards +By)
df["clock_angle"] = np.arctan2(df["by_gsm"], df["bz_gsm"])

# 3. Dynamic pressure recalculated if missing: Pdyn = 1.6726e-6 * Np * V^2 (nPa)
recalc_pdyn = 1.6726e-6 * df["np_dens"] * (df["v_flow"]**2)
df["pdyn"] = df["pdyn"].fillna(recalc_pdyn)

# 4. Half-wave rectifier electric field (convective Ey): Ey = max(0, -V * Bz * 1e-3) mV/m
df["ey_rectified"] = np.maximum(0.0, -df["v_flow"] * df["bz_gsm"] * 1e-3)

# 5. Newell Coupling Function (dPhi/dt = V^(4/3) * Bt^(2/3) * sin^(8/3)(theta/2))
sin_half_theta = np.abs(np.sin(df["clock_angle"] / 2.0))
df["newell_coupling"] = (df["v_flow"] ** (4.0 / 3.0)) * (df["b_transverse"] ** (2.0 / 3.0)) * (sin_half_theta ** (8.0 / 3.0))

# 6. Akasofu parameter epsilon = V * B^2 * sin^4(theta/2)
df["akasofu_eps"] = df["v_flow"] * (df["b_mag"] ** 2) * (sin_half_theta ** 4) * 1e-3

# 7. Borovsky coupling = V * B * sin(theta/2)
df["borovsky_coupling"] = df["v_flow"] * df["b_mag"] * sin_half_theta

# -------------------------------------------------------------
# TEMPORAL MULTI-SCALE LAGS AND ROLLING STATS
# -------------------------------------------------------------
print("Computing temporal multi-scale lags and dynamics...")

# Dst historical memory (crucial for ring current decay dynamics, tau ~ 8-12 hours)
for lag in [1, 2, 3, 6, 12, 24]:
    df[f"dst_lag_{lag}h"] = df["dst"].shift(lag)

# Dst rolling trends
df["dst_mean_6h"] = df["dst"].rolling(window=6, min_periods=3).mean()
df["dst_std_6h"] = df["dst"].rolling(window=6, min_periods=3).std().fillna(0)
df["dst_min_24h"] = df["dst"].rolling(window=24, min_periods=12).min()
df["dst_diff_1h"] = df["dst"] - df["dst_lag_1h"]

# Bz lags and rolling minimum (identifying southward magnetic shock fronts)
for lag in [1, 2, 3, 6]:
    df[f"bz_lag_{lag}h"] = df["bz_gsm"].shift(lag)
df["bz_min_3h"] = df["bz_gsm"].rolling(window=3, min_periods=1).min()
df["bz_min_6h"] = df["bz_gsm"].rolling(window=6, min_periods=1).min()
df["bz_mean_6h"] = df["bz_gsm"].rolling(window=6, min_periods=1).mean()
df["bz_diff_1h"] = df["bz_gsm"] - df["bz_lag_1h"]

# V_flow and Pdyn dynamics
for lag in [1, 2, 3]:
    df[f"v_lag_{lag}h"] = df["v_flow"].shift(lag)
    df[f"pdyn_lag_{lag}h"] = df["pdyn"].shift(lag)

df["v_diff_1h"] = df["v_flow"] - df["v_lag_1h"]
df["pdyn_diff_1h"] = df["pdyn"] - df["pdyn_lag_1h"]

# Coupling integral (accumulated energy input over 3h, 6h, 12h)
df["newell_sum_3h"] = df["newell_coupling"].rolling(window=3, min_periods=1).sum()
df["newell_sum_6h"] = df["newell_coupling"].rolling(window=6, min_periods=1).sum()
df["newell_sum_12h"] = df["newell_coupling"].rolling(window=12, min_periods=1).sum()

# -------------------------------------------------------------
# TARGETS (Forecasting Horizons: +1h, +2h, +3h, +6h)
# -------------------------------------------------------------
print("Constructing multi-horizon forecasting targets...")

for lead in [1, 2, 3, 6]:
    df[f"target_dst_lead_{lead}h"] = df["dst"].shift(-lead)
    # Binary storm flags (NOAA scales: G1+ = Dst <= -50, G3+ = Dst <= -100, G5 = Dst <= -200)
    df[f"target_storm_50_lead_{lead}h"] = (df[f"target_dst_lead_{lead}h"] <= -50).astype(int)
    df[f"target_storm_100_lead_{lead}h"] = (df[f"target_dst_lead_{lead}h"] <= -100).astype(int)

# Solar Cycle assignment for out-of-distribution evaluation
# Cycle 23: 1998 - 2008
# Cycle 24: 2009 - 2019
# Cycle 25: 2020 - 2026
def assign_cycle(year):
    if year <= 2008:
        return "Cycle 23"
    elif year <= 2019:
        return "Cycle 24"
    else:
        return "Cycle 25 (Current OOD)"

df["solar_cycle"] = df["year"].apply(assign_cycle)

# Drop rows with NaN targets due to shifts (beginning and end of dataset)
df_clean = df.dropna().copy().reset_index(drop=True)

print(f"Final clean dataset: {len(df_clean):,} records with {df_clean.shape[1]} features and targets.")
print("\nDataset split by Solar Cycle:")
for cycle, count in df_clean["solar_cycle"].value_counts().items():
    pct = count / len(df_clean) * 100
    print(f"  • {cycle}: {count:,} samples ({pct:.1f}%)")

# Save dataset
print(f"\nSaving processed dataset to: {OUTPUT_PARQUET}")
df_clean.to_parquet(OUTPUT_PARQUET, index=False)
df_clean.head(100).to_csv(OUTPUT_CSV_SAMPLE, index=False)

print("✅ Feature engineering completed successfully!")

