#!/usr/bin/env python3
"""
clean_and_repair_dataset.py — Repairs corrupted labels, fills missing historical Dst values,
and fixes ar_location string parsing for accurate 3D de-projection.
"""

import os
import ast
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(BASE_DIR, "data/processed/helios_vision_matched.parquet")

df = pd.read_parquet(PARQUET_PATH)
print(f"Loaded {len(df)} rows from {PARQUET_PATH}")

# 1. Fix Halloween 2003 label (row with bz_min = -38.5)
h2003_mask = df["t_launch"].astype(str).str.contains("2003-10-29 20:54")
df.loc[h2003_mask, "is_dangerous_southward"] = 1
df.loc[h2003_mask, "is_severe_storm"] = 1
print(f"Fixed Halloween 2003 label: is_dangerous_southward=1 (Bz={df.loc[h2003_mask, 'bz_min'].values[0]})")

# 2. Fill missing historical Dst minima from verified Kyoto WDC records
kyoto_fixes = {
    "2024-05-10": -412.0,  # May 10-11, 2024 Extreme G5 Superstorm
    "2001-03-29": -387.0,  # March 31, 2001 Major G4 Storm
    "2004-07-23": -197.0,  # July 25, 2004 Strong G3/G4 Storm
    "2005-01-05": -103.0,  # January 7, 2005 Moderate G3 Storm
    "2005-05-26": -113.0,  # May 29-30, 2005 Strong G3 Storm
}

for date_prefix, dst_val in kyoto_fixes.items():
    mask = df["t_launch"].astype(str).str.contains(date_prefix)
    df.loc[mask, "dst_min"] = dst_val
    print(f"Set verified Dst={dst_val} nT for {date_prefix} (rows: {mask.sum()})")

# Clean remaining NaNs in dst_min or bz_min with median non-severe baseline
df["dst_min"] = df["dst_min"].fillna(-15.0)
df["bz_min"] = df["bz_min"].fillna(0.0)

# Consistent binary target definition (Dst <= -110 nT)
df["is_severe_storm"] = (df["dst_min"] <= -110.0).astype(int)
df["is_dangerous_southward"] = (df["bz_min"] <= -10.0).astype(int)

# 3. Parse ar_location strings into actual (y, x) numeric tuples
def parse_ar_loc(val):
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (tuple, list, np.ndarray)):
        return val
    if isinstance(val, str):
        val = val.strip()
        if val.startswith("(") and val.endswith(")"):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, tuple) and len(parsed) == 2:
                    return (float(parsed[0]), float(parsed[1]))
            except Exception:
                pass
    return None

df["ar_location_parsed"] = df["ar_location"].apply(parse_ar_loc)
parsed_count = df["ar_location_parsed"].notna().sum()
print(f"Parsed ar_location coordinates: {parsed_count}/{len(df)} events successfully converted to numeric (y, x).")

# 4. Recompute accurate 3D de-projection using parsed active region coordinates
v_3d_list = []
for idx, r in df.iterrows():
    v = float(r.get("v_launch", 500.0))
    loc = r["ar_location_parsed"]
    if loc is not None:
        y, x = loc
        rad = np.sqrt((x - 256.0)**2 + (y - 256.0)**2)
        sin_t = np.clip(rad / 200.0, 0.0, 0.95)
        cos_t = np.sqrt(1.0 - sin_t**2)
    else:
        lat = float(r.get("lat_proxy", 15.0))
        cos_t = np.cos(np.radians(np.clip(lat, 0.0, 75.0)))
    cos_t = np.clip(cos_t, 0.35, 1.0)
    v_3d_list.append(float(v / cos_t))

df["v_launch_3d"] = v_3d_list
m_grams = np.clip(df["mass"].fillna(5e15), 1e12, 1e18)
log_m = np.log10(m_grams)
log_v_cms = np.log10(np.clip(df["v_launch_3d"], 100.0, 4000.0) * 1e5)
df["energy_3d"] = np.log10(0.5) + log_m + 2.0 * log_v_cms
df["flare_flux_proxy"] = np.log10(np.clip(df["v_launch_3d"] * df["f107"] / 100.0, 1.0, 1e6)).astype(np.float32)

# Save repaired dataset
df.to_parquet(PARQUET_PATH)
print(f"✅ Repaired clean dataset saved to {PARQUET_PATH}")
print(f"   • Total severe storms (Dst <= -110 nT): {df['is_severe_storm'].sum()} / {len(df)}")
print(f"   • Total dangerous southward (Bz <= -10 nT): {df['is_dangerous_southward'].sum()} / {len(df)}")

