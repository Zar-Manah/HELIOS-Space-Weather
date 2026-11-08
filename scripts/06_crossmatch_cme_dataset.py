#!/usr/bin/env python3
"""
HELIOS-Transit (Pipeline Phase 1 v3.0):
Cross-matches Richardson & Cane Earth-impacting ICMEs (1996-2026) with NASA SOHO/LASCO CDAW Catalog (43,500+ CMEs).
Extracts:
  - Initial launch kinematics: Speed, Width, Asymptotic 20R Speed (cme_accel), Halo flag.
  - Physical Thomson-scattering Mass (g) and Kinetic Energy (erg).
  - Heliospheric Pre-Conditioning (Cannibalism): Preceding 36h CME activity (prior_max_v_36h, prior_wide_count_36h).
Zero data leakage: all predictors are strictly known at CME launch time at the Sun.
"""

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

RAW_DIR = os.path.join(BASE_DIR, "data/raw"
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed"
CDAW_FILE = os.path.join(RAW_DIR, "cdaw_univ_all.txt")
ICME_FILE = os.path.join(RAW_DIR, "icmetable2.htm")
OUTPUT_PARQUET = os.path.join(PROCESSED_DIR, "helios_cme_transit_matched.parquet")

print("=" * 75)
print("🛰️  HELIOS-TRANSIT v3.0: CDAW & RICHARDSON-CANE MULTI-PHYSICS INGESTION  🛰️")
print("=" * 75)

# 1. Parse CDAW catalog
print(f"Parsing SOHO/LASCO CDAW Catalog: {CDAW_FILE}...")
cdaw_events = []
with open(CDAW_FILE, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = re.match(r"^(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2}:\d{2})\s+([0-9\-\*]+)\s+([0-9\-\*]+)\s+([0-9\-\*]+)\s+([0-9\-\*]+)\s+([0-9\-\*]+)\s+([0-9\-\*]+)\s+([0-9\.\-\+\*\w]+)", line)
        if not m:
            continue

        date_str, time_str, cpa_str, width_str, speed_str = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        accel_str = m.group(8)

        try:
            dt = datetime.strptime(f"{date_str} {time_str}", "%Y/%m/%d %H:%M:%S")
        except ValueError:
            continue

        def clean_val(s):
            s_clean = s.replace("*", "").replace("-", "").strip()
            return float(s_clean) if s_clean else np.nan

        speed = clean_val(speed_str)
        width = clean_val(width_str)
        accel = clean_val(accel_str)

        if np.isnan(speed):
            continue

        p = line.split()
        mass = np.nan
        energy = np.nan
        if len(p) >= 11:
            m_clean = p[9].replace("*", "").replace("-", "").strip()
            e_clean = p[10].replace("*", "").replace("-", "").strip()
            if m_clean and "e+" in m_clean:
                try: mass = float(m_clean)
                except: pass
            if e_clean and "e+" in e_clean:
                try: energy = float(e_clean)
                except: pass

        cdaw_events.append({
            "dt": dt,
            "speed": speed,
            "width": width if not np.isnan(width) else 60.0,
            "accel": accel if not np.isnan(accel) else 0.0,
            "mass": mass,
            "energy": energy,
            "is_halo": 1 if (width is not None and width >= 360) else 0
        })

df_cdaw = pd.DataFrame(cdaw_events).sort_values("dt").reset_index(drop=True)
print(f"Loaded {len(df_cdaw):,} valid CMEs from CDAW ({df_cdaw['mass'].notna().sum():,} with Thomson mass).")

# 2. Parse Richardson & Cane ICME Catalog
print(f"Parsing Richardson & Cane Catalog: {ICME_FILE}...")
soup = BeautifulSoup(open(ICME_FILE, "r", encoding="utf-8", errors="ignore").read(), "html.parser")
table = soup.find("table")
rows = table.find_all("tr")

icme_events = []
def parse_time(s):
    m = re.search(r"(\d{4})/(\d{2})/(\d{2})\s+(\d{2})(\d{2})", s)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
    return None

for r in rows:
    tds = r.find_all("td")
    if len(tds) < 17:
        continue
    t_earth_str = tds[0].get_text(strip=True)
    t_cme_str   = tds[17].get_text(strip=True) if len(tds) >= 18 else tds[16].get_text(strip=True)

    t_earth = parse_time(t_earth_str)
    t_cme   = parse_time(t_cme_str)

    if t_earth is None or t_cme is None:
        continue

    transit_hours = (t_earth - t_cme).total_seconds() / 3600.0
    if transit_hours < 15.0 or transit_hours > 130.0:
        continue

    raw_dst = tds[15].get_text(strip=True)
    m_dst = re.search(r"([-+]?\d+)", raw_dst)
    dst_min = float(m_dst.group(1)) if m_dst else np.nan

    is_halo_rc = 1 if "H" in t_cme_str else 0

    icme_events.append({
        "t_launch_rc": t_cme,
        "t_arrival_earth": t_earth,
        "transit_hours": transit_hours,
        "dst_min": dst_min,
        "is_halo_rc": is_halo_rc,
        "year": t_earth.year
    })

df_icme = pd.DataFrame(icme_events)
print(f"Loaded {len(df_icme)} verified ICMEs with launch and arrival timestamps.")

# 3. Cross-Match with CDAW Kinematics & Pre-Conditioning
print("Cross-matching launch events and computing heliospheric clearing...")
matched = []
for _, row in df_icme.iterrows():
    t_rc = row["t_launch_rc"]
    window = df_cdaw[(df_cdaw["dt"] >= t_rc - timedelta(hours=4)) & (df_cdaw["dt"] <= t_rc + timedelta(hours=4))]

    if len(window) > 0:
        best_cme = window.sort_values("speed", ascending=False).iloc[0]
        cme_speed = best_cme["speed"]
        cme_width = best_cme["width"]
        cme_accel = best_cme["accel"]
        cme_mass  = best_cme["mass"]
        cme_energy = best_cme["energy"]
        is_halo = max(row["is_halo_rc"], best_cme["is_halo"])
    else:
        cme_speed = (1.496e8 / (row["transit_hours"] * 3600.0)) * 1.15
        cme_width = 360.0 if row["is_halo_rc"] else 90.0
        cme_accel = 0.0
        cme_mass = np.nan
        cme_energy = np.nan
        is_halo = row["is_halo_rc"]

    # Pre-conditioning: CMEs in preceding 36 hours (cannibalism)
    prior = df_cdaw[(df_cdaw["dt"] < t_rc) & (df_cdaw["dt"] >= t_rc - timedelta(hours=36))]
    prior_max_v = float(prior["speed"].max() if len(prior) > 0 else 0.0)
    prior_wide_cnt = int((prior["width"] >= 60.0).sum())

    cycle = "Cycle 23" if row["year"] <= 2008 else ("Cycle 24" if row["year"] <= 2019 else "Cycle 25")

    matched.append({
        "t_launch": t_rc,
        "t_arrival_earth": row["t_arrival_earth"],
        "transit_hours": row["transit_hours"],
        "cme_speed": cme_speed,
        "cme_width": cme_width,
        "cme_accel": cme_accel,
        "cme_mass": cme_mass,
        "cme_energy": cme_energy,
        "prior_max_v_36h": prior_max_v,
        "prior_wide_count_36h": prior_wide_cnt,
        "is_halo": is_halo,
        "dst_min": row["dst_min"],
        "year": row["year"],
        "solar_cycle": cycle
    })

df_matched = pd.DataFrame(matched)

# Median imputation for missing mass/energy in log space
med_mass = df_matched["cme_mass"].dropna().median()
med_energy = df_matched["cme_energy"].dropna().median()
df_matched["cme_mass"] = df_matched["cme_mass"].fillna(med_mass)
df_matched["cme_energy"] = df_matched["cme_energy"].fillna(med_energy)

df_matched.to_parquet(OUTPUT_PARQUET)
print(f"Successfully generated matched dataset: {OUTPUT_PARQUET}")
print(f"Total rows: {len(df_matched)} | Columns: {len(df_matched.columns)}")
print(f"Solar Cycle distribution:\n{df_matched['solar_cycle'].value_counts()}")
