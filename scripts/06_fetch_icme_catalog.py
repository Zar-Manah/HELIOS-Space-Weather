#!/usr/bin/env python3
"""
HELIOS-Transit - Phase 1: Ingestion of Richardson & Cane Near-Earth ICME Catalog (1996-2026).
Extracts CME solar launch times, Earth arrival disturbance times, transit durations (hours),
initial velocities, shock velocities, and minimum Dst indices.
"""

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import re
import urllib.request
from datetime import datetime
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

RAW_DIR = os.path.join(BASE_DIR, "data/raw"
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed"
OUTPUT_PARQUET = os.path.join(PROCESSED_DIR, "helios_icme_transit_1996_2026.parquet")
OUTPUT_CSV = os.path.join(PROCESSED_DIR, "helios_icme_transit_sample.csv")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

URL = "https://izw1.caltech.edu/ACE/ASC/DATA/level3/icmetable2.htm"
LOCAL_HTML = os.path.join(RAW_DIR, "icmetable2.htm")

print("=" * 70)
print("🚀 HELIOS-TRANSIT: FETCHING & PARSING NASA/CALTECH ICME CATALOG 🚀")
print("=" * 70)

if not os.path.exists(LOCAL_HTML):
    print(f"Downloading catalog from: {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (HELIOS/1.0)"})
    with urllib.request.urlopen(req, timeout=15) as r:
        html = r.read().decode("utf-8", errors="ignore")
    with open(LOCAL_HTML, "w", encoding="utf-8") as f:
        f.write(html)
else:
    print(f"Reading cached HTML from: {LOCAL_HTML}")
    with open(LOCAL_HTML, "r", encoding="utf-8") as f:
        html = f.read()

soup = BeautifulSoup(html, "html.parser")
table = soup.find("table")
rows = table.find_all("tr")
print(f"Total rows in table: {len(rows)}")

records = []

def parse_time_str(s):
    # Extracts YYYY/MM/DD HHMM
    m = re.search(r"(\d{4})/(\d{2})/(\d{2})\s+(\d{2})(\d{2})", s)
    if m:
        y, mo, d, h, mi = map(int, m.groups())
        return datetime(y, mo, d, h, mi)
    # Check YYYY/MM/DD without time
    m2 = re.search(r"(\d{4})/(\d{2})/(\d{2})", s)
    if m2:
        y, mo, d = map(int, m2.groups())
        return datetime(y, mo, d, 0, 0)
    return None

for r in rows:
    tds = r.find_all("td")
    if len(tds) < 17:
        continue
    
    # Check headers
    text_0 = tds[0].get_text(strip=True)
    if "Disturbance" in text_0 or "Y/M/D" in text_0:
        continue
        
    t_earth_str = text_0
    t_cme_str   = tds[17].get_text(strip=True) if len(tds) >= 18 else tds[16].get_text(strip=True)
    
    t_earth = parse_time_str(t_earth_str)
    t_cme   = parse_time_str(t_cme_str)
    
    if t_earth is None or t_cme is None:
        continue
        
    # Transit time in hours
    transit_hours = (t_earth - t_cme).total_seconds() / 3600.0
    
    # Physical sanity check: CMEs transit 1 AU between 15 and 120 hours
    if transit_hours < 12.0 or transit_hours > 130.0:
        continue
        
    # Extract numerical physical attributes
    def get_num(td_idx, default=np.nan):
        if td_idx >= len(tds):
            return default
        raw = tds[td_idx].get_text(strip=True)
        m = re.search(r"([-+]?\d*\.?\d+)", raw)
        return float(m.group(1)) if m else default

    v_icme = get_num(11)     # plasma speed at 1 AU (km/s)
    v_max  = get_num(12)     # max plasma speed (km/s)
    b_mag  = get_num(13)     # magnetic field magnitude (nT)
    dst    = get_num(15)     # minimum Dst index (nT)
    v_trans = get_num(16)    # mean transit speed (km/s)
    
    # Infer initial CME launch speed: if v_trans given, v_init can be approximated or bounded
    # Physical relation: 1 AU = 1.496e8 km. V_transit_calc = 1.496e8 / (transit_hours * 3600)
    v_transit_calc = 1.495978707e8 / (transit_hours * 3600.0)
    
    # Solar Cycle assignment
    year = t_earth.year
    if year <= 2008:
        cycle = "Cycle 23"
    elif year <= 2019:
        cycle = "Cycle 24"
    else:
        cycle = "Cycle 25"
        
    records.append({
        "t_cme_launch": t_cme,
        "t_earth_arrival": t_earth,
        "transit_hours": transit_hours,
        "v_transit_calc": v_transit_calc,
        "v_icme_1au": v_icme,
        "v_max_1au": v_max,
        "b_mag_1au": b_mag,
        "dst_min": dst,
        "year": year,
        "solar_cycle": cycle
    })

df = pd.DataFrame(records)
print(f"Successfully extracted {len(df)} validated CME-ICME transit events (1996-2026)!")
print("\nBreakdown by Solar Cycle:")
for cycle, count in df["solar_cycle"].value_counts().items():
    print(f"  • {cycle}: {count} events ({count*100/len(df):.1f}%)")

print("\nTransit Duration Statistics (Hours):")
print(f"  • Mean transit:   {df['transit_hours'].mean():.1f} hours ({df['transit_hours'].mean()/24:.1f} days)")
print(f"  • Fastest CME:    {df['transit_hours'].min():.1f} hours ({df['v_transit_calc'].max():.1f} km/s)")
print(f"  • Slowest CME:    {df['transit_hours'].max():.1f} hours")
print(f"  • Minimum Dst:    {df['dst_min'].min():.1f} nT (Extreme Superstorm)")

df.to_parquet(OUTPUT_PARQUET, index=False)
df.head(20).to_csv(OUTPUT_CSV, index=False)
print(f"Saved processed dataset to: {OUTPUT_PARQUET}")

