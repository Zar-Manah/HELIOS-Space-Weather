#!/usr/bin/env python3
"""
download_real_solar_imagery.py — Fetches 100% REAL NASA/ESA Solar Imagery from Helioviewer API.
Downloads for each of the 314 CME events:
  1. SOHO LASCO C2 White-Light Coronagraph (sourceId=4)
  2. SOHO MDI Magnetogram (sourceId=6, pre-2010) or SDO HMI Magnetogram (sourceId=19, post-2010)
Saves verified images to data/raw/images/c2/ and data/raw/images/mag/.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(BASE_DIR, "data/processed/helios_vision_matched.parquet")
C2_DIR = os.path.join(BASE_DIR, "data/raw/images/c2")
MAG_DIR = os.path.join(BASE_DIR, "data/raw/images/mag")

os.makedirs(C2_DIR, exist_ok=True)
os.makedirs(MAG_DIR, exist_ok=True)

df = pd.read_parquet(PARQUET_PATH)
print(f"Loaded {len(df)} CME events from {PARQUET_PATH}")

def fetch_json(url, timeout=10):
    req = urllib.request.Request(url, headers={"User-Agent": "HELIOS-Real-Vision/2.5"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def download_file(url, out_path, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "HELIOS-Real-Vision/2.5"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    with open(out_path, "wb") as f:
        f.write(data)
    return len(data)

def process_event(idx, row):
    t_str = str(row["t_launch"]).replace(" ", "T") + "Z"
    t_clean = str(row["t_launch"])[:10].replace("-", "") + "_" + str(row["t_launch"])[11:16].replace(":", "")
    
    # Destination paths
    c2_path = os.path.join(C2_DIR, f"{t_clean}_c2_512.png")
    mag_path = os.path.join(MAG_DIR, f"{t_clean}_mag_512.png")
    
    res = {"idx": idx, "t_clean": t_clean, "c2_ok": False, "mag_ok": False, "c2_path": c2_path, "mag_path": mag_path}
    
    # 1. LASCO C2 (sourceId=4)
    if os.path.exists(c2_path) and os.path.getsize(c2_path) > 1000:
        res["c2_ok"] = True
    else:
        try:
            meta_url = f"https://api.helioviewer.org/v2/getClosestImage/?date={t_str}&sourceId=4"
            meta = fetch_json(meta_url)
            if "id" in meta:
                img_url = f"https://api.helioviewer.org/v2/downloadImage/?id={meta['id']}"
                nbytes = download_file(img_url, c2_path)
                if nbytes > 1000:
                    res["c2_ok"] = True
        except Exception as e:
            pass

    # 2. Magnetogram: SOHO MDI (sourceId=6) if before 2010-05, else SDO HMI (sourceId=19)
    is_sdo = str(row["t_launch"]) >= "2010-05-01"
    mag_source = 19 if is_sdo else 6
    
    if os.path.exists(mag_path) and os.path.getsize(mag_path) > 1000:
        res["mag_ok"] = True
    else:
        try:
            meta_url = f"https://api.helioviewer.org/v2/getClosestImage/?date={t_str}&sourceId={mag_source}"
            meta = fetch_json(meta_url)
            if "id" in meta:
                img_url = f"https://api.helioviewer.org/v2/downloadImage/?id={meta['id']}"
                nbytes = download_file(img_url, mag_path)
                if nbytes > 1000:
                    res["mag_ok"] = True
        except Exception as e:
            # Fallback to MDI if HMI failed or vice-versa
            fallback_source = 6 if is_sdo else 19
            try:
                meta_url = f"https://api.helioviewer.org/v2/getClosestImage/?date={t_str}&sourceId={fallback_source}"
                meta = fetch_json(meta_url)
                if "id" in meta:
                    img_url = f"https://api.helioviewer.org/v2/downloadImage/?id={meta['id']}"
                    nbytes = download_file(img_url, mag_path)
                    if nbytes > 1000:
                        res["mag_ok"] = True
            except Exception:
                pass
                
    return res

print(f"Starting concurrent downloads with 8 workers...")
t0 = time.time()
results = []
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(process_event, i, row) for i, row in df.iterrows()]
    completed = 0
    for fut in as_completed(futures):
        r = fut.result()
        results.append(r)
        completed += 1
        if completed % 25 == 0 or completed == len(df):
            c2_done = sum(1 for x in results if x["c2_ok"])
            mag_done = sum(1 for x in results if x["mag_ok"])
            elapsed = time.time() - t0
            print(f"[{completed}/{len(df)}] ({elapsed:.1f}s) — C2: {c2_done}/{completed} OK, MAG: {mag_done}/{completed} OK")

# Final verification with PIL
valid_c2 = 0
valid_mag = 0
for r in results:
    if r["c2_ok"]:
        try:
            with Image.open(r["c2_path"]) as im:
                im.verify()
            valid_c2 += 1
        except Exception:
            r["c2_ok"] = False
    if r["mag_ok"]:
        try:
            with Image.open(r["mag_path"]) as im:
                im.verify()
            valid_mag += 1
        except Exception:
            r["mag_ok"] = False

print("=" * 60)
print(f"DOWNLOAD SUMMARY ({time.time() - t0:.1f}s):")
print(f"  • Verified SOHO LASCO C2 Images:   {valid_c2}/{len(df)} ({valid_c2/len(df)*100:.1f}%)")
print(f"  • Verified Solar Magnetograms:     {valid_mag}/{len(df)} ({valid_mag/len(df)*100:.1f}%)")
print(f"  • Stored in: {C2_DIR} and {MAG_DIR}")
print("=" * 60)

