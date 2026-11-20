#!/usr/bin/env python3
"""HELIOS v3.4 - Stage 1.4: high-resolution OMNI + causal solar-wind context.

* Downloads the NASA SPDF 5-min OMNI files (SYM-H, AL/AE/AU, Bz/By/Bx GSM,
  B, V, Np, T, Pdyn, E_y, beta) for 1996-2026.
* Builds the hourly causal context used by every launch-time model:
  trailing means / mins / max / std / slopes over 1-48 h, pre-existing storm
  state, and the 27.275-day Carrington recurrence background.
* Never interpolates targets; gaps stay NaN.

Outputs
  data/processed/v3_omni_5min.parquet
  data/processed/v3_omni_context.parquet
  results/v3_omni_audit.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "omni5"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RAW.mkdir(parents=True, exist_ok=True)

BASE = "https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/omni_5min{y}.asc"
# 1-based token indices verified against hroformat.txt
COLS = {
    1: "year", 2: "doy", 3: "hour", 4: "minute",
    14: "b_mag", 15: "bx_gse", 16: "by_gse", 17: "bz_gse", 18: "by_gsm", 19: "bz_gsm",
    22: "v_flow", 23: "vx", 24: "vy", 25: "vz",
    26: "np_dens", 27: "temp", 28: "pdyn", 29: "e_field", 30: "beta", 31: "alfven_mach",
    38: "ae", 39: "al", 40: "au", 41: "sym_d", 42: "sym_h",
    43: "asy_d", 44: "asy_h", 45: "pc", 46: "mms",
}
MISSING = {99999.99, 9999.99, 99999.9, 999.99, 9999.9, 999999.0}


def download_year(y: int) -> Path:
    out = RAW / f"omni_5min{y}.asc"
    if out.exists() and out.stat().st_size > 1e6:
        return out
    r = requests.get(BASE.format(y=y), timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"OMNI 5min {y}: HTTP {r.status_code}")
    out.write_text(r.text)
    return out


def load_5min(years=range(1996, 2027)) -> pd.DataFrame:
    frames = []
    for y in years:
        try:
            p = download_year(y)
        except Exception as e:
            print(f"[OMNI5] {y} FAILED: {e}")
            continue
        rows = []
        for line in p.read_text().splitlines():
            parts = line.split()
            if len(parts) < 46:
                continue
            try:
                rec = {name: parts[idx - 1] for idx, name in COLS.items()}
            except IndexError:
                continue
            rows.append(rec)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.replace(list(MISSING), np.nan)
        df["datetime"] = (
            pd.to_datetime(df["year"].astype(int), format="%Y") +
            pd.to_timedelta(df["doy"].astype(int) - 1, unit="D") +
            pd.to_timedelta(df["hour"].astype(int), unit="h") +
            pd.to_timedelta(df["minute"].astype(int), unit="min")
        )
        frames.append(df.drop(columns=["year", "doy", "hour", "minute"]))
        print(f"[OMNI5] {y}: {len(df)} rows")
    out = pd.concat(frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    return out


def build_hourly_context() -> pd.DataFrame:
    h = pd.read_parquet(
        PROC / "helios_omni_features_1998_2026.parquet",
        columns=["datetime", "dst", "bz_gsm", "by_gsm", "b_mag", "v_flow", "np_dens",
                 "pdyn", "e_field", "temp", "kp", "f107"],
    )
    h["datetime"] = pd.to_datetime(h["datetime"])
    h = h.sort_values("datetime").set_index("datetime")
    # reindex to a strict hourly grid so rolling windows mean real hours
    full = pd.date_range(h.index.min(), h.index.max(), freq="h")
    h = h.reindex(full)
    h.index.name = "datetime"
    # physically-defined coupling functions (no proxies)
    b_t = np.sqrt(h["by_gsm"] ** 2 + h["bz_gsm"] ** 2)
    clock = np.arctan2(h["by_gsm"], h["bz_gsm"])
    h["newell"] = 1.6e-3 * (h["v_flow"].clip(lower=0) ** (4 / 3)) * (b_t ** (2 / 3)) * \
        (np.sin(np.abs(clock) / 2.0).clip(lower=0) ** (8 / 3))
    h["ey_rect"] = h["e_field"].clip(lower=0)
    feats = {}
    for win in (1, 3, 6, 12, 24, 48):
        r = h.rolling(win, min_periods=max(1, win // 3))
        feats[f"v_mean_{win}h"] = r["v_flow"].mean()
        feats[f"v_min_{win}h"] = r["v_flow"].min()
        feats[f"v_max_{win}h"] = r["v_flow"].max()
        feats[f"bz_min_{win}h"] = r["bz_gsm"].min()
        feats[f"bz_mean_{win}h"] = r["bz_gsm"].mean()
        feats[f"b_mean_{win}h"] = r["b_mag"].mean()
        feats[f"np_mean_{win}h"] = r["np_dens"].mean()
        feats[f"pdyn_mean_{win}h"] = r["pdyn"].mean()
        feats[f"dst_mean_{win}h"] = r["dst"].mean()
        feats[f"dst_min_{win}h"] = r["dst"].min()
        feats[f"dst_std_{win}h"] = r["dst"].std()
        feats[f"ey_max_{win}h"] = r["ey_rect"].max()
        feats[f"newell_mean_{win}h"] = r["newell"].mean()
        feats[f"newell_sum_{win}h"] = r["newell"].sum()
    ctx = pd.DataFrame(feats, index=h.index)
    # instantaneous state
    for c in ("dst", "bz_gsm", "by_gsm", "b_mag", "v_flow", "np_dens", "pdyn", "e_field", "kp", "f107"):
        ctx[f"t0_{c}"] = h[c]
    # causal slopes (linear fit over the window)
    for win in (3, 6, 12, 24):
        x = np.arange(win, dtype=float)
        x = x - x.mean()
        denom = (x ** 2).sum()

        def slope(s):
            v = s.values
            if len(v) != win or not np.isfinite(v).all():
                return np.nan
            return float((x * (v - v.mean())).sum() / denom)

        ctx[f"v_slope_{win}h"] = h["v_flow"].rolling(win, min_periods=win).apply(slope, raw=False)
        ctx[f"dst_slope_{win}h"] = h["dst"].rolling(win, min_periods=win).apply(slope, raw=False)
    # 27.275-day Carrington recurrence background: mean of the +-12 h window
    # centred on t - 27.275 d.  Vectorised with a centred rolling mean plus a
    # nearest-index shift (no per-row Python loop).
    lag_h = int(round(27.275 * 24))
    for c in ("v_flow", "np_dens", "b_mag", "bz_gsm"):
        smooth = h[c].rolling(25, center=True, min_periods=6).mean()
        shifted = smooth.shift(lag_h)
        ctx[f"carr_{c}"] = shifted.reindex(ctx.index)
    ctx = ctx.reset_index()
    return ctx


def main() -> None:
    omni5 = load_5min()
    omni5.to_parquet(PROC / "v3_omni_5min.parquet", index=False)
    print(f"[OMNI5] total rows={len(omni5)}  {omni5.datetime.min()} .. {omni5.datetime.max()}")

    ctx = build_hourly_context()
    ctx.to_parquet(PROC / "v3_omni_context.parquet", index=False)
    print(f"[CTX ] rows={len(ctx)} cols={ctx.shape[1]}")

    audit = {
        "omni5_rows": int(len(omni5)),
        "omni5_start": str(omni5.datetime.min()),
        "omni5_end": str(omni5.datetime.max()),
        "omni5_symh_present": int(omni5["sym_h"].notna().sum()),
        "omni5_bz_present": int(omni5["bz_gsm"].notna().sum()),
        "omni5_ae_present": int(omni5["ae"].notna().sum()),
        "context_rows": int(len(ctx)),
        "context_cols": int(ctx.shape[1]),
        "context_nan_frac": {c: float(ctx[c].isna().mean()) for c in
                             ("carr_v_flow", "v_mean_24h", "dst_min_48h", "newell_mean_6h")},
    }
    (RESULTS / "v3_omni_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
