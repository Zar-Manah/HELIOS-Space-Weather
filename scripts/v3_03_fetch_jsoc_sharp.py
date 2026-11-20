#!/usr/bin/env python3
"""HELIOS v3.4 - Stage 1.3: real Stanford JSOC vector-magnetogram telemetry.

Replaces every synthetic magnetic proxy (shear_angle_proxy, free_energy_proxy,
helicity_proxy) with real SDO/HMI SHARP invariants and their 24 h evolution:
USFLUX, TOTUSJH, TOTUSJZ, ABSNJZH, TOTPOT, MEANSHR, SHRGT45, R_VALUE, AREA_ACR.

Strict causality: only records with T_REC <= t_launch are used.
Pre-2010: mdi.smarp_cea_96m where available (R_VALUE, AREA_ACR only);
everything else stays native NaN.

Outputs
  data/raw/jsoc_v3/*.json                  (cache, one file per AR-day)
  data/processed/v3_sharp_features.parquet
  results/v3_sharp_audit.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "jsoc_v3"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RAW.mkdir(parents=True, exist_ok=True)

JSOC = "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info"
KEYS_HMI = ("T_REC,HARPNUM,NOAA_AR,USFLUX,TOTUSJH,TOTUSJZ,ABSNJZH,TOTPOT,MEANSHR,"
            "SHRGT45,R_VALUE,AREA_ACR,SAVNCPP")
KEYS_MDI = "T_REC,NOAA_AR,R_VALUE,AREA_ACR"
TAI_OFFSET_S = 37.0  # TAI - UTC since 2017 (constant enough for 720 s cadence)


def _query(ds: str, keys: str) -> list[dict]:
    for _ in range(3):
        try:
            r = requests.get(JSOC, params={"op": "rs_list", "ds": ds, "key": keys}, timeout=90)
            if r.status_code != 200:
                continue
            d = r.json()
            if d.get("status") != 0:
                continue
            kws = d.get("keywords", [])
            if not kws:
                return []
            names = [k["name"] for k in kws]
            n = len(kws[0]["values"])
            return [{names[i]: kws[i]["values"][j] for i in range(len(names))} for j in range(n)]
        except Exception:
            pass
    return []


def fetch_ar_day(ar: int, day: pd.Timestamp) -> tuple[str, int]:
    out = RAW / f"ar{ar}_{day.strftime('%Y%m%d')}.json"
    if out.exists():
        try:
            cached = json.loads(out.read_text())
            # An empty cache for a real AR is a transient JSOC failure, not a
            # physical answer: never trust it, refetch it.
            if isinstance(cached, list) and len(cached) > 0:
                return str(out), len(cached)
        except Exception:
            pass
    # A 7-day window ending at launch day: enough to recover ARs that were
    # numbered a few days before the eruption.  Causality is enforced later
    # by `t_rec <= t_launch`; querying a wider window does not leak the future.
    start = day - pd.Timedelta(days=3)
    if day >= pd.Timestamp("2010-05-01"):
        ds = f"hmi.sharp_cea_720s[][{start.strftime('%Y.%m.%d')}_00:00_TAI/4d@720s]"
        recs = _query(ds, KEYS_HMI)
    else:
        ds = f"mdi.smarp_cea_96m[][{start.strftime('%Y.%m.%d')}_00:00_TAI/4d@96m]"
        recs = _query(ds, KEYS_MDI)
    recs = [r for r in recs if _same_ar(r.get("NOAA_AR"), ar)]
    out.write_text(json.dumps(recs))
    return str(out), len(recs)


def _same_ar(value, ar: int) -> bool:
    """JSOC writes NOAA_AR as a string; never compare '11081.0' to '11081'."""
    try:
        return int(float(str(value).strip())) == int(ar)
    except (TypeError, ValueError):
        return False


def _same_ar_num(value) -> float:
    try:
        v = int(float(str(value).strip()))
        return float(v) if v > 0 else np.nan
    except (TypeError, ValueError):
        return np.nan


def parse_records(paths: list[str]) -> pd.DataFrame:
    rows = []
    for p in paths:
        try:
            recs = json.loads(Path(p).read_text())
        except Exception:
            # a half-written cache from an interrupted fetch is not data
            continue
        for r in recs:
            t = r.get("T_REC")
            if not t:
                continue
            ts = pd.to_datetime(str(t).replace("_TAI", "").replace("_", " "), errors="coerce")
            if pd.isna(ts):
                continue
            ts = ts - pd.Timedelta(seconds=TAI_OFFSET_S)
            rec = {"t_rec": ts, "noaa_ar": _same_ar_num(r.get("NOAA_AR"))}
            for k in ("USFLUX", "TOTUSJH", "TOTUSJZ", "ABSNJZH", "TOTPOT", "MEANSHR",
                      "SHRGT45", "R_VALUE", "AREA_ACR", "SAVNCPP", "HARPNUM"):
                try:
                    v = float(r.get(k))
                    rec[k.lower()] = v if np.isfinite(v) else np.nan
                except (TypeError, ValueError):
                    rec[k.lower()] = np.nan
            rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=["t_rec", "noaa_ar"])
    df = pd.DataFrame(rows)
    # The 4-day windows of consecutive AR-days overlap, so the same SHARP
    # record arrives from up to four files.  A record is identified by its
    # timestamp, AR and HARP number; de-duplicating keeps the audit honest
    # without changing which record is "last before launch".
    key = [c for c in ("t_rec", "noaa_ar", "harpnum") if c in df]
    return df.drop_duplicates(subset=key).sort_values("t_rec").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--build-only", action="store_true",
                    help="rebuild the parquet from the cache without fetching")
    args = ap.parse_args()

    src = pd.read_parquet(PROC / "v3_solar_sources.parquet")
    src["t_launch"] = pd.to_datetime(src["t_launch"])
    sel = src[(src["ar_num"] > 0) & (src["t_launch"] >= pd.Timestamp("1996-05-01"))].copy()
    sel["ar_num"] = sel["ar_num"].astype(int)
    sel["day"] = sel["t_launch"].dt.floor("D")
    if args.limit:
        sel = sel.head(args.limit)
    pairs = sel[["ar_num", "day"]].drop_duplicates()
    print(f"[JSOC] CMEs with AR={len(sel)}  unique AR-days={len(pairs)}", flush=True)

    def _cached_ok(path: Path) -> bool:
        if not path.exists():
            return False
        try:
            d = json.loads(path.read_text())
            return isinstance(d, list) and len(d) > 0
        except Exception:
            return False

    todo = [(int(r.ar_num), r.day) for r in pairs.itertuples()
            if not _cached_ok(RAW / f"ar{int(r.ar_num)}_{r.day.strftime('%Y%m%d')}.json")]
    print(f"[JSOC] cached={len(pairs) - len(todo)} to_fetch={len(todo)}", flush=True)
    if args.build_only:
        todo = []
    if todo:
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_ar_day, a, d): (a, d) for a, d in todo}
            for f in as_completed(futs):
                done += 1
                if done % 100 == 0:
                    print(f"  fetched {done}/{len(todo)}", flush=True)

    paths = [str(RAW / f"ar{a}_{d.strftime('%Y%m%d')}.json") for a, d in
             pairs[["ar_num", "day"]].itertuples(index=False)]
    recs = parse_records(paths)
    print(f"[JSOC] parsed records={len(recs)}  with USFLUX={recs.get('usflux', pd.Series(dtype=float)).notna().sum()}", flush=True)

    feats = []
    for _, c in sel.iterrows():
        sub = recs[(recs["noaa_ar"] == int(c["ar_num"])) & (recs["t_rec"] <= c["t_launch"])]
        if sub.empty:
            feats.append(dict(cme_id=c["cme_id"]))
            continue
        last = sub.iloc[-1]
        base = dict(cme_id=c["cme_id"], sharp_t_rec=last["t_rec"],
                    sharp_age_h=float((c["t_launch"] - last["t_rec"]).total_seconds() / 3600.0))
        for k in ("usflux", "totusjh", "totusjz", "absnjzh", "totpot", "meanshr",
                  "shrgt45", "r_value", "area_acr", "savncpp", "harpnum"):
            base[f"sharp_{k}"] = last.get(k, np.nan)
        prev = sub[sub["t_rec"] <= c["t_launch"] - pd.Timedelta(hours=24)]
        if not prev.empty:
            p = prev.iloc[-1]
            dt_h = max((last["t_rec"] - p["t_rec"]).total_seconds() / 3600.0, 1.0)
            for k in ("usflux", "totpot", "totusjh", "r_value", "shrgt45", "meanshr"):
                a, b = last.get(k, np.nan), p.get(k, np.nan)
                base[f"d_{k}_24h"] = (a - b) / dt_h if np.isfinite(a) and np.isfinite(b) else np.nan
        feats.append(base)
    fd = pd.DataFrame(feats)
    out = PROC / "v3_sharp_features.parquet"
    fd.to_parquet(out, index=False)

    audit = {
        "n_cme_with_ar": int(len(sel)),
        "n_sharp_records": int(len(recs)),
        "n_cme_with_sharp": int(fd["sharp_usflux"].notna().sum()) if "sharp_usflux" in fd else 0,
        "n_cme_with_totpot": int(fd["sharp_totpot"].notna().sum()) if "sharp_totpot" in fd else 0,
        "n_cme_with_deltas": int(fd["d_totpot_24h"].notna().sum()) if "d_totpot_24h" in fd else 0,
        "year_min": str(sel["t_launch"].min()),
        "year_max": str(sel["t_launch"].max()),
        "sharp_median_age_h": float(fd["sharp_age_h"].median()) if "sharp_age_h" in fd else None,
    }
    (RESULTS / "v3_sharp_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    print(f"[OK] {out} shape={fd.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
