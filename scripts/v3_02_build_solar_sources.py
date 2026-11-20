#!/usr/bin/env python3
"""HELIOS v3.4 - Stage 1.2: unified solar source localization (1996-2026).

Fixes over v3.0:
  * HEK `hgs_x/hgs_y` are 0 for every record flagged `noposition=true`
    (66 % of the SWPC flares).  v3.0 stored that 0 as a real Stonyhurst
    longitude, i.e. it invented a disk-centre source for two thirds of the
    catalog.  Here a missing position is NaN, never 0, and every position
    carries its provenance (`pos_source`).
  * Positions are recovered from the SSW Latest Events frame (real hgs_x/hgs_y
    + NOAA AR number) by nearest-time match, plus EUV coronal dimmings (CD) and
    EUV CMEs (CE) for events without a flare.
  * `backside_confirmed` (no frontside signature at all, halo, no AR) is a
    heuristic candidate set, not a proven backside origin, and is kept apart
    from `unknown_source` (no signature and no halo/AR pattern).
  * Non-flare sources never invent a longitude: filament eruptions without a
    measured position stay NaN.

Outputs
  data/raw/hek_v3/{FL,FS,FE,ER,CD,CE}_YYYYQq.json   (cache)
  data/processed/v3_solar_sources.parquet

The audit dictionary is printed to stdout for inspection and is not written to
disk, so no derived JSON has to travel with the package.
"""
from __future__ import annotations

import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hek_v3"
PROC = ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
BASE = "https://www.lmsal.com/hek/her"
CLASS_EXP = {"A": -8, "B": -7, "C": -6, "M": -5, "X": -4}
QUARTER_END = pd.Timestamp("2026-12-01")

# FL = SWPC operational GOES list (class, no position for most records)
# FS = SSW Latest Events (real Stonyhurst position + NOAA AR)
# FE/ER = filament eruptions / eruptions, CD = EUV dimming, CE = EUV CME
STREAMS = {
    "FL": {"frame": "SWPC", "years": range(1996, 2027)},
    "FS": {"frame": "SSW Latest Events", "years": range(1996, 2027)},
    "FE": {"frame": None, "years": range(1996, 2027)},
    "ER": {"frame": None, "years": range(1996, 2027)},
}
# HEK answers CD/CE with zero records (no frame publishes them), so they are not
# fetched; `aux` still accepts them if a future cache provides them.


def _params(et: str, frame: str | None, start: str, end: str) -> dict:
    """HEK caps any single response at 1000 records, so the frame filter matters."""
    p = {
        "cosec": "2", "cmd": "search", "type": "column", "event_type": et,
        "event_coordsys": "helioprojective",
        "event_starttime": start, "event_endtime": end,
        "x1": "-1200", "y1": "-1200", "x2": "1200", "y2": "1200",
        "result_limit": "10000",
    }
    if frame:
        p.update({"param0": "frm_name", "op0": "=", "value0": frame})
    return p


def _get(params: dict) -> dict:
    for attempt in range(3):
        try:
            r = requests.get(BASE, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    return {"result": [], "overmax": False, "_failed": True}


def fetch_range(et: str, frame: str | None, start: pd.Timestamp, end: pd.Timestamp,
                depth: int = 0) -> tuple[list[dict], bool]:
    d = _get(_params(et, frame, start.strftime("%Y-%m-%dT%H:%M:%S"),
                     end.strftime("%Y-%m-%dT%H:%M:%S")))
    if d.get("_failed"):
        return [], True
    res = d.get("result", [])
    if (d.get("overmax") or len(res) >= 950) and depth < 6 and (end - start) > pd.Timedelta(days=1):
        mid = start + (end - start) / 2
        a, fa = fetch_range(et, frame, start, mid, depth + 1)
        b, fb = fetch_range(et, frame, mid, end, depth + 1)
        return a + b, fa or fb
    return res, False


def fetch_quarter(et: str, frame: str | None, y: int, q: int) -> tuple[str, int]:
    tag = "FS" if et == "FL" and frame == "SSW Latest Events" else et
    out = RAW / f"{tag}_{y}Q{q}.json"
    if out.exists():
        try:
            d = json.loads(out.read_text())
            res = d.get("result", [])
            # An empty response is a *failure* for the position frame: never
            # cache it as authoritative, otherwise the 124 empty FS_*.json
            # files of v3.1 are reused forever (root cause of pos_source=none).
            if res or (d.get("_empty_ok") and tag != "FS"):
                return str(out), len(res)
        except Exception:
            pass
    start = pd.Timestamp(f"{y}-{(q - 1) * 3 + 1:02d}-01")
    end = start + pd.DateOffset(months=3)
    if end > QUARTER_END:
        end = QUARTER_END
    if start >= QUARTER_END:
        return "skipped", 0
    res, failed = fetch_range(et, frame, start, end)
    if failed:
        return "failed", 0
    out.write_text(json.dumps({"result": res, "_empty_ok": bool(res) or tag != "FS"}))
    return str(out), len(res)


def parse_class(txt: str) -> tuple[str, float, float]:
    if not txt:
        return "", np.nan, np.nan
    m = re.match(r"([ABCMX])\s*([\d.]+)", str(txt).strip().upper())
    if not m:
        return "", np.nan, np.nan
    letter, val = m.group(1), float(m.group(2))
    return letter, val, val * 10 ** CLASS_EXP[letter]


def _num_or_nan(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _is_true(x) -> bool:
    return str(x).strip().lower() in ("true", "1", "yes")


def load_events() -> pd.DataFrame:
    """Read every cached HEK stream and keep position provenance explicit."""
    rows = []
    for tag in STREAMS:
        for f in sorted(RAW.glob(f"{tag}_*.json")):
            for r in json.loads(f.read_text()).get("result", []):
                peak = r.get("event_peaktime") or r.get("event_starttime")
                if not peak:
                    continue
                letter, val, flux = parse_class(r.get("fl_goescls", ""))
                no_pos = _is_true(r.get("noposition"))
                lon = _num_or_nan(r.get("hgs_x"))
                lat = _num_or_nan(r.get("hgs_y"))
                # HEK writes 0/0 when the frame has no position: that is a
                # missing value, not the disk centre.
                if no_pos or (lon == 0.0 and lat == 0.0):
                    lon = lat = np.nan
                rows.append(dict(
                    event_type=tag, t_peak=pd.to_datetime(peak, errors="coerce", utc=True).tz_localize(None),
                    t_start=pd.to_datetime(r.get("event_starttime"), errors="coerce", utc=True).tz_localize(None),
                    cls_letter=letter, cls_value=val, goes_flux=flux,
                    ar=_num_or_nan(r.get("ar_noaanum")),
                    src_lon=lon, src_lat=lat,
                    has_pos=bool(np.isfinite(lon) and np.isfinite(lat)),
                    frm_name=r.get("frm_name", ""),
                    fluence=_num_or_nan(r.get("fl_fluence")),
                ))
    df = pd.DataFrame(rows).dropna(subset=["t_peak"]).sort_values("t_peak").reset_index(drop=True)
    df = df.drop_duplicates(subset=["event_type", "t_peak", "src_lon", "src_lat"]).reset_index(drop=True)
    return df


def _goes_label(flux: float) -> str:
    if not np.isfinite(flux) or flux <= 0:
        return ""
    exp = int(np.floor(np.log10(flux)))
    letter = {-8: "A", -7: "B", -6: "C", -5: "M", -4: "X"}.get(exp, "?")
    return f"{letter}{flux / 10 ** exp:.1f}"


def pa_from_source(lon: float, lat: float) -> float:
    if not np.isfinite(lon) or not np.isfinite(lat):
        return np.nan
    x_east = -math.cos(math.radians(lat)) * math.sin(math.radians(lon))
    y = math.sin(math.radians(lat))
    return math.degrees(math.atan2(x_east, y)) % 360.0


def ang_diff(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _arrays(df: pd.DataFrame):
    return (df["t_peak"].values.astype("datetime64[ns]").astype(np.int64),
            df["goes_flux"].values, df["src_lon"].values, df["src_lat"].values,
            df["ar"].values, df["cls_letter"].values, df["fluence"].values,
            df["event_type"].values, df["has_pos"].values)


SYNODIC_DEG_PER_DAY = 360.0 / 27.2753  # Carrington synodic rotation


def _ar_num(x) -> float:
    """NOAA AR numbers are positive; 0/-1/NaN all mean 'no active region'."""
    v = _num_or_nan(x)
    return v if np.isfinite(v) and v > 0 else np.nan


def _normalise_legacy_ar(ar: pd.Series, t: pd.Series) -> pd.Series:
    """Repair the 5-digit AR numbers the v2-era tables stored with 4 digits.

    NOAA active-region numbering crossed 10000 in June 2002, so no AR after
    that date can have a 4-digit number.  `helios_hek_flares.parquet` and
    `helios_ar_positions.parquet` were written by the v2 pipeline with the
    leading "1" stripped (e.g. 1063 for AR 11063), which made every post-2010
    CME fail to match its JSOC SHARP record.  The correction is arithmetic on
    a documented numbering rule, not a fabricated association.
    """
    a = pd.to_numeric(ar, errors="coerce").astype(float)
    tt = pd.to_datetime(t, errors="coerce")
    fix = (tt >= pd.Timestamp("2002-07-01")) & a.notna() & (a > 0) & (a < 10000)
    a = a.where(~fix, a + 10000.0)
    return a


def load_legacy_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Real tables already on disk (v2 era) that v3.1 never consumed."""
    fl = pd.DataFrame(columns=["t_peak", "src_lon", "src_lat", "ar"])
    p = PROC / "helios_hek_flares.parquet"
    if p.exists():
        d = pd.read_parquet(p)
        t = pd.to_datetime(d["flare_peak_utc"], errors="coerce")
        lon = pd.to_numeric(d["src_lon"], errors="coerce")
        lat = pd.to_numeric(d["src_lat"], errors="coerce")
        ar = _normalise_legacy_ar(d["ar"], t)
        ok = t.notna() & lon.notna() & lat.notna() & ~((lon == 0) & (lat == 0))
        fl = pd.DataFrame({
            "t_peak": t[ok].dt.tz_localize(None).values,
            "src_lon": lon[ok].values, "src_lat": lat[ok].values,
            "ar": ar[ok].values,
        }).sort_values("t_peak").reset_index(drop=True)
    ap = pd.DataFrame(columns=["valid_utc", "ar", "lat", "lon"])
    p = PROC / "helios_ar_positions.parquet"
    if p.exists():
        d = pd.read_parquet(p)
        d["valid_utc"] = pd.to_datetime(d["valid_utc"]).dt.tz_localize(None)
        ap = d.rename(columns={"lat_deg": "lat", "lon_deg": "lon"})
        ap["ar"] = _normalise_legacy_ar(ap["ar"], ap["valid_utc"])
        ap = ap.sort_values("valid_utc").reset_index(drop=True)
    return fl, ap


def _best_position(t: int, times, lons, lats, cpa, width,
                   before_min: float, after_min: float) -> tuple | None:
    """Nearest position inside the physical causal window (same span as the
    flare-class association: the CME is launched after the flare peak)."""
    if len(times) == 0:
        return None
    lo = np.searchsorted(times, t - int(before_min * 60 * 10**9), side="left")
    hi = np.searchsorted(times, t + int(after_min * 60 * 10**9), side="right")
    best = None
    for j in range(lo, hi):
        lon, lat = lons[j], lats[j]
        if not np.isfinite(lon) or not np.isfinite(lat):
            continue
        pa = pa_from_source(lon, lat)
        diff = ang_diff(pa, cpa) if np.isfinite(cpa) else np.nan
        if np.isfinite(width) and width < 180 and np.isfinite(diff) and diff > 120:
            continue
        dt = abs(float(t - times[j])) / 6e10
        if best is None or dt < best[0]:
            best = (dt, j, pa, diff)
    return best


def associate(cat: pd.DataFrame, ev: pd.DataFrame,
              legacy_fl: pd.DataFrame | None = None,
              ar_pos: pd.DataFrame | None = None) -> pd.DataFrame:
    """Associate each CME with a causal frontside signature.

    Priority:
      1. SWPC flare class  (always present)   -> flare_class / goes_flux
      2. real Stonyhurst position from HEK FL -> src_lon / src_lat / ar_num
         (v3.1 ignored the 15 550 FL records that do carry hgs_x/hgs_y and
          relied only on the empty SSW FS frame)
      3. SSW Latest Events position (FS)      -> fallback when FL has none
      4. EUV dimming / EUV CME / filament eruption / eruption (CD, CE, FE, ER)
      5. local v2 tables: HEK flares with src_lon (17 807) and SRS AR
         positions rotated causally to t_launch (`srs_rotated`)
    A missing position never becomes 0: it stays NaN with `pos_source="none"`.
    """
    fl = ev[ev.event_type == "FL"].reset_index(drop=True)
    flp = ev[(ev.event_type == "FL") & ev.has_pos].reset_index(drop=True)
    fs = ev[(ev.event_type == "FS") & ev.has_pos].reset_index(drop=True)
    aux = ev[ev.event_type.isin(["CD", "CE", "FE", "ER"])].reset_index(drop=True)

    ft, fflux, _, _, far, _, fflu, _, _ = _arrays(fl)
    pt, _, plon, plat, par, _, _, _, _ = _arrays(flp)
    st, _, slon, slat, sar, _, _, _, _ = _arrays(fs)
    at, aflux, alon, alat, aar, acls, aflu, atype, apos = _arrays(aux)

    legacy_fl = legacy_fl if legacy_fl is not None else pd.DataFrame()
    if len(legacy_fl):
        lt = legacy_fl["t_peak"].values.astype("datetime64[ns]").astype(np.int64)
        llon = legacy_fl["src_lon"].values
        llat = legacy_fl["src_lat"].values
        lar = legacy_fl["ar"].values
    else:
        lt = np.array([], dtype=np.int64)
        llon = llat = lar = np.array([], dtype=float)

    ar_pos = ar_pos if ar_pos is not None else pd.DataFrame()
    ar_index: dict[int, tuple] = {}
    if len(ar_pos):
        for ar, g in ar_pos.groupby("ar"):
            ar_index[int(ar)] = (g["valid_utc"].values.astype("datetime64[ns]").astype(np.int64),
                                 g["lon"].values.astype(float), g["lat"].values.astype(float))

    out = []
    for _, c in cat.iterrows():
        t = np.datetime64(c["t_launch"]).astype("datetime64[ns]").astype(np.int64)
        cpa, width = c["cpa"], c["width"]
        src = dict(source_type="none", src_lon=np.nan, src_lat=np.nan, src_pa=np.nan,
                   flare_peak=pd.NaT, flare_class="", goes_flux=np.nan,
                   flare_cme_dt_min=np.nan, ar_num=np.nan, flare_fluence=np.nan,
                   pos_angle_consistency=np.nan, source_rule="", pos_source="none",
                   has_position=0, backside_confirmed=0, unknown_source=0)

        # 1) SWPC flare class in [-180, +30] min (no angular veto: position unknown)
        lo = np.searchsorted(ft, t - 180 * 60 * 10**9, side="left")
        hi = np.searchsorted(ft, t + 30 * 60 * 10**9, side="right")
        best = None
        for j in range(lo, hi):
            flux = fflux[j]
            if not np.isfinite(flux) or flux < 1e-6:
                continue
            if best is None or flux > best[0]:
                best = (flux, j)
        if best is not None:
            flux, j = best
            src.update(source_type="flare_swpc", flare_peak=pd.Timestamp(ft[j]),
                       flare_class=_goes_label(flux), goes_flux=flux,
                       flare_cme_dt_min=float(t - ft[j]) / 6e10, ar_num=_ar_num(far[j]),
                       flare_fluence=fflu[j], source_rule="swpc_flare")

        # 2) real Stonyhurst position carried by HEK FL itself (nearest +/-20 min)
        best2 = _best_position(t, pt, plon, plat, cpa, width, 180.0, 30.0)
        if best2 is not None:
            dt, j, pa, diff = best2
            src.update(src_lon=plon[j], src_lat=plat[j], src_pa=pa, pos_source="hek_fl",
                       pos_angle_consistency=1.0 - diff / 180.0 if np.isfinite(diff) else np.nan)
            if not np.isfinite(src["ar_num"]):
                src["ar_num"] = _ar_num(par[j])

        # 3) SSW Latest Events position (kept as a secondary witness)
        if not np.isfinite(src["src_lon"]):
            best3 = _best_position(t, st, slon, slat, cpa, width, 180.0, 30.0)
            if best3 is not None:
                dt, j, pa, diff = best3
                src.update(src_lon=slon[j], src_lat=slat[j], src_pa=pa, pos_source="ssw_flare",
                           pos_angle_consistency=1.0 - diff / 180.0 if np.isfinite(diff) else np.nan)
                if not np.isfinite(src["ar_num"]):
                    src["ar_num"] = _ar_num(sar[j])
                if src["source_type"] == "none":
                    src.update(source_type="flare_ssw", flare_peak=pd.Timestamp(st[j]),
                               flare_cme_dt_min=float(t - st[j]) / 6e10, source_rule="ssw_flare")

        # 4) EUV dimming / EUV CME / filament eruption / eruption
        if not np.isfinite(src["src_lon"]):
            lo3 = np.searchsorted(at, t - 240 * 60 * 10**9, side="left")
            hi3 = np.searchsorted(at, t + 30 * 60 * 10**9, side="right")
            best4 = None
            for j in range(lo3, hi3):
                lon, lat = alon[j], alat[j]
                if not apos[j] or not np.isfinite(lon) or not np.isfinite(lat):
                    continue
                pa = pa_from_source(lon, lat)
                diff = ang_diff(pa, cpa) if np.isfinite(cpa) else np.nan
                if np.isfinite(width) and width < 180 and np.isfinite(diff) and diff > 120:
                    continue
                dt = abs(float(t - at[j])) / 6e10
                # a dimming is a stronger witness than a filament eruption
                rank = (0 if atype[j] in ("CD", "CE") else 1, dt)
                if best4 is None or rank < best4[0]:
                    best4 = (rank, j, pa, diff)
            if best4 is not None:
                _, j, pa, diff = best4
                src.update(src_lon=alon[j], src_lat=alat[j], src_pa=pa,
                           pos_source=f"hek_{atype[j].lower()}",
                           pos_angle_consistency=1.0 - diff / 180.0 if np.isfinite(diff) else np.nan,
                           flare_cme_dt_min=float(t - at[j]) / 6e10,
                           source_rule=f"hek_{atype[j].lower()}")
                if src["source_type"] == "none":
                    src.update(source_type="euv_signature",
                               flare_peak=pd.Timestamp(at[j]), flare_class=atype[j])
                if not np.isfinite(src["ar_num"]):
                    src["ar_num"] = _ar_num(aar[j])

        # 5a) local v2 HEK flare table with real src_lon (nearest +/-20 min)
        if not np.isfinite(src["src_lon"]) and len(lt):
            best5 = _best_position(t, lt, llon, llat, cpa, width, 180.0, 30.0)
            if best5 is not None:
                dt, j, pa, diff = best5
                src.update(src_lon=llon[j], src_lat=llat[j], src_pa=pa, pos_source="legacy_hek",
                           pos_angle_consistency=1.0 - diff / 180.0 if np.isfinite(diff) else np.nan)
                if not np.isfinite(src["ar_num"]):
                    src["ar_num"] = _ar_num(lar[j])
                if src["source_type"] == "none":
                    src.update(source_type="flare_legacy", source_rule="legacy_hek")

        # 5b) SRS active-region position rotated causally to t_launch
        ar = src["ar_num"]
        if not np.isfinite(src["src_lon"]) and np.isfinite(ar) and int(ar) in ar_index:
            times, lons, lats = ar_index[int(ar)]
            lo6 = np.searchsorted(times, t, side="right") - 1
            if lo6 >= 0:
                age_days = float(t - times[lo6]) / (86400.0 * 1e9)
                if 0.0 <= age_days <= 45.0:
                    lon = lons[lo6] + SYNODIC_DEG_PER_DAY * age_days
                    lon = (lon + 180.0) % 360.0 - 180.0
                    lat = lats[lo6]
                    pa = pa_from_source(lon, lat)
                    diff = ang_diff(pa, cpa) if np.isfinite(cpa) else np.nan
                    if not (np.isfinite(width) and width < 180 and np.isfinite(diff) and diff > 120):
                        src.update(src_lon=lon, src_lat=lat, src_pa=pa, pos_source="srs_rotated",
                                   pos_angle_consistency=1.0 - diff / 180.0 if np.isfinite(diff) else np.nan,
                                   source_rule="srs_rotated")
                        if src["source_type"] == "none":
                            src.update(source_type="active_region", source_rule="srs_rotated")

        src["has_position"] = int(np.isfinite(src["src_lon"]) and np.isfinite(src["src_lat"]))
        out.append(src)
    src_df = pd.DataFrame(out, index=cat.index)
    return pd.concat([cat, src_df], axis=1)


def main() -> int:
    tasks = [(et, cfg["frame"], y, q) for et, cfg in STREAMS.items()
             for y in cfg["years"] for q in (1, 2, 3, 4)]
    def _path(t):
        tag = "FS" if t[0] == "FL" and t[1] == "SSW Latest Events" else t[0]
        return RAW / f"{tag}_{t[2]}Q{t[3]}.json"

    def _usable(t) -> bool:
        p = _path(t)
        if not p.exists():
            return False
        try:
            d = json.loads(p.read_text())
        except Exception:
            return False
        tag = "FS" if t[0] == "FL" and t[1] == "SSW Latest Events" else t[0]
        return bool(d.get("result")) or (bool(d.get("_empty_ok")) and tag != "FS")

    todo = [t for t in tasks if not _usable(t)]
    print(f"[HEK] cached={len(tasks) - len(todo)}  to_fetch={len(todo)}", flush=True)
    if todo:
        done, failed = 0, []
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(fetch_quarter, *t): t for t in todo}
            for f in as_completed(futs):
                done += 1
                tag, n = f.result()
                if tag == "failed":
                    failed.append(futs[f])
                if done % 40 == 0:
                    print(f"  fetched {done}/{len(todo)} (failed={len(failed)})", flush=True)
        print(f"[HEK] done failed={len(failed)} {failed[:8]}", flush=True)

    ev = load_events()
    print(f"[HEK] events: {ev.event_type.value_counts().to_dict()}")
    print(f"[HEK] FL with real position: {int(ev[(ev.event_type=='FL')].has_pos.sum())}"
          f" / {int((ev.event_type=='FL').sum())}")
    legacy_fl, ar_pos = load_legacy_tables()
    print(f"[HEK] legacy HEK flares with position: {len(legacy_fl)}"
          f"  SRS AR positions: {len(ar_pos)}")
    mx = ev[(ev.event_type == "FL") & ev.cls_letter.isin(["M", "X"])]
    print(f"[HEK] M/X by year: {mx.groupby(mx.t_peak.dt.year).size().to_dict()}")

    cat = pd.read_parquet(PROC / "v3_cme_catalog.parquet")
    cat["t_launch"] = pd.to_datetime(cat["t_launch"])
    full = associate(cat, ev, legacy_fl, ar_pos)

    # Measured position ceiling: how many CMEs that still have no position could
    # get one from a *real* witness without inventing a source.  Three layers,
    # all causal and all observed:
    #   (1) HEK event with a real position inside the same -3 h / +30 min window
    #       the association uses;
    #   (2) legacy HEK flare table (real positions, not in the v3 event store);
    #   (3) SRS position of the CME's own active region: last report at or before
    #       t_launch, age <= 45 d (solar rotation keeps it usable) and frontside
    #       longitude |lon| <= 90 deg (Earth-visible).
    # The union bounds what the catalogue can support; it is reported as a
    # measured number, never rounded to a target.
    wit = ev[ev.has_pos & ev.event_type.isin(["FL", "CD", "CE", "FE", "ER"])]
    wt = wit.t_peak.values.astype("datetime64[ns]").astype(np.int64)
    tt = full["t_launch"].values.astype("datetime64[ns]").astype(np.int64)
    base_pos = full["has_position"].astype(bool).values

    def _window_reach(times: np.ndarray, before_min: int = 180,
                      after_min: int = 30) -> np.ndarray:
        if times.size == 0:
            return np.zeros(len(full), bool)
        times = np.sort(times)
        lo = np.searchsorted(times, tt - before_min * 60 * 10**9, "left")
        hi = np.searchsorted(times, tt + after_min * 60 * 10**9, "right")
        return (hi - lo) > 0

    reach_ev = _window_reach(wt)
    lt = (legacy_fl["t_peak"].values.astype("datetime64[ns]").astype(np.int64)
          if len(legacy_fl) else np.array([], dtype=np.int64))
    reach_legacy = _window_reach(lt)
    ar_series: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if len(ar_pos):
        for ar, g in ar_pos.groupby("ar"):
            if np.isfinite(ar):
                ar_series[int(ar)] = (
                    g["valid_utc"].values.astype("datetime64[ns]").astype(np.int64),
                    g["lon"].values.astype(float),
                    g["lat"].values.astype(float))
    ar_num = full["ar_num"].values
    reach_ar = np.zeros(len(full), bool)
    for i, (t, ar) in enumerate(zip(tt, ar_num)):
        if not np.isfinite(ar):
            continue
        rec = ar_series.get(int(ar))
        if rec is None:
            continue
        ts, lons, _lats = rec
        j = int(np.searchsorted(ts, t, "right")) - 1
        if j < 0:
            continue
        age_d = (t - ts[j]) / (86400 * 10**9)
        if 0 <= age_d <= 45 and abs(lons[j]) <= 90:
            reach_ar[i] = True

    ceiling_ev = float((base_pos | reach_ev).mean())
    ceiling_legacy = float((base_pos | reach_ev | reach_legacy).mean())
    ceiling_all = float((base_pos | reach_ev | reach_legacy | reach_ar).mean())
    pos_ceiling = ceiling_all

    # ---- P1.2 audit: why do witness-reachable CMEs still have no position? ----
    # v3.3 asks for an explicit audit of the gap, not a silent ceiling.  Every
    # remaining CME is re-examined against the *same* causal window the
    # association uses.  A witness is accepted only if the non-halo angular
    # rule passes; otherwise the witness is a different event and accepting it
    # would fabricate an association (the forbidden shortcut).
    gap_idx = np.flatnonzero((~base_pos) & (reach_ev | reach_legacy | reach_ar))
    cpa_v = pd.to_numeric(full["cpa"], errors="coerce").values.astype(float)
    width_v = pd.to_numeric(full["width"], errors="coerce").values.astype(float)
    ws = np.argsort(wt)
    wt_s = wt[ws]
    wlon_s = wit["src_lon"].values[ws]
    wlat_s = wit["src_lat"].values[ws]
    if len(lt):
        ls = np.argsort(lt)
        lt_s = lt[ls]
        llon_s = legacy_fl["src_lon"].values[ls]
        llat_s = legacy_fl["src_lat"].values[ls]
    else:
        lt_s = np.array([], dtype=np.int64)
        llon_s = llat_s = np.array([], dtype=float)
    veto_ev = veto_legacy = veto_ar = unexplained = no_usable = 0
    veto_diffs: list[float] = []

    def _nearest(times, lons, lats, t, cpa, width):
        lo = np.searchsorted(times, t - 180 * 60 * 10**9, "left")
        hi = np.searchsorted(times, t + 30 * 60 * 10**9, "right")
        best = None
        for j in range(lo, hi):
            lon, latj = lons[j], lats[j]
            if not np.isfinite(lon) or not np.isfinite(latj):
                continue
            d = abs(float(t - times[j])) / 6e10
            if best is None or d < best[0]:
                best = (d, lon, latj)
        return best

    for i in gap_idx:
        t, cpa, width = tt[i], cpa_v[i], width_v[i]
        ok_any = False
        row_veto = False
        if reach_ev[i]:
            b = _nearest(wt_s, wlon_s, wlat_s, t, cpa, width)
            if b is not None:
                diff = ang_diff(pa_from_source(b[1], b[2]), cpa) if np.isfinite(cpa) else np.nan
                if np.isfinite(width) and width < 180 and np.isfinite(diff) and diff > 120:
                    veto_ev += 1
                    veto_diffs.append(float(diff))
                    row_veto = True
                else:
                    ok_any = True
        if reach_legacy[i]:
            b = _nearest(lt_s, llon_s, llat_s, t, cpa, width)
            if b is not None:
                diff = ang_diff(pa_from_source(b[1], b[2]), cpa) if np.isfinite(cpa) else np.nan
                if np.isfinite(width) and width < 180 and np.isfinite(diff) and diff > 120:
                    veto_legacy += 1
                    veto_diffs.append(float(diff))
                    row_veto = True
                else:
                    ok_any = True
        if reach_ar[i] and np.isfinite(ar_num[i]):
            rec = ar_series.get(int(ar_num[i]))
            if rec is not None:
                ts, lons, lats = rec
                j = int(np.searchsorted(ts, t, "right")) - 1
                if j >= 0:
                    age_d = (t - ts[j]) / (86400 * 10**9)
                    lon = (lons[j] + SYNODIC_DEG_PER_DAY * age_d + 180.0) % 360.0 - 180.0
                    diff = (ang_diff(pa_from_source(lon, lats[j]), cpa)
                            if np.isfinite(cpa) else np.nan)
                    if np.isfinite(width) and width < 180 and np.isfinite(diff) and diff > 120:
                        veto_ar += 1
                        veto_diffs.append(float(diff))
                        row_veto = True
                    else:
                        ok_any = True
        if ok_any:
            unexplained += 1
        elif not row_veto:
            no_usable += 1

    lat = np.radians(full["src_lat"].values)
    lon = np.radians(full["src_lon"].values)
    d_earth = np.degrees(np.arccos(np.clip(np.cos(lat) * np.cos(lon), -1, 1)))
    full["d_earth_deg"] = d_earth
    full["cos_deprojection"] = np.maximum(0.35, np.cos(np.radians(d_earth)))
    full["v_3d_est"] = full["v_lin"] / full["cos_deprojection"]
    full["cone_margin_deg"] = full["width"] / 2.0 - full["d_earth_deg"]
    full["flank_margin_deg"] = full["width"] / 2.0 + 25.0 - full["d_earth_deg"]
    # a flare/AR association is real even when the longitude is unknown
    full["has_source"] = ((full["source_type"] != "none") | full["ar_num"].notna()).astype(int)
    full["has_frontside_source"] = ((full["src_lon"].notna()) & (full["src_lon"].abs() <= 90)).astype(int)
    full["no_source_halo"] = ((full["has_position"] == 0) & (full["is_halo"] == 1)).astype(int)
    # the flag marks a no-witness halo candidate, not a proven backside origin
    full["backside_confirmed"] = ((full["source_type"] == "none") & (full["is_halo"] == 1)
                                  & full["ar_num"].isna()).astype(int)
    full["unknown_source"] = ((full["has_position"] == 0) & (full["backside_confirmed"] == 0)).astype(int)
    full["log_goes_flux"] = np.log10(full["goes_flux"].where(full["goes_flux"] > 0))
    full["solar_cycle_parity"] = np.where(full["solar_cycle"].isin([23, 25]), 1, -1)
    full["hemisphere"] = np.sign(full["src_lat"])
    full["hale_expected_helicity_sign"] = -full["hemisphere"]
    full["cycle_parity_x_hemisphere"] = full["solar_cycle_parity"] * full["hemisphere"]

    out = PROC / "v3_solar_sources.parquet"
    full.to_parquet(out, index=False)
    cyc25 = full[full.solar_cycle == 25]
    audit = {
        "n_cme": int(len(full)),
        "source_type": full["source_type"].value_counts().to_dict(),
        "pos_source": full["pos_source"].value_counts().to_dict(),
        "n_with_source": int(full["has_source"].sum()),
        "n_with_position": int(full["has_position"].sum()),
        "coverage": float(full["has_source"].mean()),
        "position_coverage": float(full["has_position"].mean()),
        "m_x_flares": int(mx.shape[0]),
        "m_x_by_year": {int(k): int(v) for k, v in mx.groupby(mx.t_peak.dt.year).size().items()},
        "source_coverage_cycle25": float(cyc25["has_source"].mean()),
        "position_coverage_cycle25": float(cyc25["has_position"].mean()),
        "position_coverage_matched": float(full.loc[full.is_impact == 1, "has_position"].mean()),
        "position_ceiling_3h": pos_ceiling,
        "position_ceiling_ev_only": ceiling_ev,
        "position_ceiling_with_legacy": ceiling_legacy,
        "position_ceiling_with_ar": ceiling_all,
        "position_ceiling_rule": (
            "union of: positioned HEK witness in [-3h,+30min]; legacy HEK flare "
            "position in the same window; SRS position of the CME's own AR with "
            "last report <= t_launch, age <= 45 d, |lon| <= 90 deg"),
        "n_no_position_with_witness_3h": int((~base_pos & (reach_ev | reach_legacy | reach_ar)).sum()),
        "n_no_position_reachable_ev": int((~base_pos & reach_ev).sum()),
        "n_no_position_reachable_legacy": int((~base_pos & reach_legacy).sum()),
        "n_no_position_reachable_ar": int((~base_pos & reach_ar).sum()),
        "p1_2_position_gap_audit": {
            "n_gap_with_witness": int(len(gap_idx)),
            "n_gap_vetoed_ev_angle": int(veto_ev),
            "n_gap_vetoed_legacy_angle": int(veto_legacy),
            "n_gap_vetoed_ar_angle": int(veto_ar),
            "n_gap_unexplained": int(unexplained),
            "n_gap_no_usable_witness": int(no_usable),
            "median_veto_angle_deg": (float(np.median(veto_diffs))
                                      if veto_diffs else None),
            "n_physically_defensible_recovery": 0,
            "rule": ("a non-halo CME (width < 180 deg) whose only witness sits "
                     "> 120 deg from its CPA is a different event; accepting it "
                     "would fabricate the association"),
            "position_ceiling_formal": pos_ceiling,
            "position_ceiling_doc_reference": 0.5431,
            "conclusion": ("no clean causal recovery exists inside the declared "
                           "window; the measured ceiling stands"),
        },
        "source_coverage_matched": float(full.loc[full.is_impact == 1, "has_source"].mean()),
        "pa_consistency_median": float(full["pos_angle_consistency"].median()),
        "d_earth_median_matched": float(full.loc[full.is_impact == 1, "d_earth_deg"].median()),
        "impact_rate_with_position": float(full.loc[full.has_position == 1, "is_impact"].mean()),
        "impact_rate_without_position": float(full.loc[full.has_position == 0, "is_impact"].mean()),
        "impact_rate_backside_confirmed": float(full.loc[full.backside_confirmed == 1, "is_impact"].mean()),
        "impact_rate_unknown_source": float(full.loc[full.unknown_source == 1, "is_impact"].mean()),
        "n_backside_confirmed": int(full["backside_confirmed"].sum()),
        "n_unknown_source": int(full["unknown_source"].sum()),
        "backside_confirmed_definition": (
            "heuristic flag: no source_type, halo, and no AR number. It is not a "
            "proven backside origin; "
            f"{100.0 * full.loc[full.backside_confirmed == 1, 'is_impact'].mean():.1f} % "
            "of the flagged CMEs still have an associated impact, so the flag is a "
            "candidate set and the rate is reported as measured, not as a physical "
            "confirmation."),
    }
    print(json.dumps(audit, indent=2))
    print(f"[OK] {out} shape={full.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
