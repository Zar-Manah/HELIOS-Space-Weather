#!/usr/bin/env python3
"""Zar-Manah HELIOS v3.4 - unified operational entrypoint.

Five-stage causal pipeline, zero hardcoded fallbacks:

  impact   P(Earth impact) + operational tier          (launch time)
  transit  ML-ELEvo-DBM + residual + Mondrian CQR band (launch time)
  apex     P(Dst<=-50/-100/-200) + conformal severity  (launch time)
  mo       3DCORE-lite flux-rope Bz extrapolation       (once the MO is at L1)
  tactical Burton-O'Brien-McPherron ODE + residual      (+1h .. +12h)

Every upstream stream that is unreachable yields an explicit NaN plus a
`data_completeness` entry.  Nothing is replaced by a climatological constant.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models" / "v3"
RESULTS_DIR = BASE_DIR / "results"
sys.path.insert(0, str(BASE_DIR / "scripts"))

SWPC = "https://services.swpc.noaa.gov"
TIMEOUT = 20


# ---------------------------------------------------------------------------
# live telemetry (every failure is explicit, never silently imputed)
# ---------------------------------------------------------------------------
def _get_json(url: str):
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def _get_text(url: str):
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
            return r.read().decode(errors="replace")
    except Exception:
        return None


def fetch_swpc_series(name: str) -> pd.DataFrame:
    """Fetch an SWPC RTSW JSON time series and normalise it to (time, value...).

    The operational DSCOVR/ACE feed lives at /json/rtsw/<name>.json; the old
    /products/solar-wind/ path returns 404 and used to leave the live frame
    empty (silently degrading the tactical stage).
    """
    raw = _get_json(f"{SWPC}/json/rtsw/{name}.json")
    if not raw:
        return pd.DataFrame()
    rows = []
    for r in raw:
        t = r.get("time_tag")
        if not t:
            continue
        rec = {"datetime": pd.to_datetime(t, errors="coerce", utc=True).tz_localize(None)}
        for k, v in r.items():
            if k == "time_tag":
                continue
            try:
                rec[k] = float(v)
            except (TypeError, ValueError):
                continue
        rows.append(rec)
    # the RTSW feed interleaves several spacecraft per minute -> duplicate stamps
    return (pd.DataFrame(rows).dropna(subset=["datetime"])
            .drop_duplicates("datetime", keep="last").sort_values("datetime"))


def fetch_dst() -> pd.DataFrame:
    raw = _get_json(f"{SWPC}/products/kyoto-dst.json")
    if not raw:
        return pd.DataFrame()
    rows = []
    for r in raw:
        t = r.get("time_tag")
        try:
            rows.append({"datetime": pd.to_datetime(t, utc=True).tz_localize(None),
                         "dst": float(r.get("dst"))})
        except (TypeError, ValueError):
            continue
    return pd.DataFrame(rows).dropna().sort_values("datetime")


def fetch_kp() -> pd.DataFrame:
    raw = _get_json(f"{SWPC}/products/noaa-planetary-k-index.json")
    if not raw:
        return pd.DataFrame()
    rows = []
    for r in raw:
        t = r.get("time_tag")
        try:
            rows.append({"datetime": pd.to_datetime(t, utc=True).tz_localize(None),
                         "kp": float(r.get("Kp"))})
        except (TypeError, ValueError):
            continue
    return pd.DataFrame(rows).dropna().sort_values("datetime")


def fetch_f107() -> float:
    raw = _get_json(f"{SWPC}/json/f107_cm_flux.json")
    if not raw:
        return np.nan
    vals = []
    for r in raw:
        try:
            vals.append(float(r.get("flux")))
        except (TypeError, ValueError):
            continue
    return float(np.median(vals)) if vals else np.nan


def fetch_ace_hourly() -> pd.DataFrame:
    """ACE 1-hour mag + plasma (real L1 data, ~26 days of history).

    The RTSW 1-minute feed only carries the last 24 h, which is not enough for
    the 24 h lags and the MO detector; the ACE hourly archive covers a full
    solar-rotation-scale window and is the operational fallback.
    """
    mag = _get_json(f"{SWPC}/json/ace/mag/ace_mag_1h.json") or []
    pla = _get_json(f"{SWPC}/json/ace/swepam/ace_swepam_1h.json") or []
    if not mag and not pla:
        return pd.DataFrame()

    def frame(rows, mapping):
        out = []
        for r in rows:
            t = r.get("time_tag")
            if not t:
                continue
            rec = {"datetime": pd.to_datetime(t, errors="coerce", utc=True)}
            for src, dst in mapping.items():
                try:
                    rec[dst] = float(r[src])
                except (KeyError, TypeError, ValueError):
                    rec[dst] = np.nan
            out.append(rec)
        return pd.DataFrame(out).dropna(subset=["datetime"])

    m = frame(mag, {"gsm_bx": "bx_gsm", "gsm_by": "by_gsm",
                    "gsm_bz": "bz_gsm", "bt": "b_mag"})
    w = frame(pla, {"speed": "v_flow", "dens": "np_dens", "temperature": "temp"})
    df = m.set_index("datetime").join(w.set_index("datetime"), how="outer") \
        if not m.empty and not w.empty else (m if not m.empty else w).set_index("datetime")
    df.index = df.index.tz_localize(None)
    return df.sort_index()


def xray_class(flux: float) -> str | None:
    """Standard GOES class from peak flux in W/m^2 (NOAA scale)."""
    if not np.isfinite(flux) or flux <= 0:
        return None
    for letter, lo in (("X", 1e-4), ("M", 1e-5), ("C", 1e-6), ("B", 1e-7)):
        if flux >= lo:
            return f"{letter}{flux / lo:.1f}"
    return f"A{flux / 1e-8:.1f}"


def build_live_5min() -> pd.DataFrame:
    """Causal 5-minute L1 frame in the exact GSE/GSM basis used in training.

    v3.1 fed `detect_mo`/`run_3dcore` an *hourly* GSM frame, while v3_08 trained
    on 5-minute OMNI with GSE for `rot_angle` and `bx_gse`.  That train/serve
    skew is removed here: RTSW 1-minute mag/wind are resampled to 5 min and the
    GSM vector is rotated to GSE with the same dipole tilt used at training.
    """
    mag = fetch_swpc_series("rtsw_mag_1m")
    wind = fetch_swpc_series("rtsw_wind_1m")
    if mag.empty or wind.empty:
        return pd.DataFrame()
    keep_m = [c for c in ("datetime", "bx_gsm", "by_gsm", "bz_gsm", "bt") if c in mag]
    m = mag[keep_m].rename(columns={"bt": "b_mag"}).set_index("datetime")
    ren = {"proton_speed": "v_flow", "proton_density": "np_dens",
           "proton_temperature": "temp", "pressure": "pdyn"}
    w = wind.rename(columns={k: v for k, v in ren.items() if k in wind})
    keep_w = [c for c in ("v_flow", "np_dens", "temp", "pdyn") if c in w]
    w = w[["datetime"] + keep_w].set_index("datetime")
    df = pd.concat([m, w], axis=1).sort_index().resample("5min").mean()
    if df.empty:
        return pd.DataFrame()
    psi = dipole_tilt_rad(df.index)
    bx = df.get("bx_gsm", pd.Series(np.nan, index=df.index))
    by = df.get("by_gsm", pd.Series(np.nan, index=df.index))
    bz = df.get("bz_gsm", pd.Series(np.nan, index=df.index))
    # GSM -> GSE: rotation about the X axis by the dipole tilt (Hapgood).
    df["bx_gse"] = bx
    df["by_gse"] = by * np.cos(psi) + bz * np.sin(psi)
    df["bz_gse"] = -by * np.sin(psi) + bz * np.cos(psi)
    if "b_mag" not in df:
        df["b_mag"] = np.sqrt(by ** 2 + bz ** 2)
    if "pdyn" not in df and {"np_dens", "v_flow"} <= set(df.columns):
        df["pdyn"] = 1.6726e-6 * df["np_dens"] * df["v_flow"] ** 2
    if "beta" not in df and {"np_dens", "temp", "b_mag"} <= set(df.columns):
        df["beta"] = 3.4696e-5 * df["np_dens"] * df["temp"] / df["b_mag"] ** 2
    df["t_exp"] = (0.031 * df["v_flow"] - 5.1) ** 2 * 1e3
    df["t_ratio"] = df["temp"] / df["t_exp"]
    return df.reset_index()


def dipole_tilt_rad(index: pd.DatetimeIndex) -> np.ndarray:
    doy = index.dayofyear.values + index.hour.values / 24.0
    return np.radians(7.25) * np.sin(2 * np.pi * (doy - 80) / 365.25)


def fetch_goes_xray() -> dict:
    """GOES primary X-ray flux + the 7-day flare list (both real SWPC feeds)."""
    xr = _get_json(f"{SWPC}/json/goes/primary/xrays-1-day.json") or []
    fl = _get_json(f"{SWPC}/json/goes/primary/xray-flares-7-day.json") or []
    rows = []
    for r in xr:
        try:
            rows.append((pd.to_datetime(r["time_tag"], utc=True).tz_localize(None),
                         float(r["flux"])))
        except (KeyError, TypeError, ValueError):
            continue
    out: dict = {"n_xray_1d": len(rows), "n_flares_7d": len(fl)}
    if rows:
        rows.sort()
        t_last, f_last = rows[-1]
        f24 = [f for t, f in rows if t >= t_last - pd.Timedelta(hours=24)]
        out.update({"as_of": str(t_last), "flux_now_wm2": f_last,
                    "class_now": xray_class(f_last),
                    "max_class_24h": xray_class(max(f24) if f24 else f_last),
                    "max_flux_24h_wm2": max(f24) if f24 else f_last})
    recent = []
    for r in fl[-5:]:
        recent.append({"begin": r.get("begin_time"), "peak": r.get("peak_time"),
                       "end": r.get("end_time"), "class": r.get("max_class"),
                       "region": r.get("active_region")})
    out["recent_flares"] = recent
    return out


def fetch_hek_events(days: int = 3) -> dict:
    """HEK frontside flares + active regions with true heliographic positions."""
    end = pd.Timestamp.now(tz="UTC").tz_localize(None)
    start = end - pd.Timedelta(days=days)
    base = "https://www.lmsal.com/hek/her"
    out: dict = {"window_days": days, "flares": [], "active_regions": []}
    for etype, key, ret in (
        ("FL", "flares", "event_starttime,event_peaktime,fl_goescls,ar_noaanum,hgs_x,hgs_y"),
        ("AR", "active_regions", "event_starttime,ar_noaanum,hgs_x,hgs_y"),
    ):
        q = (f"{base}?cosec=2&cmd=search&type=column&event_type={etype}"
             f"&event_starttime={start:%Y-%m-%dT%H:%M:%S}"
             f"&event_endtime={end:%Y-%m-%dT%H:%M:%S}"
             f"&event_coordsys=helioprojective&x1=-2000&x2=2000&y1=-2000&y2=2000"
             f"&result_limit=200&return={ret}")
        body = _get_text(q)
        if not body:
            out[f"{key}_error"] = "unreachable"
            continue
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            out[f"{key}_error"] = "bad_json"
            continue
        rows = payload.get("result") or []
        if etype == "FL":
            rows = sorted(rows, key=lambda r: r.get("event_starttime") or "", reverse=True)
        out[key] = rows[:20]
        out[f"n_{key}"] = len(rows)
    return out


def fetch_enlil() -> dict:
    """Latest WSA-ENLIL run visible on SWPC.

    SWPC publishes ENLIL only as rendered frames (no numeric 1 AU time series),
    so this reports the newest frame timestamp and the frame count and states
    explicitly that no numeric profile is available.  No value is imputed.
    """
    frames = _get_json(f"{SWPC}/products/animations/enlil.json") or []
    times = []
    for f in frames:
        m = re.search(r"(\d{8}T\d{6})", f.get("url", ""))
        if m:
            try:
                times.append(pd.Timestamp(m.group(1)))
            except ValueError:
                continue
    out = {"n_frames": len(frames), "numeric_1au_profile": False}
    if times:
        out["latest_frame_utc"] = str(max(times))
    if not frames:
        out["status"] = "unreachable"
    return out


def build_live_frame() -> tuple[pd.DataFrame, dict]:
    """Merge RTSW mag/wind with Dst and Kp into an hourly causal frame."""
    mag = fetch_swpc_series("rtsw_mag_1m")
    wind = fetch_swpc_series("rtsw_wind_1m")
    ace = fetch_ace_hourly()
    dst = fetch_dst()
    kp = fetch_kp()
    f107 = fetch_f107()
    completeness = {
        "rtsw_mag_1m": int(len(mag)),
        "rtsw_wind_1m": int(len(wind)),
        "ace_hourly": int(len(ace)),
        "kyoto_dst": int(len(dst)),
        "kp": int(len(kp)),
        "f107": bool(np.isfinite(f107)),
    }
    if mag.empty and wind.empty and ace.empty:
        return pd.DataFrame(), completeness
    frames = []
    if not mag.empty:
        keep = [c for c in ("datetime", "bx_gsm", "by_gsm", "bz_gsm", "bt") if c in mag]
        m = mag[keep].rename(columns={"bt": "b_mag"})
        frames.append(m.set_index("datetime"))
    if not wind.empty:
        ren = {"proton_speed": "v_flow", "proton_density": "np_dens",
               "proton_temperature": "temp", "pressure": "pdyn"}
        w = wind.rename(columns={k: v for k, v in ren.items() if k in wind})
        keep = [c for c in ("v_flow", "np_dens", "temp", "pdyn") if c in w]
        frames.append(w[["datetime"] + keep].set_index("datetime"))
    rtsw = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    # base = ACE hourly archive, overlay = the freshest RTSW minutes
    df = ace.copy() if not ace.empty else pd.DataFrame(index=rtsw.index)
    if not rtsw.empty:
        rtsw = rtsw.resample("1h").mean()
        df = rtsw.combine_first(df)
    if df.empty:
        return pd.DataFrame(), completeness
    if "pdyn" not in df and {"np_dens", "v_flow"} <= set(df.columns):
        # Pdyn [nPa] = 1.6726e-6 * n[cm^-3] * v[km/s]^2  (physical, not fitted)
        df["pdyn"] = 1.6726e-6 * df["np_dens"] * df["v_flow"] ** 2
    if "b_mag" not in df:
        df["b_mag"] = np.sqrt(df.get("by_gsm", np.nan) ** 2 + df.get("bz_gsm", np.nan) ** 2)
    if "beta" not in df and {"np_dens", "temp", "b_mag"} <= set(df.columns):
        # plasma beta = 2 mu0 n k T / B^2  ->  3.47e-5 * n[cm^-3] T[K] / B[nT]^2
        df["beta"] = 3.4696e-5 * df["np_dens"] * df["temp"] / df["b_mag"] ** 2
    df["e_field"] = -df.get("v_flow", np.nan) * df.get("bz_gsm", np.nan) / 1000.0
    # Dst is a 1 h index and Kp a 3 h index: carrying the last published value
    # forward is physically correct.  A hard limit used to leave the newest row
    # with NaN Kp, which sent the XGB residual to -150 nT on a dead-quiet hour.
    if not dst.empty:
        df = df.join(dst.set_index("datetime")["dst"], how="left")
        df["dst"] = df["dst"].ffill()
    else:
        df["dst"] = np.nan
    if not kp.empty:
        df = df.join(kp.set_index("datetime")["kp"], how="left")
        df["kp"] = df["kp"].ffill()
    else:
        df["kp"] = np.nan
    df["f107"] = f107
    df = df.reset_index()
    completeness["f107_value"] = f107 if np.isfinite(f107) else None
    return df, completeness


# ---------------------------------------------------------------------------
# stage 5: tactical
# ---------------------------------------------------------------------------
def run_tactical(live: pd.DataFrame, bz_forecast: float | None = None) -> dict:
    from v3_09_train_tactical import (FEATURES, HORIZONS, add_ode_columns,
                                      build_features, ode_rollout)

    if live.empty or len(live) < 30:
        return {"status": "insufficient_l1_data", "n_hours": int(len(live))}
    h = build_features(live.copy())
    h = add_ode_columns(h)
    h = h.loc[:, ~h.columns.duplicated(keep="last")]
    missing = [c for c in FEATURES if c not in h.columns]
    for c in missing:
        h[c] = np.nan
    X = h[FEATURES].replace([np.inf, -np.inf], np.nan).tail(1)
    for c in ("kp", "dst"):
        if c in X and X[c].isna().any():
            X[c] = X[c].fillna(float(h[c].dropna().iloc[-1])) if h[c].notna().any() else X[c]
    out = {"as_of": str(h["datetime"].iloc[-1]), "dst_now": _f(h["dst"].iloc[-1]),
           "horizons": {}, "missing_features": missing,
           "nan_in_last_row": sorted(X.columns[X.isna().any()].tolist())}
    for hor in HORIZONS:
        path = MODELS_DIR / f"tactical_h{hor}.joblib"
        if not path.exists():
            continue
        blob = joblib.load(path)
        Xh = X.reindex(columns=blob["features"])
        members = blob.get("members") or [blob[k] for k in ("lgb", "xgb", "lgb_huber")
                                         if k in blob]
        resid = float(np.mean([m.predict(Xh)[0] for m in members]))
        base = _f(h[f"dst_ode_{hor}"].iloc[-1])
        raw = base + resid if np.isfinite(base) else float(resid)
        dst_now = h["dst"].iloc[-1]
        # The persistence blend weight was chosen by pre-2020 rolling-origin CV.
        wb = blob.get("blend_persistence_weight")
        wb = 1.0 if wb is None else float(wb)
        pred = wb * raw + (1 - wb) * float(dst_now) if np.isfinite(dst_now) else raw
        # Storm-tail gain fitted on the pre-2020 CV fold (stored in the model).
        tg = float(blob.get("tail_gain") or 1.0)
        if tg != 1.0 and np.isfinite(pred) and pred < -30.0:
            pred = -30.0 + tg * (pred + 30.0)
        reg = 0 if (not np.isfinite(dst_now) or dst_now > -30) else (1 if dst_now > -60 else 2)
        q_state = blob.get("aci_q0") or {}
        qq = q_state.get(reg, q_state.get(str(reg)))
        scale = 1.0
        if blob.get("aci_mode") == "normalized":
            ref = float(blob.get("scale_ref") or 1.0)
            std_now = float(h["dst_std_6h"].iloc[-1]) if "dst_std_6h" in h else np.nan
            if np.isfinite(std_now) and ref > 1e-6:
                scale = float(np.clip(std_now / ref, 0.5, 3.0))
        entry = {"dst_pred": _f(pred), "dst_ode": _f(base), "residual": _f(resid),
                 "persistence_weight": _f(1 - wb)}
        if qq is not None and np.isfinite(qq):
            entry.update({"dst_band": [_f(pred - qq * scale), _f(pred + qq * scale)],
                          "aci_width": _f(2 * qq * scale),
                          "aci_regime": int(reg),
                          "aci_mode": blob.get("aci_mode", "raw")})
        out["horizons"][f"+{hor}h"] = entry
    # ---- 3DCORE -> tactical coupling -------------------------------------
    # When a magnetic obstacle is being crossed, the flux-rope extrapolation
    # supplies a forecast Bz that replaces the persistent-Bz assumption inside
    # the ring-current ODE.  This is physics, not a fitted correction: no model
    # is retrained and the uncoupled prediction stays visible for comparison.
    if bz_forecast is not None and np.isfinite(bz_forecast):
        last = h.iloc[-1]
        v_now = _f(last.get("v_flow"))
        p_now = _f(last.get("pdyn"))
        d_now = _f(last.get("dst"))
        if v_now is not None and d_now is not None:
            ey = -v_now * float(bz_forecast) / 1000.0
            out["coupled_3dcore"] = {
                "bz_forecast_nt": _f(bz_forecast), "ey_forecast_mvm": _f(ey),
                "dst_pred_6h_ode_coupled": _f(ode_rollout(d_now, p_now or np.nan, ey, 6)),
                "note": "ODE re-run with the flux-rope Bz; uncoupled horizons above",
            }
    return out


def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _scalar(row, key: str, default=np.nan) -> float:
    """Read one numeric cell from a one-row frame or a plain mapping.

    `run_apex` builds `pd.DataFrame([cme])`, so `row.get(key)` returns a
    `Series` as soon as the column exists (live path with ambient telemetry, or
    the historical battery).  `float(Series)` raises, which is how the v3.2
    `float.abs()` bug survived: it only failed when the key was present.
    """
    if hasattr(row, "columns"):
        if key not in row.columns:
            return default
        v = row[key]
        if hasattr(v, "iloc"):
            if len(v) == 0:
                return default
            v = v.iloc[0]
    else:
        v = row.get(key, default)
    try:
        out = float(v)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


# ---------------------------------------------------------------------------
# stage 3: transit
# ---------------------------------------------------------------------------
def run_transit(cme: dict) -> dict:
    """Point transit + conformal window from the fitted model blob.

    The physics backbone (w_amb, gamma) and the DBM grid features are built by
    the exact same shared code used at training time, so there is no
    train/serve skew.
    """
    from v3_common import (add_transit_features, design_matrix,
                           transit_physics)

    path = MODELS_DIR / "transit_v3.joblib"
    if not path.exists():
        return {"status": "model_missing"}
    blob = joblib.load(path)
    row = pd.DataFrame([cme])
    for c in set(blob["features"]) | {"t_launch"}:
        if c not in row:
            row[c] = np.nan
    if "t_launch" not in row or pd.isna(row["t_launch"].iloc[0]):
        row["t_launch"] = pd.Timestamp(cme["launch_time"])
    row = add_transit_features(row)
    t_phys, v1, reach = transit_physics(row, blob["physics"])
    if not bool(reach[0]):
        # launch speed does not exceed the fitted ambient wind: the drag model
        # has no physical solution, so report that instead of inventing a time.
        return {"status": "unreachable_physics",
                "v_launch_km_s": _f(cme.get("v_lin")),
                "w_amb_km_s": _f(blob["physics"]["w_amb_km_s"]),
                "gamma_km_inv": _f(blob["physics"]["gamma_km_inv"]),
                "fallback_ballistic_h": _f(t_phys[0])}
    X = design_matrix(row, blob["features"])
    # Weighted blend of the members chosen by pre-2020 OOF (v3.2).  `models`
    # may be heterogeneous (LightGBM bag, XGBoost, ExtraTrees, Ridge, DBM
    # residual learner), so the weights are part of the artefact.
    ws = blob.get("model_weights") or [1.0 / len(blob["models"])] * len(blob["models"])
    # v3.4: members may have been trained on log(T); the artefact records the
    # target space so serving applies the same exponentiation as training.
    is_log = blob.get("target_space") == "log"
    t_hat = 0.0
    for m, w in zip(blob["models"], ws):
        if isinstance(m, list):
            if is_log:
                pred = float(np.mean([np.exp(np.clip(x.predict(X)[0], 1.5, 5.1))
                                      for x in m]))
            else:
                pred = float(np.mean([x.predict(X)[0] for x in m]))
        else:
            pv = float(m.predict(X)[0])
            pred = float(np.exp(np.clip(pv, 1.5, 5.1))) if is_log else pv
        t_hat += float(w) * pred
    edges = blob["regime_edges_v_lin"]
    regime_idx = int(np.digitize([float(cme.get("v_lin", 0.0))], edges)[0])
    # Serve exactly the band the training run selected on out-of-fold data;
    # no clamp, no fallback level.  A mismatch here would be train/serve skew.
    band_mode = blob.get("band_mode", "cqr")
    if band_mode == "abs_conformal":
        q = float(blob["q_abs"].get(str(regime_idx), blob["q_abs"]["global"]))
        lo = t_hat - q
        hi = t_hat + q
    elif band_mode == "abs_conformal_normalized":
        q = float(blob["q_norm"].get(str(regime_idx), blob["q_norm"]["global"]))
        sg = float(blob["regime_sigma"].get(str(regime_idx),
                                            blob["regime_sigma"]["global"]))
        lo = t_hat - q * sg
        hi = t_hat + q * sg
    else:
        q = float(blob["q"].get(str(regime_idx), blob["q"]["global"]))
        lo = float(blob["lo_model"].predict(X)[0]) - q
        hi = float(blob["hi_model"].predict(X)[0]) + q
    return {
        "arrival_hours": _f(t_hat),
        "arrival_time_utc": (pd.Timestamp(cme["launch_time"])
                             + pd.Timedelta(hours=t_hat)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_hours": [_f(lo), _f(hi)],
        "window_width_h": _f(hi - lo),
        "conformal_regime": regime_idx, "band_mode": band_mode, "q": _f(q),
        "t_phys_h": _f(t_phys[0]),
        "v1au_km_s": _f(v1[0]),
        "w_amb_km_s": _f(blob["physics"]["w_amb_km_s"]),
        "gamma_km_inv": _f(blob["physics"]["gamma_km_inv"]),
        "reachable_physics": True,
    }


def _fill(s, default):
    try:
        v = float(s.iloc[0])
        return v if np.isfinite(v) else default
    except (TypeError, ValueError, IndexError):
        return default


# ---------------------------------------------------------------------------
# stage 2: impact
# ---------------------------------------------------------------------------
def _eval_impact_blob(blob: dict, row: pd.DataFrame) -> float:
    """Calibrated impact probability for one fitted pipeline blob."""
    from v3_common import design_matrix
    from v3_05_train_impact import apply_calibration

    X = design_matrix(row, blob["features"])
    raw = np.column_stack([m.predict_proba(X)[:, 1] for m in blob["models"]])
    p_raw = blob["meta"].predict_proba(raw)[:, 1]
    return float(apply_calibration(p_raw, blob["calib"], blob["calib_method"])[0])


def run_impact(cme: dict) -> dict:
    path = MODELS_DIR / "impact_full.joblib"
    if not path.exists():
        return {"status": "model_missing"}
    full = joblib.load(path)
    row = pd.DataFrame([cme])
    lon = cme.get("src_lon", np.nan)
    row = _fill_geometry(row, cme)
    row["no_source_halo"] = int((not np.isfinite(lon)) and cme.get("is_halo", 0) == 1)
    row["cpa_sin"] = np.sin(np.radians(cme.get("cpa", 0.0) if np.isfinite(cme.get("cpa", np.nan)) else 0.0))
    row["cpa_cos"] = np.cos(np.radians(cme.get("cpa", 0.0) if np.isfinite(cme.get("cpa", np.nan)) else 0.0))
    has_pos = bool(np.isfinite(cme.get("src_lon", np.nan))
                   and np.isfinite(cme.get("src_lat", np.nan)))
    # The served impact predictor is whichever model won the pre-2020 OOF
    # comparison in v3_05: the unified model, the source-aware switch or the
    # 0.4/0.4/0.2 blend of full + switch + no-source.  Each branch loads the
    # artefacts the training run actually persisted, so the live path and the
    # backtest parquet cannot disagree.
    sw_path = MODELS_DIR / "impact_switch.joblib"
    bl_path = MODELS_DIR / "impact_blend.joblib"
    ns_path = MODELS_DIR / "impact_no_source.joblib"
    sw = joblib.load(sw_path) if sw_path.exists() else None
    bl = joblib.load(bl_path) if bl_path.exists() else None
    ns = joblib.load(ns_path) if ns_path.exists() else None
    needed = set(full["features"])
    if sw:
        needed |= set(sw["localized"]["features"]) | set(sw["blind"]["features"])
    if ns:
        needed |= set(ns["features"])
    for c in needed:
        if c not in row:
            row[c] = np.nan
    if bl is not None and sw is not None and ns is not None:
        w = bl.get("weights") or [0.4, 0.4, 0.2]
        branch = sw["localized"] if has_pos else sw["blind"]
        p = (float(w[0]) * _eval_impact_blob(full, row)
             + float(w[1]) * _eval_impact_blob(branch, row)
             + float(w[2]) * _eval_impact_blob(ns, row))
        tau_watch = float(bl["tau_watch"])
        tau_high = float(bl["tau_high"])
        served = "impact_blend"
    elif sw is not None:
        branch = sw["localized"] if has_pos else sw["blind"]
        p = _eval_impact_blob(branch, row)
        tau_watch = float(sw["tau_watch"])
        tau_high = float(sw["tau_high"])
        served = "source_aware_localized" if has_pos else "source_aware_blind"
    else:
        p = _eval_impact_blob(full, row)
        tau_watch = float(full["tau_watch"])
        tau_high = float(full["tau_high"])
        served = "model_full"
    tier = ("DIRECT_HIT_ALERT" if p >= tau_high
            else "GLANCING_OR_WATCH" if p >= tau_watch else "ALL_CLEAR")
    return {"p_impact": _f(p), "tier": tier, "model": served,
            "tau_watch": _f(tau_watch), "tau_high": _f(tau_high)}


def build_ambient_context(live: pd.DataFrame) -> dict:
    """Hourly OMNI-style context features, identical to v3_04's definitions.

    The training frame (`v3_omni_context.parquet`) is built with rolling
    windows over a strict hourly grid and `min_periods = win // 3`.  Serving
    must use the same windows or the apex model sees a different distribution;
    missing history stays NaN (never a climatological constant).
    """
    if live is None or live.empty:
        return {}
    h = live.copy()
    h["datetime"] = pd.to_datetime(h["datetime"])
    h = h.sort_values("datetime").set_index("datetime")
    h = h.reindex(pd.date_range(h.index.min(), h.index.max(), freq="h"))
    h.index.name = "datetime"
    out = {}
    if "pdyn" not in h and {"np_dens", "v_flow"} <= set(h.columns):
        h["pdyn"] = 1.6726e-6 * h["np_dens"] * h["v_flow"] ** 2
    if "e_field" not in h and {"v_flow", "bz_gsm"} <= set(h.columns):
        h["e_field"] = -h["v_flow"] * h["bz_gsm"] / 1000.0
    for win in (24, 48):
        r = h.rolling(win, min_periods=max(1, win // 3))
        if "v_flow" in h:
            out[f"v_mean_{win}h"] = float(r["v_flow"].mean().iloc[-1])
        if "bz_gsm" in h:
            out[f"bz_min_{win}h"] = float(r["bz_gsm"].min().iloc[-1])
        if "dst" in h:
            out[f"dst_min_{win}h"] = float(r["dst"].min().iloc[-1])
    for c in ("dst", "bz_gsm", "b_mag", "v_flow", "np_dens", "pdyn", "e_field"):
        out[f"t0_{c}"] = float(h[c].iloc[-1]) if c in h else np.nan
    lag_h = int(round(27.275 * 24))
    for c in ("v_flow", "np_dens", "b_mag", "bz_gsm"):
        if c in h:
            smooth = h[c].rolling(25, center=True, min_periods=6).mean()
            out[f"carr_{c}"] = float(smooth.shift(lag_h).iloc[-1])
        else:
            out[f"carr_{c}"] = np.nan
    return out


def _solar_cycle(ts) -> int:
    """Same cycle rule used to build the catalogue (v3_01): 23 <=2008, 24 <=2019, 25 else."""
    y = pd.Timestamp(ts).year
    return 23 if y <= 2008 else (24 if y <= 2019 else 25)


def _fill_geometry(row: pd.DataFrame, cme: dict) -> pd.DataFrame:
    """Source geometry exactly as v3_02 computes it at training time."""
    lon = cme.get("src_lon", np.nan)
    lat = cme.get("src_lat", np.nan)
    ar = cme.get("ar_num", np.nan)
    row["abs_lon"] = abs(lon) if np.isfinite(lon) else np.nan
    row["is_west"] = int(lon > 0) if np.isfinite(lon) else 0
    row["is_east"] = int(lon < 0) if np.isfinite(lon) else 0
    # `has_source` and `has_position` are *different* facts.  A CME can be
    # associated with an active region (source known) while its longitude is
    # still unknown; v3.2 collapsed both into one flag, so the live path could
    # never reproduce the `has_source=1, has_position=0` rows seen in training.
    has_pos = int(np.isfinite(lon) and np.isfinite(lat))
    has_src = int(has_pos == 1 or (ar is not None and np.isfinite(ar)))
    row["has_source"] = has_src
    row["has_position"] = has_pos
    if np.isfinite(lon) and np.isfinite(lat):
        d = float(np.degrees(np.arccos(np.clip(
            np.cos(np.radians(lat)) * np.cos(np.radians(lon)), -1, 1))))
        cosd = float(np.maximum(0.35, np.cos(np.radians(d))))
        row["d_earth_deg"] = d
        row["cos_deprojection"] = cosd
        row["v_3d_est"] = float(cme["v_lin"]) / cosd
        row["cone_margin_deg"] = float(cme["width"]) / 2.0 - d
        row["flank_margin_deg"] = float(cme["width"]) / 2.0 + 25.0 - d
        row["v_eff_launch"] = float(cme["v_lin"]) / float(np.clip(cosd, 0.45, 1.0))
    else:
        for c in ("d_earth_deg", "cos_deprojection", "v_3d_est", "cone_margin_deg",
                  "flank_margin_deg"):
            row[c] = np.nan
        row["v_eff_launch"] = float(cme["v_lin"])
    return row


# ---------------------------------------------------------------------------
# stage 4A: apex
# ---------------------------------------------------------------------------
def run_apex(cme: dict, transit: dict, ambient: dict | None = None) -> dict:
    from scipy.stats import norm
    from v3_common import design_matrix
    from v3_07_train_apex import add_geometry

    path = MODELS_DIR / "apex_v3.joblib"
    if not path.exists():
        return {"status": "model_missing"}
    blob = joblib.load(path)
    row = pd.DataFrame([cme])
    row["t_launch"] = pd.Timestamp(cme["launch_time"])
    row = _fill_geometry(row, cme)
    row["v1au_dbm"] = transit.get("v1au_km_s", np.nan)
    row["t_phys"] = transit.get("t_phys_h", np.nan)
    if ambient:
        for k, v in ambient.items():
            row[k] = v
    # physical 1-AU forcing estimates, identical to v3_07.attach_physics
    np0 = _scalar(row, "t0_np_dens")
    row["pdyn_1au"] = 1.6726e-6 * np0 * row["v1au_dbm"] ** 2
    bz0 = _scalar(row, "t0_bz_gsm")
    row["ey_1au_max"] = row["v1au_dbm"] * (abs(bz0) if np.isfinite(bz0) else np.nan) * 1e-3
    row = add_geometry(row)
    for c in blob["features"]:
        if c not in row:
            row[c] = np.nan
    X = design_matrix(row, blob["features"])
    mu = float(blob["mu_model"].predict(X)[0])
    sig = float(np.clip(blob["sig_model"].predict(X)[0], 15.0, 220.0))
    if blob.get("band_mode") == "normalized":
        half = float(blob["q90_norm"]) * sig
    else:
        half = float(blob["q90_dst"])
    out = {"p_dst_le": {}, "dst_min_pred": _f(mu),
           "dst_min_band": [_f(mu - half), _f(mu + half)],
           "dst_min_band_mode": blob.get("band_mode", "static"),
           "sigma_conditional_nt": _f(sig)}
    bpath = MODELS_DIR / "apex_bmag.joblib"
    if bpath.exists():
        bb = joblib.load(bpath)
        bm = float(bb["model"].predict(X)[0])
        out["b_mag_max_pred"] = _f(bm)
        out["b_mag_max_band"] = [_f(bm - bb["q90"]), _f(bm + bb["q90"])]
    from v3_07_train_apex import _apply_platt
    for thr in (50, 100, 200):
        key = f"-{thr}"
        which = blob["selected"].get(key)
        if which is None:
            continue
        if which == "gaussian_tail":
            p_raw = float(norm.cdf(((-thr) - mu) / blob["sigma"]))
        elif which == "gaussian_tail_conditional":
            p_raw = float(norm.cdf(((-thr) - mu) / sig))
        elif which == "conformal_empirical":
            z = ((-thr) - mu) / sig
            p_raw = float(np.mean(blob["z_oof"] <= z))
        else:
            entry = blob["classifiers"].get(key)
            if entry is None:
                continue
            kind = entry.get("kind") if isinstance(entry, dict) else None
            if kind in ("compact_logit", "blend_logit_tail"):
                feats = entry.get("features") or []
                for c in feats:
                    if c not in row:
                        row[c] = np.nan
                Xc = design_matrix(row, feats)
                clf = entry.get("model")
                p_logit = (float(entry.get("prior", 0.0)) if clf is None
                           else float(clf.predict_proba(Xc)[:, 1][0]))
                if kind == "blend_logit_tail":
                    p_tail = float(norm.cdf(((-thr) - mu) / blob["sigma"]))
                    p_raw = 0.5 * p_logit + 0.5 * p_tail
                else:
                    p_raw = p_logit
            elif isinstance(entry, dict):
                clf = entry.get("model")
                p_raw = (float(entry.get("prior", 0.0)) if clf is None
                         else float(clf.predict_proba(X)[:, 1][0]))
            else:  # legacy artifact: bare estimator
                p_raw = float(entry.predict_proba(X)[:, 1][0])
        ab = blob["platt_ab"].get(key)
        p = float(_apply_platt(np.array([p_raw]), tuple(ab))[0]) if ab else p_raw
        out["p_dst_le"][key] = _f(p)
        out.setdefault("model_used", {})[key] = which
    # Same three-way rule as `v3_07.bz_tag`, including its launch-speed input
    # (`v_lin`, not the decelerated 1-AU speed).  `sharp_r_value` is SHARP
    # `R_VALUE`, an unsigned flux, so it cannot encode Hale polarity: the
    # branch only flags a southern source with SHARP coverage.  The live CLI
    # does not fetch it, so without `--cme_sharp_r` that branch is skipped.
    r = cme.get("sharp_r_value", np.nan)
    r = float(r) if isinstance(r, (int, float, np.floating)) else np.nan
    lat0 = cme.get("src_lat", np.nan)
    lat0 = float(lat0) if isinstance(lat0, (int, float, np.floating)) else np.nan
    hem = float(np.sign(lat0)) if np.isfinite(lat0) else 0.0
    v0 = cme.get("v_lin", np.nan)
    v0 = float(v0) if isinstance(v0, (int, float, np.floating)) else np.nan
    if np.isfinite(r) and hem < 0 and r > 0:
        out["bz_risk_tag"] = "SOUTHERN_SOURCE"
    elif np.isfinite(v0) and v0 >= 900:
        out["bz_risk_tag"] = "SHEATH_DRIVEN"
    else:
        out["bz_risk_tag"] = "POLARITY_UNCERTAIN_AWAIT_L1"
    out["bz_risk_tag_inputs"] = {
        "sharp_r_value": _f(r), "src_lat": _f(lat0), "v_lin": _f(v0),
    }
    return out


# ---------------------------------------------------------------------------
# stage 4B: L1 magnetic obstacle + 3DCORE-lite
# ---------------------------------------------------------------------------
def detect_mo(live: pd.DataFrame) -> dict:
    path = MODELS_DIR / "mo_net.joblib"
    if not path.exists() or len(live) < 72 or "bx_gse" not in live:
        return {"mo_active": None, "reason": "model_or_data_missing"}
    blob = joblib.load(path)
    h = live.copy().tail(288)  # 24 h of 5-min samples, identical to v3_08 training
    h["t_exp"] = (0.031 * h["v_flow"] - 5.1) ** 2 * 1e3
    h["t_ratio"] = h["temp"] / h["t_exp"]
    for w in (6, 12, 36):
        h[f"b_std_{w}"] = h["b_mag"].rolling(w, min_periods=w // 2).std()
        h[f"v_mean_{w}"] = h["v_flow"].rolling(w, min_periods=w // 2).mean()
        h[f"pdyn_max_{w}"] = h["pdyn"].rolling(w, min_periods=w // 2).max()
        h[f"beta_mean_{w}"] = h["beta"].rolling(w, min_periods=w // 2).mean()
        h[f"bz_min_{w}"] = h["bz_gsm"].rolling(w, min_periods=w // 2).min()
    h["db_mag"] = h["b_mag"].diff()
    h["dpdyn"] = h["pdyn"].diff()
    # v3_08 computes rot_angle on the GSE vector (line 219); serving must match.
    u = h[["bx_gse", "by_gse", "bz_gse"]].div(h["b_mag"].replace(0, np.nan), axis=0)
    dot = (u * u.shift()).sum(axis=1).clip(-1, 1)
    h["rot_angle"] = np.degrees(np.arccos(dot.values))
    h["rot_var_12"] = h["rot_angle"].rolling(12, min_periods=6).mean()
    X = h.reindex(columns=blob["features"]).replace([np.inf, -np.inf], np.nan).tail(1)
    p = float(blob["model"].predict_proba(X)[:, 1][0])
    return {"mo_active": bool(p >= 0.5), "p_mo": _f(p)}


def run_3dcore(live: pd.DataFrame) -> dict:
    from v3_08_flux_rope_3dcore import fit_flux_rope, lundquist_profile

    if len(live) < 36 or "bx_gse" not in live:
        return {"status": "insufficient_data"}
    h = live.tail(288).dropna(subset=["bx_gse", "by_gsm", "bz_gsm", "v_flow"])
    if len(h) < 36:
        return {"status": "insufficient_data"}
    t_s = (h["datetime"] - h["datetime"].iloc[0]).dt.total_seconds().values
    # v3_08 fits the rope with bx_gse / by_gsm / bz_gsm (line 150).
    B = h[["bx_gse", "by_gsm", "bz_gsm"]].values
    v = float(np.nanmedian(h["v_flow"].values))
    fit = fit_flux_rope(t_s, B, v)
    if fit is None:
        return {"status": "fit_failed"}
    fut = np.arange(0, 6 * 3600 + 1, 300.0) + t_s[-1]
    pred = lundquist_profile(fut, v, fit["b0"], fit["theta"], fit["phi"],
                             fit["t_c"], fit["radius"], fit["helicity"])
    return {"status": "ok", **{k: _f(vv) for k, vv in fit.items()},
            "bz_forecast_6h_min": _f(np.min(pred[:, 2])),
            "bz_forecast_6h_mean": _f(np.mean(pred[:, 2]))}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_cme(args) -> dict:
    lon = args.cme_lon
    lat = args.cme_lat
    width = 360.0 if args.halo else args.cme_width
    d = np.nan
    if np.isfinite(lon) and np.isfinite(lat):
        d = float(np.degrees(np.arccos(np.clip(
            np.cos(np.radians(lat)) * np.cos(np.radians(lon)), -1, 1))))
    launch = args.launch_time or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    v = args.cme_speed
    v20 = args.cme_v20r
    mass = args.cme_mass
    energy = args.cme_energy
    return {
        "launch_time": launch,
        "v_lin": v, "width": width,
        "cpa": np.nan if width >= 360 else args.cme_cpa,
        "is_halo": int(width >= 360), "is_partial_halo": int(180 <= width < 360),
        "is_narrow": int(width < 30), "accel_true": args.cme_accel,
        "v_20R": v20,
        "v_20r_ratio": (float(v20 / v) if (np.isfinite(v20) and np.isfinite(v) and v > 0)
                        else np.nan),
        "dv_2nd": args.cme_dv2,
        "mass": mass, "energy": energy,
        "log_mass": float(np.log10(mass)) if (np.isfinite(mass) and mass > 0) else np.nan,
        "log_energy": float(np.log10(energy)) if (np.isfinite(energy) and energy > 0) else np.nan,
        "log_speed": float(np.log10(v)) if (np.isfinite(v) and v > 0) else np.nan,
        "log_momentum": (float(np.log10(mass * v))
                         if (np.isfinite(mass) and mass > 0
                             and np.isfinite(v) and v > 0) else np.nan),
        "src_lon": lon, "src_lat": lat,
        "sharp_r_value": args.cme_sharp_r,
        "solar_cycle": _solar_cycle(launch), "ar_num": args.cme_ar,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="HELIOS v3.4 operational forecast")
    ap.add_argument("--version", "-V", action="version", version="Zar-Manah HELIOS v3.4")
    ap.add_argument("--live", action="store_true", help="fetch real-time NOAA SWPC telemetry")
    ap.add_argument("--cme", action="store_true", help="run the launch-time CME pipeline")
    ap.add_argument("--cme_speed", "--cme-speed", "--speed", dest="cme_speed", type=float)
    ap.add_argument("--cme_width", "--cme-width", "--width", dest="cme_width", type=float, default=60.0)
    ap.add_argument("--cme_lat", "--cme-lat", "--lat", dest="cme_lat", type=float, default=np.nan)
    ap.add_argument("--cme_lon", "--cme-lon", "--lon", dest="cme_lon", type=float, default=np.nan)
    ap.add_argument("--cme_accel", "--cme-accel", "--accel", dest="cme_accel", type=float, default=np.nan)
    ap.add_argument("--cme_cpa", "--cme-cpa", "--cpa", dest="cme_cpa", type=float, default=np.nan,
                    help="central position angle in degrees (ignored for full halos)")
    ap.add_argument("--cme_v20r", "--cme-v20r", dest="cme_v20r", type=float, default=np.nan,
                    help="CDAW 20R speed (km/s) from the DONKI CDAW analysis")
    ap.add_argument("--cme_dv2", "--cme-dv2", dest="cme_dv2", type=float, default=np.nan,
                    help="second-order speed difference dv2 (km/s) from CDAW")
    ap.add_argument("--cme_mass", "--cme-mass", dest="cme_mass", type=float, default=np.nan,
                    help="CME mass estimate (g) from the DONKI CDAW analysis")
    ap.add_argument("--cme_energy", "--cme-energy", dest="cme_energy", type=float, default=np.nan,
                    help="CME kinetic energy estimate (erg) from the DONKI CDAW analysis")
    ap.add_argument("--cme_ar", "--cme-ar", dest="cme_ar", type=float, default=np.nan,
                    help="NOAA active-region number (source known, position may be unknown)")
    ap.add_argument("--cme_sharp_r", "--cme-sharp-r", dest="cme_sharp_r", type=float,
                    default=np.nan,
                    help="SHARP R_VALUE (unsigned flux near polarity inversion lines); "
                         "it carries no Hale-polarity sign, so it only flags a "
                         "southern source; not fetched automatically")
    ap.add_argument("--halo", action="store_true")
    ap.add_argument("--launch_time", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--dashboard", action="store_true")
    args = ap.parse_args()

    report = {"version": "3.4.0", "generated_utc": datetime.now(timezone.utc).isoformat()}
    live = pd.DataFrame()
    if args.live:
        live, completeness = build_live_frame()
        live5 = build_live_5min()
        report["data_completeness"] = completeness
        report["data_completeness"]["l1_5min_rows"] = int(len(live5))
        if not live.empty:
            report["l1_now"] = {
                "as_of": str(live["datetime"].iloc[-1]),
                "b_mag": _f(live["b_mag"].iloc[-1]),
                "bz_gsm": _f(live["bz_gsm"].iloc[-1]),
                "v_flow": _f(live["v_flow"].iloc[-1]),
                "np_dens": _f(live["np_dens"].iloc[-1]),
                "dst": _f(live["dst"].iloc[-1]),
            }
            report["mo"] = detect_mo(live5)
            bz6 = None
            if report["mo"].get("mo_active"):
                report["flux_rope_3dcore"] = run_3dcore(live5)
                if report["flux_rope_3dcore"].get("status") == "ok":
                    bz6 = report["flux_rope_3dcore"].get("bz_forecast_6h_min")
            report["tactical"] = run_tactical(live, bz_forecast=bz6)
            report["live_feeds"] = {
                "goes_xray": fetch_goes_xray(),
                "hek": fetch_hek_events(days=3),
                "enlil": fetch_enlil(),
            }
    if args.cme:
        if not args.cme_speed:
            ap.error("--cme requires --cme_speed")
        cme = build_cme(args)
        report["cme_input"] = {k: _f(v) if isinstance(v, (int, float, np.floating)) else v
                               for k, v in cme.items()}
        report["impact"] = run_impact(cme)
        report["transit"] = run_transit(cme)
        ambient = build_ambient_context(live) if not live.empty else {}
        report["apex"] = run_apex(cme, report["transit"], ambient)
    if args.dashboard:
        print("HELIOS v3.4 serves no dashboard from this CLI: open Zar-Manah_HELIOS.html "
              "(physics + live L1) or run server.py (this pipeline on 127.0.0.1:8787). "
              "Use --json for machine output.")
    if args.json:
        print(json.dumps(report, indent=2, default=float))
    else:
        _pretty(report)
    return 0


def _pretty(report: dict) -> None:
    print(f"HELIOS v3.4  generated {report['generated_utc']}")
    if "data_completeness" in report:
        print("data completeness:", report["data_completeness"])
    if "l1_now" in report:
        print("L1 now:", report["l1_now"])
    if "tactical" in report and "horizons" in report["tactical"]:
        print(f"tactical (Dst now {report['tactical'].get('dst_now')}):")
        for h, v in report["tactical"]["horizons"].items():
            print(f"  {h:>5}  pred={v['dst_pred']}  ode={v['dst_ode']}  resid={v['residual']}")
    if "mo" in report:
        print("magnetic obstacle:", report["mo"])
    if "flux_rope_3dcore" in report:
        print("3DCORE:", report["flux_rope_3dcore"])
    if "cme_input" in report:
        print("CME:", report["cme_input"])
    if "impact" in report:
        print("impact:", report["impact"])
    if "transit" in report:
        print("transit:", report["transit"])
    if "apex" in report:
        print("apex:", report["apex"])


if __name__ == "__main__":
    sys.exit(main())
