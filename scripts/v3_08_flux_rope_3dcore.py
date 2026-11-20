#!/usr/bin/env python3
"""HELIOS v3.4 - Stage 4B: L1 magnetic-obstacle detection + Lundquist flux-rope fit.

Two coupled pieces:

1. MO-Net: LightGBM binary classifier on 5-min OMNI windows that decides whether
   the spacecraft is inside a magnetic obstacle (ICME/MC).  Labels come from the
   Richardson & Cane ICME intervals (mc_flag >= 1), never from hand rules.

2. 3DCORE-lite: constant-alpha Lundquist (circular-cylinder) force-free fit
   B(t) = B0 [ J0(a r) a_hat + H J1(a r) ( ... ) ],  r(t) = |p(t)| / R
   to the first 20/35/50 % of the observed magnetic obstacle.  The remaining
   Bz(t) profile and Bz_min are extrapolated and scored against the truth.

Strict OOS protocol: the MO-Net classifier and the 3DCORE amplitude calibrator
are trained on Cycles 23-24 and scored on Cycle 25 (mc_flag >= 1).  The
Lundquist fit has no learned parameters, so it runs on every magnetic cloud;
only the Cycle-25 fits enter the scored table.

Outputs
  models/v3/mo_net.joblib
  results/v3_3dcore_benchmark.json
  results/v3_mo_benchmark.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import j0, j1
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models" / "v3"
MODELS.mkdir(parents=True, exist_ok=True)

ALPHA_R = 2.4048255577  # first zero of J0 -> field vanishes at the rope boundary


def lundquist_profile(t_s: np.ndarray, v: float, b0: float, theta: float, phi: float,
                      t_c: float, radius: float, helicity: float) -> np.ndarray:
    """Return (n, 3) GSM magnetic field of a Lundquist flux rope crossed at speed v.

    Geometry: the spacecraft enters the rope at the boundary (r = 1) at t = 0 and
    travels along a straight chord at speed v.  The closest approach to the rope
    axis happens at t = t_c, so the impact parameter follows from the boundary
    condition  v^2 t_c^2 + y0^2 = R^2 .  `t_c` and `R` are the two free geometric
    parameters; the previous version pinned the entry at the axis (r = 0) and set
    R = v * duration / 2, which forced every rope to be crossed centrally.
    """
    th, ph = np.radians(theta), np.radians(phi)
    axis = np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), np.sin(th)])
    xhat = np.array([1.0, 0.0, 0.0])
    e_perp = xhat - axis * np.dot(xhat, axis)
    n = np.linalg.norm(e_perp)
    if n < 1e-6:  # axis along X -> pick any perpendicular
        e_perp = np.array([0.0, 1.0, 0.0])
    e_perp = e_perp / np.linalg.norm(e_perp)
    e2 = np.cross(axis, e_perp)
    radius = max(radius, 1e3)
    # impact parameter from the entry condition r(0) = 1
    y0 = np.sqrt(max(radius ** 2 - (v * t_c) ** 2, 0.0))
    u = v * (t_s - t_c) * np.dot(xhat, e_perp)
    w = v * (t_s - t_c) * np.dot(xhat, e2)
    r = np.sqrt(u ** 2 + y0 ** 2 + w ** 2) / radius
    r = np.clip(r, 1e-6, 1.0)
    a = ALPHA_R * r
    axial = b0 * j0(a)
    tang = helicity * b0 * j1(a)
    with np.errstate(divide="ignore", invalid="ignore"):
        cu = np.where(r > 1e-6, u / (r * radius), 0.0)
        cw = np.where(r > 1e-6, w / (r * radius), 0.0)
    B = (axial[:, None] * axis[None, :]
         + tang[:, None] * (cu[:, None] * e2[None, :] - cw[:, None] * e_perp[None, :]))
    return B


def fit_flux_rope(t_s: np.ndarray, B: np.ndarray, v: float) -> dict | None:
    """5-parameter fit [B0, theta, phi, t_c, R]; helicity chosen by lower cost."""
    ok = np.isfinite(B).all(axis=1)
    if ok.sum() < 30:
        return None
    t_s, B = t_s[ok], B[ok]
    if len(t_s) > 150:
        stride = int(np.ceil(len(t_s) / 150))
        t_s, B = t_s[::stride], B[::stride]
    duration = float(max(t_s.max() - t_s.min(), 3600.0))
    best = None
    b_scale = max(float(np.abs(B).max()), 1.0)
    lb = [0.1, -90.0, -180.0, -0.5 * duration, 0.05 * v * duration]
    ub = [200.0, 90.0, 180.0, 1.5 * duration, 3.0 * v * duration]

    # Minimum-variance analysis gives the rope axis directly: the eigenvector of
    # the smallest covariance eigenvalue is the direction along which the field
    # barely varies, i.e. the flux-rope axis.  Seeding from it removes the need
    # for a blind orientation grid.
    seeds = []
    try:
        _, evec = np.linalg.eigh(np.cov(B.T))
        axis = evec[:, 0]
        seeds.append((float(np.degrees(np.arcsin(np.clip(axis[2], -1, 1)))),
                      float(np.degrees(np.arctan2(axis[1], axis[0])))))
    except Exception:
        pass
    seeds.append((0.0, 90.0))
    for th0, ph0 in seeds:
        for hel in (1.0, -1.0):
            def resid(p, hel=hel):
                pred = lundquist_profile(t_s, v, p[0], p[1], p[2], p[3], p[4], hel)
                return (pred - B).ravel()

            # central crossing as the starting point: t_c = D/2, R = v*D/2
            p0 = [b_scale, th0, ph0, 0.5 * duration, 0.5 * v * duration]
            try:
                r = least_squares(resid, p0, bounds=(lb, ub), method="trf",
                                  loss="soft_l1", f_scale=2.0, max_nfev=200)
            except Exception:
                continue
            if best is None or r.cost < best[0]:
                best = (float(r.cost), r.x, hel)
    if best is None:
        return None
    cost, p, hel = best
    n_obs = len(B)
    return {
        "b0": float(p[0]), "theta": float(p[1]), "phi": float(p[2]),
        "t_c": float(p[3]), "radius": float(p[4]), "helicity": hel,
        "rmse_fit": float(np.sqrt(2 * cost / (3 * n_obs))),
    }


def evaluate_mc(mc: pd.DataFrame, omni: pd.DataFrame) -> list[dict]:
    out = []
    for _, ev in mc.iterrows():
        t0, t1 = ev["t_icme_start"], ev["t_icme_end"]
        if pd.isna(t0) or pd.isna(t1) or (t1 - t0) < pd.Timedelta(hours=3):
            continue
        w = omni[(omni.datetime >= t0) & (omni.datetime <= t1)]
        if len(w) < 36:
            continue
        # GSM frame: Bz in GSE is rotated by the dipole tilt (up to ~15 nT at
        # equinox), so the flux rope must be fitted and scored in GSM.
        B = w[["bx_gse", "by_gsm", "bz_gsm"]].values
        ok = np.isfinite(B).all(axis=1)
        if ok.sum() < 36:
            continue
        t_s = (w.datetime - t0).dt.total_seconds().values
        v = float(np.nanmedian(w["v_flow"].values))
        if not np.isfinite(v) or v < 100:
            continue
        bz_obs = w["bz_gsm"].values
        duration = float(t_s[-1])
        for frac in (0.20, 0.35, 0.50):
            cut = t_s <= frac * duration
            fit = fit_flux_rope(t_s[cut], B[cut], v)
            if fit is None:
                continue
            t_pred = t_s[~cut]
            if len(t_pred) < 6:
                continue
            pred = lundquist_profile(t_pred, v, fit["b0"], fit["theta"], fit["phi"],
                                     fit["t_c"], fit["radius"], fit["helicity"])
            bz_true_rest = bz_obs[~cut]
            m = np.isfinite(bz_true_rest)
            if m.sum() < 6:
                continue
            bz_seen = bz_obs[cut]
            b_seen = w["b_mag"].values[cut]
            out.append({
                "cme_id": ev["cme_id"], "t_icme_start": str(t0), "frac_observed": frac,
                "t_launch": str(ev["t_launch"]),
                "duration_h": duration / 3600.0,
                "v_flow": v,
                "bz_min_obs_so_far": float(np.nanmin(bz_seen)) if np.isfinite(bz_seen).any() else float("nan"),
                "bz_mean_obs_so_far": float(np.nanmean(bz_seen)) if np.isfinite(bz_seen).any() else float("nan"),
                "bz_max_obs_so_far": float(np.nanmax(bz_seen)) if np.isfinite(bz_seen).any() else float("nan"),
                "b_mean_obs_so_far": float(np.nanmean(b_seen)) if np.isfinite(b_seen).any() else float("nan"),
                "bz_min_whole": float(np.nanmin(bz_obs)) if np.isfinite(bz_obs).any() else float("nan"),
                "n_fit": int(cut.sum()), "n_rest": int(m.sum()),
                "rmse_bz_rest": float(np.sqrt(np.mean((pred[m, 2] - bz_true_rest[m]) ** 2))),
                "mae_bzmin_rest": float(abs(np.min(pred[m, 2]) - np.nanmin(bz_true_rest[m]))),
                "bz_min_pred": float(np.min(pred[m, 2])),
                "bz_min_true": float(np.nanmin(bz_true_rest[m])),
                "corr_bz_rest": float(np.corrcoef(pred[m, 2], bz_true_rest[m])[0, 1])
                if m.sum() > 5 and np.std(pred[m, 2]) > 1e-9 else float("nan"),
                **{k: fit[k] for k in ("b0", "theta", "phi", "t_c", "radius",
                                       "helicity", "rmse_fit")},
            })
    return out


def t_exp(v: np.ndarray) -> np.ndarray:
    """Lopez (1987) expected proton temperature for a given bulk speed (K)."""
    return (0.031 * v - 5.1) ** 2 * 1e3


def build_mo_dataset(omni: pd.DataFrame, cat: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """5-min feature matrix + MO labels from R&C mc intervals."""
    o = omni.copy()
    o = o.sort_values("datetime").reset_index(drop=True)
    o["t_exp"] = t_exp(o["v_flow"].values)
    o["t_ratio"] = o["temp"] / o["t_exp"]
    for w in (6, 12, 36):  # 30 min, 1 h, 3 h in 5-min samples
        o[f"b_std_{w}"] = o["b_mag"].rolling(w, min_periods=w // 2).std()
        o[f"v_mean_{w}"] = o["v_flow"].rolling(w, min_periods=w // 2).mean()
        o[f"pdyn_max_{w}"] = o["pdyn"].rolling(w, min_periods=w // 2).max()
        o[f"beta_mean_{w}"] = o["beta"].rolling(w, min_periods=w // 2).mean()
        o[f"bz_min_{w}"] = o["bz_gsm"].rolling(w, min_periods=w // 2).min()
    o["db_mag"] = o["b_mag"].diff()
    o["dpdyn"] = o["pdyn"].diff()
    # field-rotation smoothness: angle between successive unit vectors
    u = o[["bx_gse", "by_gse", "bz_gse"]].div(o["b_mag"].replace(0, np.nan), axis=0)
    dot = (u * u.shift()).sum(axis=1).clip(-1, 1)
    o["rot_angle"] = np.degrees(np.arccos(dot.values))
    o["rot_var_12"] = o["rot_angle"].rolling(12, min_periods=6).mean()

    feats = ["b_mag", "bz_gsm", "by_gsm", "v_flow", "np_dens", "temp", "pdyn", "beta",
             "t_ratio", "b_std_6", "b_std_12", "b_std_36", "v_mean_6", "v_mean_12",
             "v_mean_36", "pdyn_max_6", "pdyn_max_36", "beta_mean_12", "beta_mean_36",
             "bz_min_6", "bz_min_36", "db_mag", "dpdyn", "rot_angle", "rot_var_12"]

    label = np.zeros(len(o), dtype=int)
    for _, ev in cat[cat["mc_flag"] >= 1].iterrows():
        t0, t1 = ev["t_icme_start"], ev["t_icme_end"]
        if pd.isna(t0) or pd.isna(t1):
            continue
        label[(o.datetime >= t0) & (o.datetime <= t1)] = 1
    return o[["datetime"] + feats], label


def main() -> int:
    omni_path = PROC / "v3_omni_5min.parquet"
    if not omni_path.exists():
        print(f"[3dcore] {omni_path.name} is not part of this package: it is the "
              "5-minute OMNI archive (about 1 GB) and only the hourly context "
              "table is shipped.  Rebuild it with scripts/v3_04_build_highres_omni.py "
              "before running this stage.", file=sys.stderr)
        return 2
    omni = pd.read_parquet(omni_path)
    omni["datetime"] = pd.to_datetime(omni["datetime"])
    cat = pd.read_parquet(PROC / "v3_cme_catalog.parquet")
    cat["t_icme_start"] = pd.to_datetime(cat["t_icme_start"])
    cat["t_icme_end"] = pd.to_datetime(cat["t_icme_end"])
    cat["t_launch"] = pd.to_datetime(cat["t_launch"])
    print(f"[3dcore] omni5 rows={len(omni)}  MCs={int((cat.mc_flag>=1).sum())}")

    # ---------- MO-Net -----------------------------------------------------
    X, y = build_mo_dataset(omni, cat)
    tr = (X.datetime < pd.Timestamp("2020-01-01")).values
    te = (X.datetime >= pd.Timestamp("2020-01-01")).values
    import lightgbm as lgb
    cols = [c for c in X.columns if c != "datetime"]
    Xv = X[cols].replace([np.inf, -np.inf], np.nan)
    clf = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                             min_child_samples=60, subsample=0.9, colsample_bytree=0.8,
                             reg_lambda=1.0, class_weight="balanced", random_state=3,
                             n_jobs=1, verbose=-1)
    clf.fit(Xv[tr], y[tr])
    p_te = clf.predict_proba(Xv[te])[:, 1]
    mo_res = {
        "train": {"n": int(tr.sum()), "pos": int(y[tr].sum())},
        "test_cycle25": {"n": int(te.sum()), "pos": int(y[te].sum()),
                         "roc_auc": float(roc_auc_score(y[te], p_te)) if y[te].sum() else None,
                         "pr_auc": float(average_precision_score(y[te], p_te)) if y[te].sum() else None},
        "features": cols,
    }
    joblib.dump({"model": clf, "features": cols}, MODELS / "mo_net.joblib")
    print(f"[MO-Net] test ROC-AUC={mo_res['test_cycle25']['roc_auc']} "
          f"PR-AUC={mo_res['test_cycle25']['pr_auc']} pos={mo_res['test_cycle25']['pos']}")
    (RESULTS / "v3_mo_benchmark.json").write_text(json.dumps(mo_res, indent=2))

    # ---------- 3DCORE-lite: fit every MC, calibrate on Cycles 23-24 --------
    mc = cat[cat.mc_flag >= 1].drop_duplicates("cme_id")
    print(f"[3dcore] MCs to fit: {len(mc)}")
    rows = evaluate_mc(mc, omni)
    df = pd.DataFrame(rows)
    if df.empty:
        print("[3dcore] no evaluable MC")
        return 1
    df["t_launch"] = pd.to_datetime(df["t_launch"])
    df.to_parquet(RESULTS / "v3_3dcore_predictions.parquet", index=False)
    print(f"[3dcore] fits={len(df)}  MCs={df.cme_id.nunique()}")

    calib_feats = ["b0", "theta", "phi", "t_c", "radius", "helicity", "rmse_fit",
                   "v_flow", "duration_h", "frac_observed",
                   "bz_min_obs_so_far", "bz_mean_obs_so_far", "bz_max_obs_so_far",
                   "b_mean_obs_so_far", "bz_min_pred"]
    tr = df[df["t_launch"] < pd.Timestamp("2020-01-01")].copy()
    te = df[df["t_launch"] >= pd.Timestamp("2020-01-01")].copy()
    calib = {}
    if len(tr) >= 60 and len(te) >= 20:
        import lightgbm as lgb
        reg = lgb.LGBMRegressor(n_estimators=350, learning_rate=0.04, num_leaves=15,
                                max_depth=4, min_child_samples=20, subsample=0.85,
                                colsample_bytree=0.8, reg_lambda=2.0, random_state=13,
                                n_jobs=1, verbose=-1, objective="huber")
        reg.fit(tr[calib_feats], tr["bz_min_true"].values)
        te["bz_min_calibrated"] = reg.predict(te[calib_feats])
        # The calibrator is scored inside this script and its numbers are written
        # to results/v3_3dcore_benchmark.json.  It is not persisted: nothing in
        # predict.py or in the four-cell table loads it.
        calib = {"n_train": int(len(tr)), "n_test": int(len(te))}
        print(f"[3dcore] amplitude calibrator: train={len(tr)} test={len(te)}")

    summary = {}
    for frac, g in te.groupby("frac_observed"):
        persist_mae = float((g["bz_min_obs_so_far"] - g["bz_min_true"]).abs().median())
        model_mae = float(g["mae_bzmin_rest"].median())
        block = {
            "n": int(len(g)),
            "rmse_bz_rest": float(g["rmse_bz_rest"].median()),
            "mae_bzmin_rest": model_mae,
            "mae_bzmin_persistence": persist_mae,
            "bzmin_skill_vs_persistence": float(1.0 - model_mae / max(persist_mae, 1e-9)),
            "mean_corr_bz_rest": float(g["corr_bz_rest"].mean()),
            "fit_rmse_median": float(g["rmse_fit"].median()),
        }
        if "bz_min_calibrated" in g:
            c_mae = float((g["bz_min_calibrated"] - g["bz_min_true"]).abs().median())
            block["mae_bzmin_calibrated"] = c_mae
            block["bzmin_skill_calibrated"] = float(1.0 - c_mae / max(persist_mae, 1e-9))
        summary[f"frac_{int(frac*100)}"] = block
    res = {"n_mc_total": int(df.cme_id.nunique()), "n_fits": int(len(df)),
           "calibrator": calib, "by_fraction": summary}
    (RESULTS / "v3_3dcore_benchmark.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
