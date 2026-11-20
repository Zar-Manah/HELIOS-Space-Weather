#!/usr/bin/env python3
"""HELIOS v3.4 - Stage 5: HELIOS-Tactical v3 ring-current engine.

Physics backbone: Burton-O'Brien-McPherron (2000) ring-current ODE rolled
forward causally under a persisted/exponentially-decaying L1 driver:

    Dst* = Dst - b*sqrt(Pdyn) + c
    dDst*/dt = Q(Ey) - Dst*/tau(Ey)
    Q(Ey)    = -4.4*(Ey - 0.49)  for Ey >= 0.49 mV/m
    tau(Ey)  = 2.40*exp(9.74/(4.69 + Ey))  h

The GBDT learns the residual Dst(t+h) - Dst_ODE(t+h), so the ODE carries the
bulk dynamics and the trees correct the systematic bias.  Adaptive conformal
inference (Gibbs & Candes) then produces intervals whose width tracks the
current regime instead of a single static quantile.

Outputs
  models/v3/tactical_h{h}.joblib
  results/v3_tactical_benchmark.json
  results/v3_tactical_test_predictions.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models" / "v3"
MODELS.mkdir(parents=True, exist_ok=True)

HORIZONS = (1, 2, 3, 6, 12)
B_PDYN = 7.26   # nT / nPa^0.5   (O'Brien & McPherron 2000)
C_OFF = 11.0    # nT


def q_injection(ey: np.ndarray) -> np.ndarray:
    return np.where(ey >= 0.49, -4.4 * (ey - 0.49), 0.0)


def tau_decay(ey: np.ndarray) -> np.ndarray:
    return 2.40 * np.exp(9.74 / (4.69 + np.clip(ey, 0.0, None)))


def ode_rollout(dst: float, pdyn: float, ey: float, horizon: int,
                driver_decay_h: float = 3.0) -> float:
    """Integrate the ring-current ODE h hours ahead with a decaying driver."""
    if not np.isfinite(dst):
        return np.nan
    p = pdyn if np.isfinite(pdyn) else 1.0
    e = ey if np.isfinite(ey) else 0.0
    state = dst - B_PDYN * np.sqrt(max(p, 0.0)) + C_OFF
    for k in range(horizon):
        decay = np.exp(-k / driver_decay_h)
        p_k = max(p * decay + 1.0 * (1 - decay), 0.0)
        e_k = max(e * decay, 0.0)
        dst_star = dst - B_PDYN * np.sqrt(p_k) + C_OFF
        q = float(q_injection(np.array([e_k]))[0])
        tau = float(tau_decay(np.array([e_k]))[0])
        dstar = dst_star + (q - dst_star / tau) * 1.0
        dst = dstar + B_PDYN * np.sqrt(p_k) - C_OFF
    return dst


def build_features(h: pd.DataFrame) -> pd.DataFrame:
    h = h.sort_values("datetime").reset_index(drop=True)
    b_t = np.sqrt(h["by_gsm"] ** 2 + h["bz_gsm"] ** 2)
    clock = np.arctan2(h["by_gsm"], h["bz_gsm"])
    h["newell"] = 1.6e-3 * (h["v_flow"].clip(lower=0) ** (4 / 3)) * (b_t ** (2 / 3)) * \
        (np.sin(np.abs(clock) / 2.0).clip(lower=0) ** (8 / 3))
    h["ey_rect"] = h["e_field"].clip(lower=0)
    feats = {}
    for w in (1, 2, 3, 6, 12, 24):
        r = h.rolling(w, min_periods=1)
        for c in ("dst", "bz_gsm", "by_gsm", "b_mag", "v_flow", "pdyn", "ey_rect", "newell"):
            feats[f"{c}_mean_{w}h"] = r[c].mean()
        feats[f"dst_min_{w}h"] = r["dst"].min()
        feats[f"bz_min_{w}h"] = r["bz_gsm"].min()
        feats[f"newell_sum_{w}h"] = r["newell"].sum()
        feats[f"pdyn_max_{w}h"] = r["pdyn"].max()
        feats[f"dst_std_{w}h"] = r["dst"].std()
        feats[f"bz_frac_south_{w}h"] = (h["bz_gsm"] < -5).rolling(w, min_periods=1).mean()
    h = pd.concat([h, pd.DataFrame(feats, index=h.index)], axis=1)
    h = h.loc[:, ~h.columns.duplicated(keep="last")]
    h["dst_slope_1h"] = h["dst"].diff()
    h["dst_slope_3h"] = h["dst"].diff(3) / 3.0
    h["bz_slope_1h"] = h["bz_gsm"].diff()
    h["pdyn_slope_1h"] = h["pdyn"].diff()
    h["dpdyn_max_3h"] = h["pdyn"].diff().rolling(3, min_periods=1).max()
    h["hours_since_dpdyn_spike"] = _hours_since(h["pdyn"].diff().fillna(0) > 2.0)
    h["southward"] = (h["bz_gsm"] < 0).astype(float)
    h["dst_star"] = h["dst"] - B_PDYN * np.sqrt(h["pdyn"].clip(lower=0)) + C_OFF
    return h


def _hours_since(flag: pd.Series) -> pd.Series:
    idx = np.arange(len(flag))
    last = pd.Series(np.where(flag.values, idx, np.nan)).ffill().values
    return pd.Series(idx - last, index=flag.index).clip(upper=72)


# NOTE: dst_std_1h is deliberately absent: a 1-sample rolling std is NaN by
# construction and can never split.  dst_std_6h is the ACI volatility scale.
FEATURES = [
    "dst", "bz_gsm", "by_gsm", "b_mag", "v_flow", "np_dens", "pdyn", "ey_rect", "newell",
    "dst_star", "kp", "southward", "dst_slope_1h", "dst_slope_3h", "bz_slope_1h",
    "pdyn_slope_1h", "dpdyn_max_3h", "hours_since_dpdyn_spike",
] + [f"{c}_mean_{w}h" for w in (1, 2, 3, 6, 12, 24)
     for c in ("dst", "bz_gsm", "by_gsm", "b_mag", "v_flow", "pdyn", "ey_rect", "newell")] \
  + [f"dst_min_{w}h" for w in (1, 2, 3, 6, 12, 24)] \
  + [f"bz_min_{w}h" for w in (1, 2, 3, 6, 12, 24)] \
  + [f"newell_sum_{w}h" for w in (1, 2, 3, 6, 12, 24)] \
  + [f"pdyn_max_{w}h" for w in (1, 2, 3, 6, 12, 24)] \
  + [f"dst_std_{w}h" for w in (3, 6, 12)] \
  + [f"bz_frac_south_{w}h" for w in (1, 3, 6)] \
  + ["dst_ode_1", "dst_ode_2", "dst_ode_3", "dst_ode_6", "dst_ode_12",
     "ode_imbalance_1"]


def add_ode_columns(h: pd.DataFrame) -> pd.DataFrame:
    """Vectorised ring-current ODE rollout for every horizon in a single pass."""
    dst = h["dst"].values.astype(float)
    pdyn = h["pdyn"].values.astype(float)
    ey = h["ey_rect"].values.astype(float)
    p0 = np.where(np.isfinite(pdyn), np.clip(pdyn, 0.0, None), 1.0)
    e0 = np.where(np.isfinite(ey), np.clip(ey, 0.0, None), 0.0)
    state = dst.copy()
    out = {}
    for k in range(max(HORIZONS)):
        decay = np.exp(-k / 3.0)
        p_k = np.clip(p0 * decay + 1.0 * (1 - decay), 0.0, None)
        e_k = np.clip(e0 * decay, 0.0, None)
        bsp = B_PDYN * np.sqrt(p_k)
        dst_star = state - bsp + C_OFF
        q = np.where(e_k >= 0.49, -4.4 * (e_k - 0.49), 0.0)
        tau = 2.40 * np.exp(9.74 / (4.69 + e_k))
        state = dst_star + (q - dst_star / tau) + bsp - C_OFF
        if (k + 1) in HORIZONS:
            out[f"dst_ode_{k + 1}"] = state.copy()
    out_df = pd.DataFrame(out, index=h.index)
    h = pd.concat([h.loc[:, ~h.columns.isin(out_df.columns)], out_df], axis=1)
    h["ode_imbalance_1"] = h["dst_ode_1"] - h["dst"]
    return h


def rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else float("nan")


def mae(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[m] - b[m]))) if m.any() else float("nan")


def tss(y, pred, thr):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    m = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[m] <= thr, pred[m] <= thr
    tp = int((y & pred).sum())
    fn = int((y & ~pred).sum())
    fp = int((~y & pred).sum())
    tn = int((~y & ~pred).sum())
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    return float(tpr - fpr), {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "tpr": tpr, "fpr": fpr}


def make_members(n_estimators: int = 600) -> list:
    """The five-member bag, used identically inside CV and for serving.

    v3.2 built two members with 400 trees inside the rolling-origin CV and five
    members with 600 trees for serving, so the blend weight, tail gain, ACI mode
    and initial quantiles were all selected on a *different* predictor than the
    one shipped.  One factory removes that mismatch: the CV fold and the served
    model now differ only in the data they see, never in the model family.
    """
    import lightgbm as lgb
    import xgboost as xgb
    return [
        lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=0.03, num_leaves=31,
                          max_depth=6, min_child_samples=40, subsample=0.85,
                          colsample_bytree=0.8, reg_lambda=2.0, random_state=5,
                          n_jobs=1, verbose=-1, objective="regression"),
        lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=0.03, num_leaves=31,
                          max_depth=6, min_child_samples=40, subsample=0.85,
                          colsample_bytree=0.8, reg_lambda=2.0, random_state=105,
                          n_jobs=1, verbose=-1, objective="regression"),
        xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=0.03, max_depth=6,
                         min_child_weight=10, subsample=0.85, colsample_bytree=0.8,
                         reg_lambda=2.0, random_state=5, n_jobs=1, tree_method="hist",
                         objective="reg:squarederror"),
        xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=0.03, max_depth=6,
                         min_child_weight=10, subsample=0.85, colsample_bytree=0.8,
                         reg_lambda=2.0, random_state=205, n_jobs=1, tree_method="hist",
                         objective="reg:squarederror"),
        lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=0.03, num_leaves=31,
                          max_depth=6, min_child_samples=40, subsample=0.85,
                          colsample_bytree=0.8, reg_lambda=2.0, random_state=17,
                          n_jobs=1, verbose=-1, objective="huber", alpha=20.0),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", default="2020-01-01")
    ap.add_argument("--tag", default="v3")
    args = ap.parse_args()
    train_end = pd.Timestamp(args.train_end)

    h = pd.read_parquet(PROC / "helios_omni_features_1998_2026.parquet")
    h["datetime"] = pd.to_datetime(h["datetime"])
    h = h.sort_values("datetime").reset_index(drop=True)
    print(f"[tactical] hourly rows={len(h)} {h.datetime.min()} .. {h.datetime.max()}")
    h = build_features(h)
    h = add_ode_columns(h)
    print("[tactical] ODE columns built")

    import lightgbm as lgb
    import xgboost as xgb

    results = {"train_end": str(train_end), "horizons": {}}
    store = []
    for hor in HORIZONS:
        h[f"target_{hor}"] = h["dst"].shift(-hor)
        h[f"ode_err_{hor}"] = h[f"dst_ode_{hor}"] - h[f"target_{hor}"]
        h[f"persist_err_{hor}"] = h["dst"] - h[f"target_{hor}"]

    tr = h[h.datetime < train_end]
    te = h[h.datetime >= train_end]
    print(f"[tactical] train={len(tr)} test={len(te)}")

    for hor in HORIZONS:
        y = h[f"target_{hor}"]
        resid = y - h[f"dst_ode_{hor}"]
        m_tr = (h.datetime < train_end) & resid.notna()
        m_te = (h.datetime >= train_end) & resid.notna()
        Xtr = h.loc[m_tr, FEATURES].replace([np.inf, -np.inf], np.nan)
        Xte = h.loc[m_te, FEATURES].replace([np.inf, -np.inf], np.nan)
        ytr = resid[m_tr].values
        yte = resid[m_te].values
        # storm-aware weighting.  The weight must be causal: it uses the *current*
        # storm state (dst_min_6h = min over the last 6 h), never the future.
        # v3.0 used dst.rolling(6).min().shift(-6), i.e. it leaked the target
        # into the training weights.
        w = np.ones(int(m_tr.sum()))
        w += 4.0 * (h.loc[m_tr, "dst_min_6h"].values <= -50)
        w += 10.0 * (h.loc[m_tr, "dst_min_6h"].values <= -100)

        # 5-member bag: two L2 LightGBM seeds, two XGBoost seeds, one Huber
        # LightGBM.  Averaging independent seeds is variance reduction, decided
        # a priori; the member families were fixed before looking at Cycle 25.
        # The *same* factory is used inside CV below, so every CV decision is
        # made on the predictor that is actually served.
        members = make_members(600)
        # ---- pre-2020 rolling-origin CV: persistence blend + ACI variant ----
        # The last 15 % of the pre-2020 window is held out *inside* training;
        # Cycle 25 is never touched when these two choices are made.
        cut = tr["datetime"].quantile(0.85)
        m_cv_tr = (h["datetime"] < cut) & resid.notna()
        m_cv_va = (h["datetime"] >= cut) & (h["datetime"] < train_end) & resid.notna()
        w_cv = np.ones(int(m_cv_tr.sum()))
        w_cv += 4.0 * (h.loc[m_cv_tr, "dst_min_6h"].values <= -50)
        w_cv += 10.0 * (h.loc[m_cv_tr, "dst_min_6h"].values <= -100)
        cv_members = make_members(600)
        Xc_tr = h.loc[m_cv_tr, FEATURES].replace([np.inf, -np.inf], np.nan)
        Xc_va = h.loc[m_cv_va, FEATURES].replace([np.inf, -np.inf], np.nan)
        for m in cv_members:
            m.fit(Xc_tr, resid[m_cv_tr].values, sample_weight=w_cv)
        p_cv = np.mean([m.predict(Xc_va) for m in cv_members], axis=0)
        pred_cv = h.loc[m_cv_va, f"dst_ode_{hor}"].values + p_cv
        truth_cv = y[m_cv_va].values
        persist_cv = h.loc[m_cv_va, "dst"].values
        blend_scores = {}
        for wb in (0.0, 0.25, 0.5, 0.75, 1.0):
            blend_scores[wb] = rmse(wb * pred_cv + (1 - wb) * persist_cv, truth_cv)
        w_persist = min(blend_scores, key=blend_scores.get)
        pred_cv_b = w_persist * pred_cv + (1 - w_persist) * persist_cv

        # Tail calibration (storm deepening).  A squared-loss ensemble regresses
        # the tail towards the mean: at +12 h it under-predicts the deepest
        # hours and loses TSS to persistence even with a lower RMSE.  One gain
        # on the storm side of a knee is fitted on the pre-2020 CV fold only,
        # and accepted only if the CV RMSE does not degrade more than 2 %.
        def _tail(pv, k, knee=-30.0):
            pv = np.asarray(pv, float)
            return np.where(pv < knee, knee + k * (pv - knee), pv)

        cv_rmse0 = rmse(pred_cv_b, truth_cv)
        cv_tss0 = tss(truth_cv, pred_cv_b, -50)[0]
        gain_cv = {}
        for k in np.round(np.arange(1.0, 1.65, 0.05), 2):
            q = _tail(pred_cv_b, float(k))
            gain_cv[float(k)] = (rmse(q, truth_cv), tss(truth_cv, q, -50)[0])
        # The gain is only for horizons where the model *loses* to persistence
        # on the CV fold: there is no reason to trade point accuracy for storm
        # depth when the model already wins.  Where it loses, the gain must
        # reach CV persistence TSS with at most 1 % CV RMSE, otherwise 1.0.
        cv_tss_persist = tss(truth_cv, persist_cv, -50)[0]
        cv_rmse_persist = rmse(persist_cv, truth_cv)
        tail_gain = 1.0
        if cv_tss0 < cv_tss_persist:
            # The horizons with a frozen point-RMSE gate (+1 h, +6 h) may spend
            # at most 1 % of CV RMSE; the rest may spend 5 % but must still beat
            # persistence in CV RMSE.  Never trade away a gated metric.
            cap = cv_rmse0 * (1.01 if hor in (1, 6) else 1.05)
            cap = min(cap, cv_rmse_persist) if hor not in (1, 6) else cap
            eligible = {k: v for k, v in gain_cv.items()
                        if v[0] <= cap and v[1] >= cv_tss_persist + 0.005}
            if eligible:
                tail_gain = max(eligible, key=lambda k: (eligible[k][1], -eligible[k][0]))

        # ACI variant: raw vs normalised by the causal 6 h Dst volatility.
        dst_cv_now = persist_cv
        reg_cv = np.where(dst_cv_now > -30, 0, np.where(dst_cv_now > -60, 1, 2))
        std_cv = h.loc[m_cv_va, "dst_std_6h"].values
        ref_std = float(np.nanmedian(h.loc[m_tr, "dst_std_6h"].values))
        ref_std = ref_std if np.isfinite(ref_std) and ref_std > 1e-6 else 1.0
        scale_cv = np.clip(std_cv / ref_std, 0.5, 3.0)
        sc_cv = np.abs(pred_cv_b - truth_cv)

        def simulate(scores, scale, reg, q_init, gamma_mult=0.02):
            qs = dict(q_init)
            gm = {r: gamma_mult * max(qs[r], 1.0) for r in qs}
            cov, wid = [], []
            for i in range(len(scores)):
                r = int(reg[i])
                q = qs[r]
                ok = scores[i] <= q
                cov.append(ok)
                wid.append(2 * q * scale[i])
                qs[r] = max(0.0, q + gm[r] * ((0.0 if ok else 1.0) - 0.10))
            cov = np.asarray(cov, bool)
            wid = np.asarray(wid, float)
            out = {"coverage": float(np.mean(cov)), "width": float(np.mean(wid))}
            for thr in (50, 100):
                sel = truth_cv <= -thr
                out[f"storm{thr}"] = float(np.mean(cov[sel])) if sel.sum() >= 20 else None
            return out

        def _regime_quantiles(scores: np.ndarray, reg: np.ndarray,
                              alpha: float = 0.10) -> dict:
            """Per-regime initial quantile from *CV ensemble* residuals only."""
            q = {}
            finite = np.isfinite(scores)
            for r in (0, 1, 2):
                sel = (reg == r) & finite
                pool = scores[sel] if sel.sum() >= 50 else scores[finite]
                q[r] = float(np.quantile(pool, 1 - alpha))
            return q

        # The ACI score is |ensemble - truth| / scale, so its initial quantile
        # must be measured on exactly that quantity in the pre-2020 CV fold.
        q_init_cv_raw = _regime_quantiles(sc_cv, reg_cv)
        q_init_cv_norm = _regime_quantiles(sc_cv / scale_cv, reg_cv)
        cv_raw = simulate(sc_cv, np.ones_like(sc_cv), reg_cv, q_init_cv_raw)
        cv_norm = simulate(sc_cv / scale_cv, scale_cv, reg_cv, q_init_cv_norm)

        def aci_loss(d):
            errs = [abs((d.get(f"storm{t}") if d.get(f"storm{t}") is not None else 0.90) - 0.90)
                    for t in (50, 100)]
            return float(np.mean(errs) + 0.002 * d["width"])

        aci_mode = "normalized" if aci_loss(cv_norm) < aci_loss(cv_raw) else "raw"
        print(f"[+{hor}h] CV blend w_persist={w_persist} "
              f"({ {k: round(v, 2) for k, v in blend_scores.items()} }) | "
              f"ACI raw={cv_raw} norm={cv_norm} -> {aci_mode} | "
              f"tail_gain={tail_gain} (CV TSS50 {cv_tss0:.3f} vs persist "
              f"{cv_tss_persist:.3f} -> {gain_cv[tail_gain][1]:.3f})", flush=True)

        for m in members:
            m.fit(Xtr, ytr, sample_weight=w)
        p_res = np.mean([m.predict(Xte) for m in members], axis=0)
        pred = h.loc[m_te, f"dst_ode_{hor}"].values + p_res
        truth = y[m_te].values
        persist_te = h.loc[m_te, "dst"].values
        pred = w_persist * pred + (1 - w_persist) * persist_te
        pred = _tail(pred, tail_gain)

        block = {
            "n_test": int(m_te.sum()),
            "rmse_model": rmse(pred, truth),
            "mae_model": mae(pred, truth),
            "rmse_persistence": rmse(h.loc[m_te, "dst"].values, truth),
            "rmse_ode": rmse(h.loc[m_te, f"dst_ode_{hor}"].values, truth),
            "mae_ode": mae(h.loc[m_te, f"dst_ode_{hor}"].values, truth),
        }
        # storm-only skill
        for thr in (-50, -100):
            storm = truth <= thr
            if storm.sum() >= 20:
                block[f"storm{abs(thr)}_n"] = int(storm.sum())
                block[f"storm{abs(thr)}_rmse_model"] = rmse(pred[storm], truth[storm])
                block[f"storm{abs(thr)}_rmse_persistence"] = rmse(
                    h.loc[m_te, "dst"].values[storm], truth[storm])
                block[f"storm{abs(thr)}_rmse_ode"] = rmse(
                    h.loc[m_te, f"dst_ode_{hor}"].values[storm], truth[storm])
                s_model, _ = tss(truth, pred, thr)
                s_persist, _ = tss(truth, h.loc[m_te, "dst"].values, thr)
                s_ode, _ = tss(truth, h.loc[m_te, f"dst_ode_{hor}"].values, thr)
                block[f"tss{abs(thr)}_model"] = s_model
                block[f"tss{abs(thr)}_persistence"] = s_persist
                block[f"tss{abs(thr)}_ode"] = s_ode

        # ---- adaptive conformal inference, by ring-current regime -----------
        # The state is causal (current Dst), so the regime is known at forecast
        # time.  Each regime keeps its own adaptive quantile, which is what makes
        # the band honest during storms instead of averaging them with quiet
        # hours (v3.0: overall 0.899 but only 0.683 on storm hours at +1h).
        def regime_of(dst_now: np.ndarray) -> np.ndarray:
            return np.where(dst_now > -30, 0, np.where(dst_now > -60, 1, 2))

        alpha = 0.10
        te_reg = regime_of(h.loc[m_te, "dst"].values)
        std_te = h.loc[m_te, "dst_std_6h"].values
        scale_te = (np.clip(std_te / ref_std, 0.5, 3.0) if aci_mode == "normalized"
                    else np.ones(len(std_te)))
        # `aci_q0` is the *initial* state that reaches serving: it comes from the
        # pre-2020 CV residuals of the served 5-member ensemble (never from the
        # raw ODE residuals, which describe a different predictor, and never
        # from test labels).  The adaptive recursion starts exactly there.
        q0 = dict(q_init_cv_norm if aci_mode == "normalized" else q_init_cv_raw)
        q_state = dict(q0)
        gamma = {r: 0.02 * max(q_state[r], 1.0) for r in (0, 1, 2)}
        scores = np.abs(pred - truth) / scale_te
        covered, widths = [], []
        for i in range(len(scores)):
            r = int(te_reg[i])
            q = q_state[r]
            si = scores[i]
            ok = bool(np.isfinite(si) and si <= q)
            covered.append(ok)
            widths.append(2 * q * scale_te[i] if np.isfinite(scale_te[i]) else np.nan)
            # a non-finite score is an uncovered observation, never a silent pass
            q_state[r] = max(0.0, q + gamma[r] * ((0.0 if ok else 1.0) - alpha))
        covered = np.asarray(covered, bool)
        block["aci_coverage"] = float(np.mean(covered))
        block["aci_width_mean"] = float(np.nanmean(widths))
        block["aci_coverage_storm50"] = (
            float(np.mean(covered[truth <= -50])) if (truth <= -50).sum() >= 20 else None)
        block["aci_coverage_storm100"] = (
            float(np.mean(covered[truth <= -100])) if (truth <= -100).sum() >= 20 else None)
        block["aci_regime_counts"] = {str(r): int((te_reg == r).sum()) for r in (0, 1, 2)}
        # Static reference quantile fitted on the pre-2020 CV fold only.  v3.2
        # took it from the test scores, which made the published width a
        # post-hoc number; it is now a pre-registered quantity.
        sc_cv_used = sc_cv / scale_cv if aci_mode == "normalized" else sc_cv
        sc_cv_used = sc_cv_used[np.isfinite(sc_cv_used)]
        static_q = float(np.quantile(sc_cv_used, 1 - alpha))
        block["static_q"] = static_q
        block["static_q_source"] = "pre2020_cv_ensemble"
        block["static_coverage"] = float(np.mean(scores <= static_q))
        block["static_width"] = float(np.mean(2 * static_q * scale_te))
        block["q_init_cv"] = {str(k): float(v) for k, v in q0.items()}
        block["q_init_cv_raw"] = {str(k): float(v) for k, v in q_init_cv_raw.items()}
        block["q_init_cv_norm"] = {str(k): float(v) for k, v in q_init_cv_norm.items()}

        results["horizons"][f"+{hor}h"] = block
        print(f"[+{hor}h] RMSE model={block['rmse_model']:.3f} "
              f"persist={block['rmse_persistence']:.3f} ode={block['rmse_ode']:.3f} "
              f"| storm50 RMSE={block.get('storm50_rmse_model', float('nan')):.3f} "
              f"ACI cov={block['aci_coverage']:.3f} storm50={block['aci_coverage_storm50']} "
              f"w={block['aci_width_mean']:.2f}")

        block["aci_mode"] = aci_mode
        block["aci_q0_source"] = "pre2020_cv_ensemble"
        block["blend_persistence_weight"] = float(w_persist)
        block["tail_gain"] = float(tail_gain)
        block["cv_tss50"] = float(cv_tss0)
        block["cv_tss50_gain"] = float(gain_cv[tail_gain][1])
        block["cv_blend_rmse"] = {str(k): v for k, v in blend_scores.items()}
        block["cv_aci"] = {"raw": cv_raw, "normalized": cv_norm,
                           "chosen": aci_mode}
        joblib.dump({"members": members, "features": FEATURES, "horizon": hor,
                     "aci_q0": q0, "aci_q_final": q_state,
                     "aci_q0_source": "pre2020_cv_ensemble",
                     "static_q": static_q, "static_q_source": "pre2020_cv_ensemble",
                     "aci_mode": aci_mode, "scale_ref": ref_std,
                     "blend_persistence_weight": float(w_persist),
                     "tail_gain": float(tail_gain),
                     "regime_edges": (-30.0, -60.0)},
                    MODELS / f"tactical_h{hor}.joblib")
        part = h.loc[m_te, ["datetime", "dst"]].copy()
        part["horizon"] = hor
        part["truth"] = truth
        part["pred"] = pred
        part["ode"] = h.loc[m_te, f"dst_ode_{hor}"].values
        part["persistence"] = h.loc[m_te, "dst"].values
        part["covered_aci"] = covered
        part["aci_width"] = widths
        part["regime"] = te_reg
        store.append(part)

    pd.concat(store, ignore_index=True).to_parquet(
        RESULTS / "v3_tactical_test_predictions.parquet", index=False)
    (RESULTS / f"v3_tactical_benchmark{'' if args.tag == 'v3' else '_' + args.tag}.json").write_text(
        json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
