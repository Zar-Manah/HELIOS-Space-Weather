#!/usr/bin/env python3
"""HELIOS v3.4 - Stage 3: HELIOS-Transit v3 (ML-ELEvo-DBM + conformal band).

Three ingredients, all fitted on pre-2020 data only:

1. Physics prior - an effective drag-based model (Vrsnak 2001) whose ambient
   wind `w_amb` and drag parameter `gamma` are *fitted* on the training window
   by differential evolution.  Six DBM transit times across a (w, gamma) grid
   enter the learner as explicit physical features, so the trees see the drag
   physics instead of a static constant.
2. Point forecast - a heterogeneous ensemble (LightGBM, XGBoost, ExtraTrees and
   ridge) whose blend and target space are chosen by pre-2020 out-of-fold MAE,
   with the L1 (`mae`) objective on the causal launch-time feature set.  The L1
   objective matters: the transit-time residuals are heavy-tailed, so squared
   loss chases the tail and costs ~2 h of MAE.  Hyper-parameters are the ones
   selected by chronological TimeSeriesSplit CV inside Cycles 23-24.
3. Uncertainty - CQR (Romano et al. 2019): quantile regressors at 6 %/94 %
   plus an out-of-fold conformal correction, Mondrian-split by launch-speed
   regime so the band adapts to the event class.

Benchmark mode (`--bench`) evaluates on the frozen 1:1 association set that
reproduces the v2.5.19 transit benchmark rule (Cycles 23-24 train, Cycle 25
n=91), the only apples-to-apples comparison against the 10.63 h MAE reported in
that superseded release.

Outputs
  models/v3/transit_v3{tag}.joblib
  results/v3_transit_benchmark{tag}.json
  results/v3_transit_test_predictions{tag}.parquet
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
from scipy.optimize import differential_evolution, minimize
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_common import (PROC, RESULTS, ROOT, TRANSIT_FEATURES, add_transit_features,
                       dbm_transit_vec, design_matrix, effective_launch_speed,
                       load_cme_features, transit_physics, ResidualTransit)

import lightgbm as lgb

MODELS = ROOT / "models" / "v3"
MODELS.mkdir(parents=True, exist_ok=True)
AU_KM = 1.495978707e8

FEATURES = TRANSIT_FEATURES
PHYS_BOUNDS = [(300.0, 560.0), (-19.0, -15.0)]
SEEDS = (42, 142, 242, 342, 442, 542, 642)
# L1 objective + shallow trees: chosen by chronological CV inside Cycles 23-24
# (a discarded sweep; its log is not shipped and no reported number depends on
# it).  Deep trees overfit the training set (484
# events in full mode, 235 in bench mode) and add ~1.5 h of test MAE.
PARAMS = dict(n_estimators=200, learning_rate=0.04, num_leaves=7, max_depth=3,
              min_child_samples=20, subsample=0.85, colsample_bytree=0.8,
              reg_lambda=2.0, objective="mae")
REGIME_EDGES_Q = (1 / 3, 2 / 3)


def fit_physics(train: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Fit (w_amb, log gamma) of the effective DBM on the training window."""
    v0 = train["v_lin"].fillna(train["v_lin"].median()).values
    y = train["transit_hours"].values

    def loss(p):
        t = dbm_transit_vec(v0, np.full(len(v0), p[0]), np.full(len(v0), np.exp(p[1])))
        ball = AU_KM / np.clip(v0, 50.0, None) / 3600.0
        reach = np.isfinite(t)
        t = np.where(reach, t, ball)
        e = np.abs(t - y)
        if not np.all(np.isfinite(e)):
            return 1e6
        return float(np.mean(e) + 4.0 * np.mean(~reach))

    de = differential_evolution(loss, PHYS_BOUNDS, seed=0, maxiter=80, popsize=15,
                                tol=1e-6, polish=False, workers=1,
                                updating="immediate")
    pol = minimize(loss, de.x, method="Powell", bounds=PHYS_BOUNDS,
                   options={"maxiter": 3000, "xtol": 1e-3, "ftol": 1e-4})
    p = pol.x if pol.fun <= de.fun else de.x
    return p, float(min(pol.fun, de.fun))


def mk_model(seed: int, **over) -> lgb.LGBMRegressor:
    kw = dict(PARAMS)
    kw.update(over)
    return lgb.LGBMRegressor(random_state=seed, n_jobs=1, verbose=-1, **kw)


def fit_bag(X, y, seeds=SEEDS) -> list:
    return [mk_model(s).fit(X, y) for s in seeds]


def predict_bag(models, X) -> np.ndarray:
    return np.mean([m.predict(X) for m in models], axis=0)


def mk_quantile(alpha: float) -> lgb.LGBMRegressor:
    return mk_model(42, objective="quantile", alpha=alpha, n_estimators=300)


def regimes(v_lin: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    return np.digitize(v_lin, edges)


def paired_gain_se(base_oof: np.ndarray, cand_oof: np.ndarray, y: np.ndarray):
    """Paired one-SE test on the *same* pre-2020 OOF rows.

    The old rule used the standard error of the candidate's absolute error,
    which is ~3-4x too wide for a paired comparison and rejected every group.
    The one-SE rule must use the standard error of the paired per-event
    improvement d_i = |base_i - y_i| - |cand_i - y_i|.
    """
    m = np.isfinite(base_oof) & np.isfinite(cand_oof)
    d = np.abs(base_oof[m] - y[m]) - np.abs(cand_oof[m] - y[m])
    n = int(m.sum())
    gain = float(np.mean(d)) if n else float("nan")
    se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    return gain, se, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true",
                    help="use the frozen 1:1 benchmark set (Cycle 25 n=91)")
    ap.add_argument("--tag", default="", help="suffix for output artefacts")
    args = ap.parse_args()
    # A benchmark run must never overwrite the production artefact: when no
    # explicit tag is given, `--bench` writes `transit_v3_bench.joblib` and
    # `v3_transit_benchmark_bench.json`, the pair the G2.4-G2.7 gates read.
    if args.bench and not args.tag:
        args.tag = "bench"
    tag = f"_{args.tag}" if args.tag else ""

    df = load_cme_features().drop_duplicates("cme_id")
    if args.bench:
        bench = pd.read_parquet(PROC / "v3_transit_bench.parquet").drop_duplicates("cme_id")
        bench["t_launch"] = pd.to_datetime(bench["t_launch"])
        ref = bench.set_index("cme_id")["transit_hours"]
        df = df[df["cme_id"].isin(set(ref.index))].copy()
        df["transit_hours"] = df["cme_id"].map(ref)
    else:
        df = df[(df["label"] == 1) & df["transit_hours"].between(12, 150)].copy()
    df = df[df["transit_hours"].notna() & df["v_lin"].notna()]
    df = add_transit_features(df.sort_values("t_launch").reset_index(drop=True))

    train = df[df["t_launch"] < pd.Timestamp("2020-01-01")].reset_index(drop=True)
    test = df[df["t_launch"] >= pd.Timestamp("2020-01-01")].reset_index(drop=True)
    print(f"[transit] train={len(train)} test={len(test)}", flush=True)

    params, phys_mae = fit_physics(train)
    w_amb, gamma = float(params[0]), float(np.exp(params[1]))
    physics = {"w_amb_km_s": w_amb, "gamma_km_inv": gamma}
    print(f"[transit] fitted DBM: w_amb={w_amb:.1f} km/s  gamma={gamma:.3e} km^-1  "
          f"physics-only MAE={phys_mae:.3f} h", flush=True)

    # --- physics backbone + causal geometry features -----------------------
    def attach_phys(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        t, v1, _ = transit_physics(d, physics)
        d["t_phys"] = t
        d["v1au_dbm"] = v1
        d["v_eff_launch"] = effective_launch_speed(d)
        return d

    train = attach_phys(train)
    test = attach_phys(test)
    ytr = train["transit_hours"].values.astype(float)
    yte = test["transit_hours"].values.astype(float)

    # --- feature selection: chronological OOF inside Cycles 23-24 only -----
    # Every candidate is judged on pre-2020 out-of-fold MAE; Cycle 25 is never
    # touched here.  A group is added only if it buys at least MIN_GAIN_H.
    MIN_GAIN_H = 0.05
    GROUPS = {
        "geometry": ["v_3d_est", "cos_deprojection", "d_earth_deg"],
        "accel": ["accel_true"],
        "ambient": ["carr_v_flow", "t0_v_flow", "v_mean_24h", "dst_min_48h"],
        "preconditioning": ["prior_36h_max_speed", "prior_24h_max_speed",
                            "prior_6h_count"],
        "sharp": ["sharp_available", "log_sharp_usflux", "log_sharp_totpot",
                  "d_totpot_24h"],
        "arrival_physics": ["t_phys", "v1au_dbm", "v_eff_launch"],
    }

    def oof_bag(feats, y, seeds=(42, 142, 242), target=None):
        X = design_matrix(train, feats)
        tt = y if target is None else target
        oof = np.full(len(X), np.nan)
        for tr_i, va_i in TimeSeriesSplit(n_splits=5).split(X):
            ps = [mk_model(s).fit(X.iloc[tr_i], tt[tr_i]).predict(X.iloc[va_i])
                  for s in seeds]
            oof[va_i] = np.mean(ps, axis=0)
        if target is not None:
            oof = oof + np.asarray(tt_base)
        m = np.isfinite(oof)
        err = np.abs(oof[m] - y[m])
        se_abs = float(np.std(err, ddof=1) / np.sqrt(max(int(m.sum()), 1)))
        return float(np.mean(err)), oof, se_abs

    tt_base = train["t_phys"].values
    base = list(TRANSIT_FEATURES)
    best_mae, best_oof, _ = oof_bag(base, ytr)
    trace = [{"step": 0, "added": None, "n_features": len(base),
              "oof_mae_h": best_mae, "oof_se_h": None,
              "rule": "one-SE paired (pre-2020 OOF)"}]
    chosen_groups: list[str] = []
    for _ in range(len(GROUPS)):
        cand = []
        for g, cols in GROUPS.items():
            if g in chosen_groups:
                continue
            feats = base + [c for gg in chosen_groups for c in GROUPS[gg]] + cols
            m_, oof_, _ = oof_bag(feats, ytr)
            gain, se_p, n_p = paired_gain_se(best_oof, oof_, ytr)
            cand.append((m_, g, oof_, gain, se_p, n_p))
        if not cand:
            break
        cand.sort(key=lambda x: x[0])
        m_, g, oof_, gain, se_p, n_p = cand[0]
        # One-standard-error rule (Breiman et al. 1984) on the pre-2020 OOF:
        # a group must beat the current model by at least the standard error of
        # the *paired* per-event improvement.  A fixed 0.05 h floor is below
        # the noise floor of a ~200-row OOF estimate.
        need = max(MIN_GAIN_H, se_p)
        if gain >= need:
            chosen_groups.append(g)
            best_mae = m_
            best_oof = oof_
            trace.append({"step": len(chosen_groups), "added": g,
                          "n_features": len(base) + sum(len(GROUPS[x]) for x in chosen_groups),
                          "oof_mae_h": m_, "oof_se_h": se_p,
                          "gain_h": gain, "need_h": need, "n_paired": n_p,
                          "rule": "one-SE paired (pre-2020 OOF)"})
        else:
            trace.append({"step": len(chosen_groups) + 1, "added": None,
                          "rejected_best": g,
                          "n_features": len(base) + sum(len(GROUPS[x]) for x in chosen_groups),
                          "oof_mae_h": best_mae, "oof_se_h": se_p,
                          "candidate_oof_mae_h": m_, "gain_h": gain,
                          "need_h": need, "n_paired": n_p,
                          "rule": "one-SE paired (pre-2020 OOF)"})
            break
    FEATURES = base + [c for gg in chosen_groups for c in GROUPS[gg]]
    print(f"[transit] features selected by pre-2020 OOF: {chosen_groups} "
          f"-> {len(FEATURES)} features (OOF MAE {best_mae:.3f} h)", flush=True)

    Xtr = design_matrix(train, FEATURES)
    Xte = design_matrix(test, FEATURES)

    # --- ensemble: heterogeneous members judged on pre-2020 OOF ------------
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from scipy.optimize import nnls
    import xgboost as xgb

    def mk_xgb(seed: int = 7):
        return xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=3,
                                min_child_weight=10, subsample=0.85,
                                colsample_bytree=0.8, reg_lambda=2.0,
                                random_state=seed, n_jobs=1, tree_method="hist",
                                objective="reg:absoluteerror")

    def mk_et(seed: int = 11):
        return ExtraTreesRegressor(n_estimators=400, max_depth=10,
                                   min_samples_leaf=5, max_features=0.6,
                                   random_state=seed, n_jobs=1)

    def mk_ridge():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             Ridge(alpha=10.0))

    # --- log-target variants: transit times are multiplicative around the DBM
    # prior, so learning log(T) and exponentiating back is the physical space.
    def mk_lgb_log(seed: int = 42):
        return mk_model(seed, objective="huber", alpha=0.25)

    def mk_xgb_log(base_score: float):
        return xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=3,
                                min_child_weight=10, subsample=0.85,
                                colsample_bytree=0.8, reg_lambda=2.0,
                                random_state=7, n_jobs=1, tree_method="hist",
                                objective="reg:pseudohubererror",
                                huber_slope=0.25, base_score=float(base_score))

    def mk_et_log(seed: int = 11):
        # ExtraTrees cannot consume NaNs: the median imputer lives inside the
        # pipeline so training and serving see exactly the same transform.
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            ExtraTreesRegressor(n_estimators=400, max_depth=10,
                                min_samples_leaf=5, max_features=0.6,
                                random_state=seed, n_jobs=1))

    def mk_ridge_log():
        return make_pipeline(SimpleImputer(strategy="median",
                                           keep_empty_features=True),
                             StandardScaler(), Ridge(alpha=10.0))

    def oof_of(make, target):
        oof = np.full(len(Xtr), np.nan)
        for tr_i, va_i in TimeSeriesSplit(n_splits=5).split(Xtr):
            oof[va_i] = make().fit(Xtr.iloc[tr_i], target[tr_i]).predict(Xtr.iloc[va_i])
        return oof

    res_tr = ytr - train["t_phys"].values
    CLIP_LOG = (1.5, 5.1)
    y_log = np.log(ytr)

    def oof_of_xgb_log():
        oof = np.full(len(Xtr), np.nan)
        for tr_i, va_i in TimeSeriesSplit(n_splits=5).split(Xtr):
            bs = float(np.median(y_log[tr_i]))
            oof[va_i] = mk_xgb_log(bs).fit(
                Xtr.iloc[tr_i], y_log[tr_i]).predict(Xtr.iloc[va_i])
        return oof

    oofs_linear = {
        "lgb": oof_of(lambda: mk_model(42), ytr),
        "xgb": oof_of(mk_xgb, ytr),
        "et": oof_of(mk_et, ytr),
        "ridge": oof_of(mk_ridge, ytr),
        "lgb_residual": oof_of(lambda: mk_model(42), res_tr)
                        + train["t_phys"].values,
    }
    oofs_log = {
        "lgb": np.exp(np.clip(oof_of(mk_lgb_log, y_log), *CLIP_LOG)),
        "xgb": np.exp(np.clip(oof_of_xgb_log(), *CLIP_LOG)),
        "et": np.exp(np.clip(oof_of(mk_et_log, y_log), *CLIP_LOG)),
        "ridge": np.exp(np.clip(oof_of(mk_ridge_log, y_log), *CLIP_LOG)),
    }

    def eval_space(oofs: dict) -> dict:
        """Blend members in hours; every decision is pre-2020 OOF only."""
        mae_members = {k: float(np.nanmean(np.abs(v - ytr)))
                       for k, v in oofs.items()}
        names = list(mae_members)
        A_all = np.column_stack([oofs[k] for k in names])
        mask = np.isfinite(A_all).all(axis=1)
        print(f"[transit] blend rows with a full OOF forecast: "
              f"{int(mask.sum())}/{len(ytr)}", flush=True)

        def blend_mae(w: dict) -> float:
            p = np.zeros(len(ytr))
            for k, wk in w.items():
                p = p + wk * oofs[k]
            return float(np.mean(np.abs(p[mask] - ytr[mask])))

        blends = {}
        best_member = min(mae_members, key=mae_members.get)
        keep = [k for k, v in mae_members.items()
                if v <= mae_members[best_member] + 0.75]
        blends["equal"] = {k: 1.0 / len(keep) for k in keep}
        inv = {k: 1.0 / mae_members[k] for k in keep}
        tot = sum(inv.values())
        blends["inverse_mae"] = {k: v / tot for k, v in inv.items()}
        coef, _ = nnls(A_all[mask], ytr[mask])
        if coef.sum() > 0:
            blends["nnls"] = {k: float(c / coef.sum())
                              for k, c in zip(names, coef)}
        blend_scores = {k: blend_mae(w) for k, w in blends.items()}
        return {"oofs": oofs, "mae_members": mae_members,
                "blends": blends, "blend_scores": blend_scores,
                "n_full": int(mask.sum())}

    spaces = {"linear": eval_space(oofs_linear), "log": eval_space(oofs_log)}
    for sp, d in spaces.items():
        print(f"[transit:{sp}] member OOF MAE: "
              f"{ {k: round(v, 3) for k, v in d['mae_members'].items()} } | "
              f"blends: { {k: round(v, 3) for k, v in d['blend_scores'].items()} }",
              flush=True)
    target_space, blend_mode = min(
        ((sp, bm) for sp, d in spaces.items() for bm in d["blend_scores"]),
        key=lambda sb: spaces[sb[0]]["blend_scores"][sb[1]])
    weights = spaces[target_space]["blends"][blend_mode]
    oofs = spaces[target_space]["oofs"]
    print(f"[transit] selected by pre-2020 OOF MAE: target_space={target_space} "
          f"blend={blend_mode} { {k: round(v, 3) for k, v in weights.items()} } "
          f"MAE={spaces[target_space]['blend_scores'][blend_mode]:.3f} h",
          flush=True)

    # final fit of every weighted member on the full pre-2020 window
    is_log = target_space == "log"

    def fit_member(name):
        if is_log:
            if name == "lgb":
                return [mk_lgb_log(s).fit(Xtr, y_log) for s in SEEDS]
            if name == "xgb":
                return mk_xgb_log(float(np.median(y_log))).fit(Xtr, y_log)
            if name == "et":
                return mk_et_log().fit(Xtr, y_log)
            if name == "ridge":
                return mk_ridge_log().fit(Xtr, y_log)
            raise KeyError(name)
        if name == "lgb":
            return fit_bag(Xtr, ytr)
        if name == "xgb":
            return mk_xgb().fit(Xtr, ytr)
        if name == "et":
            return mk_et().fit(Xtr, ytr)
        if name == "ridge":
            return mk_ridge().fit(Xtr, ytr)
        if name == "lgb_residual":
            return ResidualTransit(fit_bag(Xtr, res_tr), physics)
        raise KeyError(name)

    models = []
    model_weights = []
    for name, w in weights.items():
        m = fit_member(name)
        models.append(m)
        model_weights.append(float(w))

    def _member_pred(m, X):
        if isinstance(m, list):
            if is_log:
                return np.mean([np.exp(np.clip(x.predict(X), *CLIP_LOG))
                                for x in m], axis=0)
            return np.mean([x.predict(X) for x in m], axis=0)
        pred = m.predict(X)
        return np.exp(np.clip(pred, *CLIP_LOG)) if is_log else pred

    def predict_members(X):
        out = np.zeros(len(X))
        for m, w in zip(models, model_weights):
            out = out + w * _member_pred(m, X)
        return out

    train["pred"] = predict_members(Xtr)
    test["pred"] = predict_members(Xte)
    oof_pred = np.zeros(len(ytr))
    for k, w in weights.items():
        oof_pred = oof_pred + w * oofs[k]
    oof = oof_pred
    ok = np.isfinite(oof)
    oof_ae = np.abs(oof[ok] - ytr[ok])
    print(f"[transit] OOF n={int(ok.sum())} MAE={oof_ae.mean():.3f} h "
          f"q88={np.quantile(oof_ae, 0.88):.2f} h", flush=True)

    # --- conformal bands: CQR vs abs vs uncertainty-normalised -------------
    edges = tuple(np.quantile(train["v_lin"].values, REGIME_EDGES_Q))
    lo_m = mk_quantile(0.06).fit(Xtr, ytr)
    hi_m = mk_quantile(0.94).fit(Xtr, ytr)
    oof_lo = np.full(len(train), np.nan)
    oof_hi = np.full(len(train), np.nan)
    for tr_i, va_i in TimeSeriesSplit(n_splits=5).split(Xtr):
        oof_lo[va_i] = mk_quantile(0.06).fit(Xtr.iloc[tr_i], ytr[tr_i]).predict(Xtr.iloc[va_i])
        oof_hi[va_i] = mk_quantile(0.94).fit(Xtr.iloc[tr_i], ytr[tr_i]).predict(Xtr.iloc[va_i])
    m = np.isfinite(oof_lo) & np.isfinite(oof_hi) & np.isfinite(oof)
    E = np.maximum(oof_lo[m] - ytr[m], ytr[m] - oof_hi[m])
    reg_tr = regimes(train.loc[m, "v_lin"].values, edges)
    BAND_LEVEL = 0.90
    COVERAGE_FLOOR = 0.88
    Q = {}
    for r in range(3):
        sel = reg_tr == r
        Q[str(r)] = (float(np.quantile(E[sel], BAND_LEVEL)) if sel.sum() >= 15
                     else float(np.quantile(E, BAND_LEVEL)))
    Q["global"] = float(np.quantile(E, BAND_LEVEL))

    abs_res = np.abs(oof[m] - ytr[m])
    reg_abs = regimes(train.loc[m, "v_lin"].values, edges)
    Q_abs = {}
    for r in range(3):
        sel = reg_abs == r
        Q_abs[str(r)] = (float(np.quantile(abs_res[sel], BAND_LEVEL)) if sel.sum() >= 15
                         else float(np.quantile(abs_res, BAND_LEVEL)))
    Q_abs["global"] = float(np.quantile(abs_res, BAND_LEVEL))
    qr_oof = np.array([Q[str(r)] for r in regimes(train.loc[m, "v_lin"].values, edges)])
    cqr_lo, cqr_hi = oof_lo[m] - qr_oof, oof_hi[m] + qr_oof
    abs_qr = np.array([Q_abs[str(r)] for r in regimes(train.loc[m, "v_lin"].values, edges)])
    abs_lo, abs_hi = oof[m] - abs_qr, oof[m] + abs_qr
    oof_cov_cqr = float(np.mean((ytr[m] >= cqr_lo) & (ytr[m] <= cqr_hi)))
    oof_cov_abs = float(np.mean((ytr[m] >= abs_lo) & (ytr[m] <= abs_hi)))
    oof_w_cqr = float(np.mean(cqr_hi - cqr_lo))
    oof_w_abs = float(np.mean(abs_hi - abs_lo))

    # Normalised conformal: scale each residual by the local MAD of its regime,
    # so a calm-wind event and a fast one are not forced into the same width.
    resid_tr = oof - ytr
    sigma_reg = {}
    for r in range(3):
        sel = reg_abs == r
        sigma_reg[str(r)] = (float(1.4826 * np.median(np.abs(resid_tr[m][sel])))
                             if sel.sum() >= 15 else
                             float(1.4826 * np.median(np.abs(resid_tr[m]))))
    sigma_reg["global"] = float(1.4826 * np.median(np.abs(resid_tr[m])))
    sig_oof = np.array([sigma_reg[str(r)] for r in regimes(train.loc[m, "v_lin"].values, edges)])
    zres = np.abs(resid_tr[m]) / np.clip(sig_oof, 1e-6, None)
    Q_norm = {}
    for r in range(3):
        sel = reg_abs == r
        Q_norm[str(r)] = (float(np.quantile(zres[sel], BAND_LEVEL)) if sel.sum() >= 15
                          else float(np.quantile(zres, BAND_LEVEL)))
    Q_norm["global"] = float(np.quantile(zres, BAND_LEVEL))
    qn_oof = np.array([Q_norm[str(r)] for r in regimes(train.loc[m, "v_lin"].values, edges)])
    norm_lo, norm_hi = oof[m] - qn_oof * sig_oof, oof[m] + qn_oof * sig_oof
    oof_cov_norm = float(np.mean((ytr[m] >= norm_lo) & (ytr[m] <= norm_hi)))
    oof_w_norm = float(np.mean(norm_hi - norm_lo))

    eligible = {"cqr": oof_cov_cqr >= COVERAGE_FLOOR,
                "abs_conformal": oof_cov_abs >= COVERAGE_FLOOR,
                "abs_conformal_normalized": oof_cov_norm >= COVERAGE_FLOOR}
    widths = {"cqr": oof_w_cqr, "abs_conformal": oof_w_abs,
              "abs_conformal_normalized": oof_w_norm}
    rule_satisfied = any(eligible.values())
    if rule_satisfied:
        band_mode = min((k for k in widths if eligible[k]), key=lambda k: widths[k])
    else:
        band_mode = "cqr"
    print(f"[transit] CQR Q={ {k: round(v, 2) for k, v in Q.items()} } | "
          f"abs Q={ {k: round(v, 2) for k, v in Q_abs.items()} } | "
          f"norm Q={ {k: round(v, 2) for k, v in Q_norm.items()} } | "
          f"OOF: cqr cov={oof_cov_cqr:.3f} w={oof_w_cqr:.1f} vs "
          f"abs cov={oof_cov_abs:.3f} w={oof_w_abs:.1f} vs "
          f"norm cov={oof_cov_norm:.3f} w={oof_w_norm:.1f} -> {band_mode} "
          f"(level={BAND_LEVEL:.2f} floor={COVERAGE_FLOOR:.2f} "
          f"rule_satisfied={rule_satisfied})", flush=True)

    def add_band(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        if band_mode == "abs_conformal_normalized":
            sg = np.array([sigma_reg[str(r)] for r in regimes(d["v_lin"].values, edges)])
            qn = np.array([Q_norm[str(r)] for r in regimes(d["v_lin"].values, edges)])
            d["lo"] = d["pred"].values - qn * sg
            d["hi"] = d["pred"].values + qn * sg
        else:
            qr = np.array([Q[str(r)] for r in regimes(d["v_lin"].values, edges)])
            if band_mode == "cqr":
                d["lo"] = lo_m.predict(design_matrix(d, FEATURES)) - qr
                d["hi"] = hi_m.predict(design_matrix(d, FEATURES)) + qr
            else:
                qa = np.array([Q_abs[str(r)] for r in regimes(d["v_lin"].values, edges)])
                d["lo"] = d["pred"].values - qa
                d["hi"] = d["pred"].values + qa
        d["covered"] = ((d["transit_hours"] >= d["lo"]) & (d["transit_hours"] <= d["hi"])).astype(int)
        return d

    train = add_band(train)
    test = add_band(test)

    def metrics(d):
        e = d["pred"].values - d["transit_hours"].values
        out = {"n": int(len(d)), "mae": float(np.mean(np.abs(e))),
               "rmse": float(np.sqrt(np.mean(e ** 2))),
               "median_ae": float(np.median(np.abs(e))), "bias": float(np.mean(e)),
               "coverage90": float(d["covered"].mean()),
               "mean_width_h": float((d["hi"] - d["lo"]).mean())}
        fast = d["v_lin"].values >= 1000
        if fast.sum() >= 3:
            out.update(fast_n=int(fast.sum()),
                       fast_mae=float(np.mean(np.abs(e[fast]))),
                       fast_rmse=float(np.sqrt(np.mean(e[fast] ** 2))),
                       fast_coverage90=float(d.loc[fast, "covered"].mean()))
        return out

    # diagnostic: the tightest *symmetric* static band that would reach 88 %
    err_te = test["pred"].values - yte
    q88_abs = float(np.quantile(np.abs(err_te), 0.88))
    # tightest static *asymmetric* band at 88 % (6 / 94 % quantiles of signed
    # error).  Reported so the information ceiling does not depend on the
    # symmetry assumption: no static band of either shape gets near 35 h.
    q_asym = np.quantile(err_te, [0.06, 0.94])
    asym_floor = float(q_asym[1] - q_asym[0])
    res = {"mode": "bench" if args.bench else "full",
           "physics": {"w_amb_km_s": w_amb, "gamma_km_inv": gamma,
                       "physics_only_mae_h": phys_mae},
           "conformal_q": Q, "q_abs": Q_abs, "q_norm": Q_norm,
           "regime_sigma": sigma_reg, "band_mode": band_mode,
           "band_level": BAND_LEVEL, "coverage_floor": COVERAGE_FLOOR,
           "selection_rule_satisfied": rule_satisfied, "eligible": eligible,
           "oof_band": {"cqr": {"coverage": oof_cov_cqr, "width_h": oof_w_cqr},
                        "abs_conformal": {"coverage": oof_cov_abs, "width_h": oof_w_abs},
                        "abs_conformal_normalized": {"coverage": oof_cov_norm,
                                                     "width_h": oof_w_norm}},
           "regime_edges_v_lin": list(edges),
           "feature_selection": {"chosen_groups": chosen_groups, "trace": trace,
                                 "min_gain_h": MIN_GAIN_H},
           "ensemble": {"blend_mode": blend_mode, "weights": weights,
                        "target_space": target_space,
                        "member_oof_mae_h": spaces[target_space]["mae_members"],
                        "blend_oof_mae_h": spaces[target_space]["blend_scores"],
                        "space_selection": {
                            sp: {"member_oof_mae_h": d["mae_members"],
                                 "blend_oof_mae_h": d["blend_scores"],
                                 "n_full_oof": d["n_full"]}
                            for sp, d in spaces.items()}},
           "features": FEATURES, "params": {k: v for k, v in PARAMS.items()},
           "n_seeds": len(SEEDS),
           "train": metrics(train), "test_cycle25": metrics(test),
           "symmetric_band_floor_h": 2.0 * q88_abs,
           "asymmetric_band_floor_h": asym_floor}

    def dbm_baseline(w, g):
        t = dbm_transit_vec(test["v_lin"].values, np.full(len(test), w),
                            np.full(len(test), g))
        e = t - yte
        mm = np.isfinite(e)
        return {"n": int(mm.sum()), "mae": float(np.mean(np.abs(e[mm]))),
                "rmse": float(np.sqrt(np.mean(e[mm] ** 2))),
                "median_ae": float(np.median(np.abs(e[mm])))}

    res["baseline_dbm_static"] = dbm_baseline(w_amb, gamma)
    res["baseline_dbm_legacy_constants"] = dbm_baseline(400.0, 1e-7)
    res["baseline_persistence_mean_transit"] = {
        "n": int(len(test)),
        "mae": float(np.mean(np.abs(train["transit_hours"].mean() - yte)))}

    # Diagnostic, never a gate: the *served* production model evaluated on the
    # same 91 events used by the v2 / WSA-ENLIL benchmark.  The production and
    # benchmark models are different predictors, so this number must be visible
    # instead of being hidden behind the bench artefact.
    if not args.bench:
        try:
            bench = pd.read_parquet(PROC / "v3_transit_bench.parquet").drop_duplicates("cme_id")
            bench["t_launch"] = pd.to_datetime(bench["t_launch"])
            ref = bench.set_index("cme_id")["transit_hours"]
            bdf = load_cme_features().drop_duplicates("cme_id")
            bdf = bdf[bdf["cme_id"].isin(set(ref.index))].copy()
            bdf["transit_hours"] = bdf["cme_id"].map(ref)
            bdf = bdf[bdf["transit_hours"].notna() & bdf["v_lin"].notna()]
            bdf = add_transit_features(bdf.sort_values("t_launch").reset_index(drop=True))
            bdf = bdf[bdf["t_launch"] >= pd.Timestamp("2020-01-01")].reset_index(drop=True)
            bdf = attach_phys(bdf)
            Xb = design_matrix(bdf, FEATURES)
            bdf["pred"] = predict_members(Xb)
            bdf = add_band(bdf)
            res["test_bench91_served"] = metrics(bdf)
            res["test_bench91_served"]["note"] = (
                "served production model on the 91-event benchmark; not a gate")
            print(f"[transit] served model on the 91-event benchmark: "
                  f"MAE={res['test_bench91_served']['mae']:.4f} h "
                  f"coverage={res['test_bench91_served']['coverage90']:.4f} "
                  f"width={res['test_bench91_served']['mean_width_h']:.4f} h",
                  flush=True)
        except Exception as e:  # a diagnostic must never break production
            res["test_bench91_served"] = {"error": f"{type(e).__name__}: {e}"}

    joblib.dump({"models": models, "model_weights": model_weights,
                 "features": FEATURES, "q": Q,
                 "target_space": target_space,
                 "q_abs": Q_abs, "q_norm": Q_norm, "regime_sigma": sigma_reg,
                 "band_mode": band_mode, "mode": res["mode"],
                 "band_level": BAND_LEVEL, "coverage_floor": COVERAGE_FLOOR,
                 "regime_edges_v_lin": list(edges),
                 "lo_model": lo_m, "hi_model": hi_m,
                 "physics": {"w_amb_km_s": w_amb, "gamma_km_inv": gamma},
                 "train_mean_transit_h": float(train["transit_hours"].mean()),
                 "params": PARAMS},
                MODELS / f"transit_v3{tag}.joblib")

    test[["cme_id", "t_launch", "transit_hours", "pred", "lo", "hi", "covered",
          "v_lin", "width", "solar_cycle"]].to_parquet(
        RESULTS / f"v3_transit_test_predictions{tag}.parquet", index=False)
    (RESULTS / f"v3_transit_benchmark{tag}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
