#!/usr/bin/env python3
"""HELIOS v3.4 - Stage 4A: HELIOS-Apex v3 calibrated 24-72 h severity.

Predicts, strictly from launch-time observables, the *distribution* of the
minimum Dst reached by the storm this CME drives:

    P(Dst_min <= t)  for t in {-50, -100, -200} nT
    Dst_min conformal band (90 %)
    B_mag_max conformal band (90 %)
    Bz risk tag (SHEATH_DRIVEN / SOUTHERN_SOURCE / POLARITY_UNCERTAIN_AWAIT_L1)

Why a distribution and not three independent classifiers: with 114 training
positives at -100 nT a hard classifier has no chance of being calibrated.  A
single heteroscedastic regressor shares strength across thresholds; several
probability maps (Gaussian tail, empirical conformal law, logit-tail blend) are
built from its predictive distribution, the map is chosen per threshold by
pre-2020 out-of-fold Brier skill, and the winner is Platt-aligned on the same
out-of-fold predictions.  The calibrator is fitted on Cycles 23-24 only.

No binary "will Bz be south" classifier: the sign of Bz two days ahead is not
predictable without in-situ sensing, so the model reports a calibrated risk
instead of a binary southward call.

Outputs
  models/v3/apex_*.joblib
  results/v3_apex_benchmark.json
  results/v3_apex_test_predictions.parquet
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_common import (RESULTS, ROOT, MeanRegressor, design_matrix,
                       load_cme_features)

import lightgbm as lgb
import xgboost as xgb

MODELS = ROOT / "models" / "v3"
MODELS.mkdir(parents=True, exist_ok=True)
THRESHOLDS = (-50, -100, -200)

APEX_FEATURES = [
    # launch kinematics / morphology
    "v_lin", "width", "is_halo", "is_partial_halo", "accel_true", "v_20R",
    "v_20r_ratio", "dv_2nd", "log_mass", "log_energy", "log_momentum",
    # direction of the eruption relative to Earth
    "src_lon", "src_lat", "abs_lon", "d_earth_deg", "cos_deprojection",
    "v_3d_est", "cone_margin_deg", "has_position", "has_source",
    # eruptive driver
    "log_goes_flux", "flare_cme_dt_min", "pos_angle_consistency",
    # ambient heliosphere at launch (causal)
    "t0_dst", "t0_v_flow", "t0_bz_gsm", "t0_pdyn", "v_mean_24h",
    "bz_min_24h", "dst_min_48h", "carr_v_flow", "carr_np_dens",
    # preconditioning
    "prior_6h_count", "prior_24h_count", "prior_36h_count",
    "prior_36h_max_speed", "prior_36h_max_width",
    # source-region magnetism
    "sharp_available", "log_sharp_usflux", "log_sharp_totpot",
    "sharp_r_value", "sharp_meanshr", "sharp_area_acr", "d_totpot_24h",
    # deterministic Russell-McPherron geometry of the impact date
    "dipole_tilt", "rm_equinox",
]


def add_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, fully causal seasonal geometry of the impact window."""
    df = df.copy()
    t = pd.to_datetime(df["t_launch"])
    doy = t.dt.dayofyear.values + t.dt.hour.values / 24.0
    tilt = np.radians(7.25) * np.sin(2 * np.pi * (doy - 80) / 365.25)
    df["dipole_tilt"] = np.degrees(tilt)
    df["rm_equinox"] = np.abs(np.sin(2 * np.pi * (doy - 80) / 365.25))
    if "log_goes_flux" in df:
        df["log_goes_flux"] = pd.to_numeric(df["log_goes_flux"], errors="coerce")
    return df


def attach_physics(df: pd.DataFrame) -> pd.DataFrame:
    """Causal arrival physics: fitted DBM speed/time plus 1-AU forcing estimates.

    `pdyn_1au` and `ey_1au_max` are physical formulas (ram pressure and
    rectified dawn-dusk field) evaluated with the *ambient* L1 state at launch
    and the DBM-predicted arrival speed.  They use no fitted constants, and
    both are computable live from the same inputs predict.py has.
    """
    from v3_common import attach_transit_physics

    d = attach_transit_physics(df)
    v1 = pd.to_numeric(d["v1au_dbm"], errors="coerce").astype(float)
    n0 = pd.to_numeric(d["t0_np_dens"], errors="coerce").astype(float)
    bz0 = pd.to_numeric(d["t0_bz_gsm"], errors="coerce").astype(float)
    d["pdyn_1au"] = 1.6726e-6 * n0 * v1 ** 2
    d["ey_1au_max"] = v1 * bz0.abs() * 1e-3
    return d


def mk_reg(seed: int, **over) -> lgb.LGBMRegressor:
    kw = dict(n_estimators=400, learning_rate=0.03, num_leaves=3, max_depth=3,
              min_child_samples=20, subsample=0.85, colsample_bytree=0.8,
              reg_lambda=2.0, objective="huber", random_state=seed, n_jobs=1,
              verbose=-1)
    if over.get("objective", kw["objective"]) == "huber" and "alpha" not in over:
        # A Huber delta of 1 nT on a target whose std is ~61 nT treats almost
        # every event as an outlier and collapses mu to the conditional median
        # (the v3.3 Apex failure).  50 nT keeps the robustness but preserves
        # the storm-to-storm spread; chosen on pre-2020 OOF only.
        kw.update(alpha=50.0, num_leaves=5, max_depth=3, learning_rate=0.025)
    kw.update(over)
    return lgb.LGBMRegressor(**kw)


def mk_clf(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=3,
                              max_depth=3, min_child_samples=20, subsample=0.85,
                              colsample_bytree=0.8, reg_lambda=2.0,
                              random_state=seed, n_jobs=1, verbose=-1)


def _platt(p, y):
    eps = 1e-6
    z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))[:, None]
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def _apply_platt(p, ab):
    eps = 1e-6
    z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    return 1.0 / (1.0 + np.exp(-(ab[0] * z + ab[1])))


def _bss(y, p) -> float:
    ref = brier_score_loss(y, np.full(len(y), np.mean(y)))
    return float(1 - brier_score_loss(y, p) / max(ref, 1e-12))


def _oof_predict(X: pd.DataFrame, y: np.ndarray, make, n_splits: int = 5):
    oof = np.full(len(X), np.nan)
    for tr_i, va_i in TimeSeriesSplit(n_splits=n_splits).split(X):
        oof[va_i] = make().fit(X.iloc[tr_i], y[tr_i]).predict(X.iloc[va_i])
    return oof


def _oof_predict_proba(X: pd.DataFrame, y: np.ndarray, make, n_splits: int = 5):
    oof = np.full(len(X), np.nan)
    for tr_i, va_i in TimeSeriesSplit(n_splits=n_splits).split(X):
        oof[va_i] = make().fit(X.iloc[tr_i], y[tr_i]).predict_proba(X.iloc[va_i])[:, 1]
    return oof


def main() -> int:
    df = load_cme_features()
    df = df[(df["label"] == 1) & df["dst_min_target"].notna()].copy()
    df = attach_physics(add_geometry(df)).sort_values("t_launch").reset_index(drop=True)
    train = df[df["t_launch"] < pd.Timestamp("2020-01-01")].reset_index(drop=True)
    test = df[df["t_launch"] >= pd.Timestamp("2020-01-01")].reset_index(drop=True)
    ytr = train["dst_min_target"].values.astype(float)
    yte = test["dst_min_target"].values.astype(float)
    print(f"[apex] train={len(train)} test={len(test)} "
          f"test<=-100={int((yte<=-100).sum())} <=-50={int((yte<=-50).sum())} "
          f"<=-200={int((yte<=-200).sum())}", flush=True)

    # ---- feature selection: chronological OOF, pre-2020 only --------------
    MIN_GAIN_NT = 0.5
    GROUPS = {
        "arrival_physics": ["v1au_dbm", "t_phys", "v_eff_launch"],
        "causal_forcing": ["pdyn_1au", "ey_1au_max"],
    }

    def oof_mae(feats, **over):
        X = design_matrix(train, feats)
        oof = _oof_predict(X, ytr, lambda: mk_reg(9, **over))
        m = np.isfinite(oof)
        return float(np.mean(np.abs(oof[m] - ytr[m]))), oof

    base_feats = list(APEX_FEATURES)
    best_mae, _ = oof_mae(base_feats)
    trace = [{"step": 0, "added": None, "n_features": len(base_feats),
              "oof_mae_nt": best_mae}]
    chosen: list[str] = []
    for _ in range(len(GROUPS)):
        cand = []
        for g, cols in GROUPS.items():
            if g in chosen:
                continue
            feats = base_feats + [c for gg in chosen for c in GROUPS[gg]] + cols
            m_, _ = oof_mae(feats)
            cand.append((m_, g))
        if not cand:
            break
        cand.sort()
        m_, g = cand[0]
        if m_ < best_mae - MIN_GAIN_NT:
            chosen.append(g)
            best_mae = m_
            trace.append({"step": len(chosen), "added": g,
                          "n_features": len(base_feats) + sum(len(GROUPS[x]) for x in chosen),
                          "oof_mae_nt": m_})
        else:
            break
    APEX_SELECTED = base_feats + [c for gg in chosen for c in GROUPS[gg]]
    print(f"[apex] features selected by pre-2020 OOF: {chosen} -> "
          f"{len(APEX_SELECTED)} features (OOF MAE {best_mae:.2f} nT)", flush=True)

    # ---- capacity: only grow the tree if OOF MAE improves -----------------
    CAPACITIES = [{"num_leaves": 3, "max_depth": 3},
                  {"num_leaves": 7, "max_depth": 4},
                  {"num_leaves": 15, "max_depth": 5}]
    cap_scores = {str(c["num_leaves"]): oof_mae(APEX_SELECTED, **c)[0] for c in CAPACITIES}
    cap_key = min(cap_scores, key=cap_scores.get)
    cap = next(c for c in CAPACITIES if str(c["num_leaves"]) == cap_key)
    print(f"[apex] capacity OOF MAE: { {k: round(v, 2) for k, v in cap_scores.items()} } "
          f"-> num_leaves={cap['num_leaves']}", flush=True)

    Xtr = design_matrix(train, APEX_SELECTED)
    Xte = design_matrix(test, APEX_SELECTED)

    # ---- heteroscedastic regressor: mu, conditional scale, conformal law ----
    # The mean model is itself a pre-2020 OOF decision.  A single LightGBM
    # Huber with delta 50 nT, an XGBoost pseudo-Huber, a strongly regularised
    # Ridge and their average compete on the same chronological folds; the
    # winner's OOF vector feeds the scale model and the conformal law, so the
    # nested-OOF structure (no leakage) is preserved.
    def mk_mu_lgb():
        return mk_reg(9, objective="huber", alpha=50.0, num_leaves=5,
                      max_depth=3, learning_rate=0.025)

    def mk_mu_xgb(base_score: float):
        return xgb.XGBRegressor(
            n_estimators=350, learning_rate=0.025, max_depth=3,
            min_child_weight=15, subsample=0.85, colsample_bytree=0.8,
            reg_lambda=5.0, objective="reg:pseudohubererror", huber_slope=50.0,
            base_score=float(base_score), random_state=9, n_jobs=1,
            tree_method="hist")

    def mk_mu_ridge():
        return Pipeline([
            ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("sc", StandardScaler()),
            ("r", Ridge(alpha=100.0)),
        ])

    mu_cand_oof = {"lgb_h50": np.full(len(Xtr), np.nan),
                   "xgb_h50": np.full(len(Xtr), np.nan),
                   "ridge_100": np.full(len(Xtr), np.nan)}
    for tr_i, va_i in TimeSeriesSplit(n_splits=5).split(Xtr):
        mu_cand_oof["lgb_h50"][va_i] = mk_mu_lgb().fit(
            Xtr.iloc[tr_i], ytr[tr_i]).predict(Xtr.iloc[va_i])
        mu_cand_oof["xgb_h50"][va_i] = mk_mu_xgb(
            float(np.median(ytr[tr_i]))).fit(
            Xtr.iloc[tr_i], ytr[tr_i]).predict(Xtr.iloc[va_i])
        mu_cand_oof["ridge_100"][va_i] = mk_mu_ridge().fit(
            Xtr.iloc[tr_i], ytr[tr_i]).predict(Xtr.iloc[va_i])
    mu_cand_oof["blend_3"] = (mu_cand_oof["lgb_h50"] + mu_cand_oof["xgb_h50"]
                              + mu_cand_oof["ridge_100"]) / 3.0
    mu_scores = {k: float(np.nanmean(np.abs(v - ytr)))
                 for k, v in mu_cand_oof.items()}
    mu_winner = min(mu_scores, key=mu_scores.get)
    m_lgb = mk_mu_lgb().fit(Xtr, ytr)
    m_xgb = mk_mu_xgb(float(np.median(ytr))).fit(Xtr, ytr)
    m_ridge = mk_mu_ridge().fit(Xtr, ytr)
    mu_models = {"lgb_h50": m_lgb, "xgb_h50": m_xgb, "ridge_100": m_ridge,
                 "blend_3": MeanRegressor([m_lgb, m_xgb, m_ridge])}
    mu_oof = mu_cand_oof[mu_winner]
    r_oof = ytr - mu_oof
    okr = np.isfinite(r_oof)

    def mk_sig():
        return mk_reg(13, objective="regression", num_leaves=5, max_depth=2,
                      n_estimators=250, learning_rate=0.04)

    # The conditional scale must itself be predicted out-of-fold: fitting it on
    # Xtr and then using sig_model.predict(Xtr) to standardise the OOF
    # residuals leaks the scale model into its own evaluation.  `sig_oof` is
    # the nested OOF scale; the final `sig_model` is fitted on all OOF
    # residuals and used only for serving.
    sig_oof = np.full(len(Xtr), np.nan)
    for tr_i, va_i in TimeSeriesSplit(n_splits=5).split(Xtr):
        ok_fit = np.isfinite(r_oof[tr_i])
        if ok_fit.sum() < 20:
            continue
        sig_oof[va_i] = mk_sig().fit(
            Xtr.iloc[tr_i][ok_fit], np.abs(r_oof[tr_i][ok_fit])
        ).predict(Xtr.iloc[va_i])
    sig_tr = np.clip(sig_oof, 15.0, 220.0)
    sigma = float(np.nanstd(r_oof))
    # The first chronological fold has no earlier fold to train a scale model
    # on, so its nested-OOF scale is undefined.  Those rows are *never* used to
    # fit the scale, so filling them with the global OOF residual scale is
    # clean w.r.t. their own labels and keeps the calibration sample complete.
    sig_tr = np.where(np.isfinite(sig_tr), sig_tr, sigma)
    sig_model = mk_sig().fit(Xtr[okr], np.abs(r_oof[okr]))
    sig_te = np.clip(sig_model.predict(Xte), 15.0, 220.0)
    okr = okr & np.isfinite(sig_tr)
    z_oof = r_oof / sig_tr
    mu_model = mu_models[mu_winner]
    mu_te = mu_model.predict(Xte)
    print(f"[apex] mu winner={mu_winner} OOF MAE={mu_scores[mu_winner]:.2f} nT "
          f"(candidates { {k: round(v, 2) for k, v in mu_scores.items()} })  "
          f"sigma_oof={sigma:.2f} nT  sigma_cond range="
          f"[{sig_tr.min():.1f},{sig_tr.max():.1f}]", flush=True)

    def empirical_law(t: float, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
        """Conformal predictive law: P(Dst_min <= t) from the OOF residual law."""
        zq = (t - mu) / sig
        zz = z_oof[okr]
        return np.array([np.mean(zz <= z) for z in np.atleast_1d(zq)])

    res = {"n_train": int(len(train)), "n_test": int(len(test)),
           "sigma_dst_oof": sigma, "features": APEX_SELECTED,
           "mu_selection": {"winner": mu_winner,
                            "oof_mae_nt": mu_scores,
                            "criterion": "pre-2020 chronological OOF MAE (TimeSeriesSplit 5)"},
           "feature_selection": {"chosen_groups": chosen, "trace": trace,
                                 "min_gain_nt": MIN_GAIN_NT},
           "capacity": {"chosen": cap, "oof_mae_nt": cap_scores},
           "dst_min_test_observed": {str(t): int((yte <= t).sum()) for t in THRESHOLDS}}
    probs = {}
    clf_store = {}
    # Compact physical block: twelve causal columns that a linear model can
    # actually identify on ~200 events.  The full set is used by the trees.
    COMPACT_PHYS_12 = [c for c in ("v_lin", "v_3d_est", "cos_deprojection",
                                   "is_halo", "v1au_dbm", "pdyn_1au",
                                   "ey_1au_max", "bz_min_24h", "dst_min_48h",
                                   "rm_equinox", "log_goes_flux", "has_position")
                       if c in train.columns]
    Xc_tr = design_matrix(train, COMPACT_PHYS_12)
    Xc_te = design_matrix(test, COMPACT_PHYS_12)
    for thr in THRESHOLDS:
        ytr_b = (ytr <= thr).astype(int)
        yte_b = (yte <= thr).astype(int)
        cand_oof, cand_raw = {}, {}
        cand_oof["gaussian_tail"] = norm.cdf((thr - mu_oof) / sigma)
        cand_raw["gaussian_tail"] = norm.cdf((thr - mu_te) / sigma)
        cand_oof["gaussian_tail_conditional"] = norm.cdf((thr - mu_oof) / sig_tr)
        cand_raw["gaussian_tail_conditional"] = norm.cdf((thr - mu_te) / sig_te)
        cand_oof["conformal_empirical"] = empirical_law(thr, mu_oof, sig_tr)
        cand_raw["conformal_empirical"] = empirical_law(thr, mu_te, sig_te)
        p_clf_oof = _oof_predict(Xtr, ytr_b, lambda: mk_clf(9))
        clf_model = mk_clf(9).fit(Xtr, ytr_b)
        p_clf = clf_model.predict_proba(Xte)[:, 1]
        cand_oof["direct_classifier"] = p_clf_oof
        cand_raw["direct_classifier"] = p_clf

        # Candidate 5: regularised logistic regression on standardised features.
        # Trees can memorise a 200-row OOF; a linear model with a strong L2
        # penalty is the honest low-variance alternative.  Median imputation is
        # fitted inside each fold, so no statistic crosses the split.
        def mk_logit():
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            return Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("lr", LogisticRegression(C=0.3, max_iter=2000, solver="lbfgs")),
            ])

        degenerate = not (0 < int(ytr_b.sum()) < len(ytr_b))
        p_logit_oof = (np.full(len(ytr_b), float(ytr_b.mean())) if degenerate
                       else _oof_predict_proba(Xtr, ytr_b, mk_logit))
        logit_model = None if degenerate else mk_logit().fit(Xtr, ytr_b)
        cand_oof["logistic_regularized"] = p_logit_oof
        cand_raw["logistic_regularized"] = (np.full(len(yte_b), float(ytr_b.mean()))
                                            if logit_model is None
                                            else logit_model.predict_proba(Xte)[:, 1])

        # Candidate 5b: the same regularised logistic on the compact physical
        # block, plus its 50/50 blend with the Gaussian tail.  Both were
        # validated on pre-2020 OOF only.
        def mk_compact_logit():
            return Pipeline([
                ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("sc", StandardScaler()),
                ("lr", LogisticRegression(C=0.1, max_iter=1000)),
            ])

        p_compact_oof = (np.full(len(ytr_b), float(ytr_b.mean())) if degenerate
                         else _oof_predict_proba(Xc_tr, ytr_b, mk_compact_logit))
        compact_model = (None if degenerate
                         else mk_compact_logit().fit(Xc_tr, ytr_b))
        cand_oof["compact_logit"] = p_compact_oof
        cand_raw["compact_logit"] = (np.full(len(yte_b), float(ytr_b.mean()))
                                     if compact_model is None
                                     else compact_model.predict_proba(Xc_te)[:, 1])
        cand_oof["blend_logit_tail"] = (0.5 * p_compact_oof
                                        + 0.5 * cand_oof["gaussian_tail"])
        cand_raw["blend_logit_tail"] = (0.5 * cand_raw["compact_logit"]
                                        + 0.5 * cand_raw["gaussian_tail"])

        # Candidate 6: climatology (training base rate).  Its BSS is exactly 0
        # by construction, so selecting the OOF argmax can never be *worse*
        # than climatology: this is the guard against destructive skill.
        models_by_name = {"direct_classifier": clf_model,
                          "logistic_regularized": logit_model,
                          "compact_logit": compact_model,
                          "blend_logit_tail": compact_model}
        ok = np.isfinite(cand_oof["gaussian_tail"]) & np.isfinite(sig_tr)
        for v in cand_oof.values():
            ok &= np.isfinite(v)
        # the prior is measured on exactly the rows the BSS is evaluated on
        prior = float(ytr_b[ok].mean())
        cand_oof["climatology"] = np.full(len(ytr_b), prior)
        cand_raw["climatology"] = np.full(len(yte_b), prior)
        scores = {k: _bss(ytr_b[ok], np.clip(v[ok], 1e-6, 1 - 1e-6)) for k, v in cand_oof.items()}
        # Pre-registered simplicity rule: with fewer than 20 out-of-fold positives
        # the selection is noise, so the model with the fewest free parameters wins
        # (global scale).  This is decided a priori, never from test performance.
        n_pos_oof = int(ytr_b[ok].sum())
        if n_pos_oof < 20:
            which = "gaussian_tail"
        else:
            which = max(scores, key=scores.get)
        p_oof = cand_oof[which]
        p_raw = cand_raw[which]
        if 0 < int(ytr_b[ok].sum()) < int(ok.sum()):
            ab = _platt(p_oof[ok], ytr_b[ok])
        else:  # a single-class OOF slice cannot identify a Platt map
            ab = (1.0, 0.0)
        p_cal = _apply_platt(p_raw, ab)
        entry = {"kind": which,
                 "model": models_by_name.get(which),
                 "prior": prior}
        if which in ("compact_logit", "blend_logit_tail"):
            entry["features"] = COMPACT_PHYS_12
        clf_store[str(thr)] = entry
        det = {
            "model_selected_by_oof": which, "oof_bss_candidates": scores,
            "oof_n_pos": n_pos_oof, "oof_n_eval": int(ok.sum()),
            "platt_a": ab[0], "platt_b": ab[1],
            "roc_auc": float(roc_auc_score(yte_b, p_cal)) if 0 < yte_b.sum() < len(yte_b) else None,
            "pr_auc": float(average_precision_score(yte_b, p_cal)) if yte_b.sum() else None,
            "bss_calibrated": _bss(yte_b, p_cal),
            "bss_raw": _bss(yte_b, p_raw),
            "bss_oof": scores[which],
            "test_pos": int(yte_b.sum()), "test_n": int(len(yte_b)),
            "p_mean": float(np.mean(p_cal)), "observed_rate": float(yte_b.mean()),
            # the maximum Brier skill any monotone recalibration of this ranking
            # could reach on the test labels (diagnostic ceiling, never used to fit)
            "bss_oracle_recalibration": _bss(
                yte_b, IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                .fit(p_raw, yte_b).predict(p_raw)),
        }
        res[f"p_dst_le_{abs(thr)}"] = det
        probs[thr] = p_cal
        print(f"[apex] Dst<={thr}: model={which} ROC={det['roc_auc']} "
              f"BSS={det['bss_calibrated']:+.4f} (raw {det['bss_raw']:+.4f}, "
              f"oracle {det['bss_oracle_recalibration']:+.4f}) "
              f"pmean={det['p_mean']:.3f} obs={det['observed_rate']:.3f}", flush=True)

    # ---- conformal band for Dst_min: static vs locally normalised ----------
    ae = np.abs(mu_oof - ytr)
    ok = np.isfinite(ae) & np.isfinite(sig_tr)
    q90 = float(np.quantile(ae[ok], 0.90))
    zn = ae[ok] / np.clip(sig_tr[ok], 1e-6, None)
    q90_norm = float(np.quantile(zn, 0.90))
    cov_static = float(np.mean(ae[ok] <= q90))
    cov_norm = float(np.mean(ae[ok] <= q90_norm * np.clip(sig_tr[ok], 1e-6, None)))
    w_static = float(2 * q90)
    w_norm = float(np.mean(2 * q90_norm * np.clip(sig_tr[ok], 1e-6, None)))
    BAND_FLOOR = 0.88
    cands = {"static": (cov_static, w_static), "normalized": (cov_norm, w_norm)}
    eligible = {k: v[0] >= BAND_FLOOR for k, v in cands.items()}
    band_mode = (min((k for k in cands if eligible[k]), key=lambda k: cands[k][1])
                 if any(eligible.values()) else "static")
    test = test.copy()
    test["dst_min_pred"] = mu_te
    if band_mode == "normalized":
        half = q90_norm * sig_te
    else:
        half = np.full(len(mu_te), q90)
    test["dst_min_band"] = [[float(m - h), float(m + h)] for m, h in zip(mu_te, half)]
    test["dst_min_covered"] = ((yte >= mu_te - half) & (yte <= mu_te + half)).astype(int)
    res["dst_min_band"] = {"band_mode": band_mode, "q90_h": q90,
                           "q90_norm": q90_norm,
                           "oof": {"static": {"coverage": cov_static, "width_nt": w_static},
                                   "normalized": {"coverage": cov_norm, "width_nt": w_norm}},
                           "coverage_test": float(test["dst_min_covered"].mean()),
                           "mean_width_nt": float(np.mean(2 * half))}
    print(f"[apex] Dst_min band -> {band_mode}: static q90={q90:.1f} nT "
          f"(OOF cov={cov_static:.3f} w={w_static:.1f}) | norm q90={q90_norm:.2f} "
          f"(OOF cov={cov_norm:.3f} w={w_norm:.1f}) | test coverage="
          f"{res['dst_min_band']['coverage_test']:.3f}", flush=True)

    # ---- B_mag_max conformal band ----------------------------------------
    if "b_mag_max_obs" in df and df["b_mag_max_obs"].notna().sum() > 100:
        yb = train["b_mag_max_obs"].values.astype(float)
        okb = np.isfinite(yb)
        if okb.sum() > 80:
            Xb = Xtr[okb].reset_index(drop=True)
            yb2 = yb[okb]
            # Huber delta for B_mag is chosen on pre-2020 OOF, exactly like mu.
            bmag_scores, bmag_oof = {}, {}
            for a in (1.0, 5.0, 10.0):
                oof_a = np.full(len(Xb), np.nan)
                for tr_i, va_i in TimeSeriesSplit(n_splits=5).split(Xb):
                    oof_a[va_i] = mk_reg(11, objective="huber", alpha=a).fit(
                        Xb.iloc[tr_i], yb2[tr_i]).predict(Xb.iloc[va_i])
                bmag_oof[a] = oof_a
                bmag_scores[a] = float(np.nanmean(np.abs(oof_a - yb2)))
            bmag_alpha = min(bmag_scores, key=bmag_scores.get)
            mb_oof = bmag_oof[bmag_alpha]
            mb = mk_reg(11, objective="huber", alpha=bmag_alpha).fit(Xb, yb2)
            qb = float(np.nanquantile(np.abs(mb_oof - yb2), 0.90))
            mb_te = mb.predict(Xte)
            res["b_mag_max_band"] = {"q90_nt": qb,
                                     "huber_alpha": bmag_alpha,
                                     "oof_mae_nt": bmag_scores,
                                     "median_pred_nt": float(np.median(mb_te))}
            joblib.dump({"model": mb, "q90": qb}, MODELS / "apex_bmag.joblib")
            print(f"[apex] B_mag_max band: q90={qb:.1f} nT", flush=True)

    # ---- Bz risk tag ------------------------------------------------------
    def bz_tag(row) -> str:
        r = row.get("sharp_r_value")
        hem = np.sign(row.get("src_lat") or 0.0)
        v = row.get("v_lin") or 0.0
        if np.isfinite(r) and hem != 0:
            # `sharp_r_value` is SHARP `R_VALUE`, the *unsigned* flux near
            # polarity inversion lines (Schrijver 2007): it is never negative,
            # so it carries no Hale-polarity sign.  With the catalogue at hand
            # this branch only identifies a southern source with SHARP coverage;
            # the tag says exactly that and does not claim a measured helicity.
            if hem < 0 and r > 0:
                return "SOUTHERN_SOURCE"
        if v >= 900:
            return "SHEATH_DRIVEN"
        return "POLARITY_UNCERTAIN_AWAIT_L1"

    test["bz_risk_tag"] = test.apply(bz_tag, axis=1)
    test["p_dst_le_50"] = probs[-50]
    test["p_dst_le_100"] = probs[-100]
    test["p_dst_le_200"] = probs[-200]
    test[["cme_id", "t_launch", "dst_min_target", "dst_min_pred", "dst_min_band",
          "p_dst_le_50", "p_dst_le_100", "p_dst_le_200", "bz_risk_tag",
          "v_lin", "width", "src_lon", "src_lat"]].to_parquet(
        RESULTS / "v3_apex_test_predictions.parquet", index=False)

    joblib.dump({"mu_model": mu_model, "sigma": sigma, "q90_dst": q90,
                 "q90_norm": q90_norm, "band_mode": band_mode,
                 "features": APEX_SELECTED,
                 "sig_model": sig_model, "z_oof": z_oof[okr],
                 "classifiers": clf_store,
                 "platt": {str(t): res[f"p_dst_le_{abs(t)}"]["model_selected_by_oof"]
                           for t in THRESHOLDS},
                 "platt_ab": {str(t): [res[f"p_dst_le_{abs(t)}"]["platt_a"],
                                       res[f"p_dst_le_{abs(t)}"]["platt_b"]]
                              for t in THRESHOLDS},
                 "selected": {str(t): res[f"p_dst_le_{abs(t)}"]["model_selected_by_oof"]
                              for t in THRESHOLDS}},
                MODELS / "apex_v3.joblib")
    (RESULTS / "v3_apex_benchmark.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k.startswith(("p_dst", "dst_min_band", "b_mag"))},
                     indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
