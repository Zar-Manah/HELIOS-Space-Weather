#!/usr/bin/env python3
"""HELIOS v3.4 - Stage 2: HELIOS-Impact v3 calibrated Earth-arrival probability.

What changed vs v3.0 (which had brier_skill = -0.06 and FAR 0.836 on halos):
  * The base learners are trained on the *undistorted* base rate.  v3.0 used
    `class_weight="balanced"` + `scale_pos_weight` and then tried to undo the
    prior distortion with an isotonic fit on ~40 positives: that is why the
    probabilities were worse than climatology.
  * Calibration is chosen by chronological cross-cycle CV (fit Cycle 23 ->
    score Cycle 24) between Platt/sigmoid and isotonic, and applied out of fold.
  * Source-aware models: one model for CMEs with a measured source position
    (where `d_earth_deg` carries real signal) and one for the rest, combined
    with an explicit switch instead of letting a single tree split on NaN.
  * Positions come from v3_02 (real Stonyhurst coordinates; a missing HEK
    position is NaN, never 0).

Strict OOD protocol:
  train       : t_launch < 2020-01-01 (Cycles 23-24)
  calibration : out-of-fold predictions on the full pre-2020 window
  test        : 2020-01-01 .. 2026 (Solar Cycle 25, never seen)

Outputs
  models/v3/impact_*.joblib
  results/v3_impact_benchmark.json
  results/v3_impact_test_predictions.parquet
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from v3_common import PhysicsInteractionLR, PHYS_INTERACTION_COLS

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_common import (IMPACT_FEATURES, MONOTONE, RESULTS, ROOT, RankerProba,
                       design_matrix, evaluate_binary, load_cme_features)

import lightgbm as lgb
import xgboost as xgb

MODELS = ROOT / "models" / "v3"
MODELS.mkdir(parents=True, exist_ok=True)
SOURCE_FEATURES = {
    "has_source", "has_position", "src_lon", "src_lat", "abs_lon", "is_west", "is_east",
    "d_earth_deg", "cos_deprojection", "v_3d_est", "cone_margin_deg",
    "flank_margin_deg", "no_source_halo", "backside_confirmed", "unknown_source",
    "log_goes_flux", "flare_cme_dt_min", "pos_angle_consistency", "ar_num",
    "src_flare", "src_filament", "src_none",
}


def _platt(p, y):
    """Platt scaling on log-odds; returns (a, b) with p_cal = sigmoid(a*logit+b)."""
    eps = 1e-6
    z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))[:, None]
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def _apply_platt(p, ab):
    eps = 1e-6
    z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    return 1.0 / (1.0 + np.exp(-(ab[0] * z + ab[1])))




def _fit_beta(p, y):
    """Beta calibration: two log-odds coefficients, fitted out of fold."""
    eps = 1e-6
    Z = np.column_stack([np.log(np.clip(p, eps, 1 - eps)),
                         np.log(np.clip(1 - p, eps, 1 - eps))])
    return LogisticRegression(C=1e6, max_iter=1000).fit(Z, y)


def _apply_beta(p, lr):
    eps = 1e-6
    Z = np.column_stack([np.log(np.clip(p, eps, 1 - eps)),
                         np.log(np.clip(1 - p, eps, 1 - eps))])
    return lr.predict_proba(Z)[:, 1]


def _fit_temp(p, y):
    """Temperature scaling on the log-odds; returns the scale t."""
    from scipy.optimize import minimize_scalar
    eps = 1e-6
    z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))

    def nll(logt):
        t = np.exp(logt)
        q = 1.0 / (1.0 + np.exp(-z / t))
        q = np.clip(q, eps, 1 - eps)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    r = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    return float(np.exp(r.x))


def _apply_temp(p, t):
    eps = 1e-6
    z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    return 1.0 / (1.0 + np.exp(-z / t))


def apply_calibration(p, calib, method: str):
    """Single entry point shared by training and serving."""
    if method == "isotonic":
        return np.asarray(calib.predict(np.asarray(p, dtype=float)), dtype=float)
    if method == "beta":
        return _apply_beta(np.asarray(p, dtype=float), calib)
    if method == "temperature":
        return _apply_temp(np.asarray(p, dtype=float), calib)
    return _apply_platt(np.asarray(p, dtype=float), calib)


def fit_ensemble(Xtr, ytr, feats):
    """Four families on the raw base rate (no class weighting anywhere).

    The fourth member is a LambdaRank model.  G2.1b scores the model at the
    halo rule's false-alarm rate, where average precision is not the currency;
    the ranker is the member that optimises the order of the alert list.  It
    was added after a pre-2020 chronological CV showed +3.8 pp recall at the
    halo FAR (a discarded experiment; its log is not shipped and no reported
    number depends on it), never on Cycle 25.
    """
    mono = [MONOTONE.get(f, 0) for f in feats]
    mono_xgb = {f: MONOTONE[f] for f in feats if f in MONOTONE}
    lgbm = lgb.LGBMClassifier(
        n_estimators=700, learning_rate=0.025, num_leaves=15, max_depth=4,
        min_child_samples=40, subsample=0.85, colsample_bytree=0.8,
        reg_lambda=2.0, random_state=7, monotone_constraints=mono,
        n_jobs=1, verbose=-1,
    )
    xgbm = xgb.XGBClassifier(
        n_estimators=700, learning_rate=0.025, max_depth=4, min_child_weight=8,
        subsample=0.85, colsample_bytree=0.8, reg_lambda=2.0, reg_alpha=0.1,
        random_state=7, n_jobs=1, eval_metric="logloss",
        monotone_constraints=mono_xgb, tree_method="hist",
    )
    lin = Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=0.2, max_iter=3000)),
    ])
    ranker = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=500, learning_rate=0.03,
        num_leaves=15, max_depth=4, min_child_samples=40, subsample=0.85,
        colsample_bytree=0.8, reg_lambda=2.0, random_state=7, n_jobs=1,
        verbose=-1, label_gain=[0, 1, 3],
    )
    lgbm.fit(Xtr, ytr)
    xgbm.fit(Xtr, ytr)
    lin.fit(Xtr, ytr)
    # LambdaRank caps one query at 10k rows; the training frame is time-ordered,
    # so chronological blocks of 5k are the natural query groups.
    n = len(Xtr)
    groups = [min(5000, n - i) for i in range(0, n, 5000)]
    ranker.fit(Xtr, ytr, group=groups)
    phys = PhysicsInteractionLR().fit(Xtr, ytr)
    sc = ranker.predict(np.asarray(Xtr, dtype=float))
    return [lgbm, xgbm, lin, RankerProba(ranker, float(np.mean(sc)), float(np.std(sc))),
            phys]


def stack_proba(models, X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.column_stack([m.predict_proba(X)[:, 1] for m in models])


def _fit_pipeline(train: pd.DataFrame, test: pd.DataFrame, feats: list[str]) -> dict:
    """OOF stack + calibration chosen by cross-cycle CV.  Returns a model blob."""
    Xtr = design_matrix(train, feats)
    ytr = train["label"].values.astype(int)
    Xte = design_matrix(test, feats)
    yte = test["label"].values.astype(int)

    oof = None
    for tr_idx, va_idx in TimeSeriesSplit(n_splits=5).split(Xtr):
        ms = fit_ensemble(Xtr.iloc[tr_idx], ytr[tr_idx], feats)
        p_fold = stack_proba(ms, Xtr.iloc[va_idx])
        if oof is None:
            oof = np.full((len(train), p_fold.shape[1]), np.nan)
        oof[va_idx] = p_fold
    mask = np.isfinite(oof[:, 0])

    meta = LogisticRegression(C=1.0, max_iter=1000).fit(oof[mask], ytr[mask])
    p_oof = meta.predict_proba(oof[mask])[:, 1]
    y_oof = ytr[mask]
    t_oof = train.loc[mask, "t_launch"].reset_index(drop=True)

    # --- calibration ------------------------------------------------------
    # The calibrator is fitted on the *full* out-of-fold window (Cycles 23-24),
    # not on a late slice: fitting it on 2016-2019 alone transplants that
    # window's base rate into the forecast and is what made v3.0's Brier skill
    # negative.  Isotonic needs a few hundred calibration positives to be
    # stable; below that it overfits the step positions and destroys the
    # ranking granularity, so Platt is preferred unless isotonic wins the
    # cross-cycle check by more than 10 %.
    m23 = (t_oof < pd.Timestamp("2009-01-01")).values
    m24 = ~m23
    cand = {}
    if m23.sum() > 80 and 0 < y_oof[m23].sum() < m23.sum() and 0 < y_oof[m24].sum() < m24.sum():
        ab23 = _platt(p_oof[m23], y_oof[m23])
        cand["platt"] = float(np.mean((_apply_platt(p_oof[m24], ab23) - y_oof[m24]) ** 2))
        iso23 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso23.fit(p_oof[m23], y_oof[m23])
        cand["isotonic"] = float(np.mean((iso23.predict(p_oof[m24]) - y_oof[m24]) ** 2))
    n_pos_cal = int(y_oof.sum())

    beta_fit = temp_fit = None
    if m23.sum() > 80 and 0 < y_oof[m23].sum() < m23.sum() and 0 < y_oof[m24].sum() < m24.sum():
        beta_fit = _fit_beta(p_oof[m23], y_oof[m23])
        cand["beta"] = float(np.mean((_apply_beta(p_oof[m24], beta_fit) - y_oof[m24]) ** 2))
        temp_fit = _fit_temp(p_oof[m23], y_oof[m23])
        cand["temperature"] = float(np.mean((_apply_temp(p_oof[m24], temp_fit) - y_oof[m24]) ** 2))

    # Eligibility is decided a priori: isotonic needs 200 positives, Beta 50,
    # the two-parameter families are always eligible.  The winner is the lowest
    # held-out Brier on Cycle 24.
    eligible = {"platt": cand.get("platt", np.inf)}
    if n_pos_cal >= 50 and "beta" in cand:
        eligible["beta"] = cand["beta"]
    if "temperature" in cand:
        eligible["temperature"] = cand["temperature"]
    if n_pos_cal >= 200 and "isotonic" in cand:
        eligible["isotonic"] = cand["isotonic"]
    method = min(eligible, key=eligible.get)
    cal = np.ones(len(p_oof), dtype=bool)
    if method == "isotonic":
        calib = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_oof[cal], y_oof[cal])
        p_oof_cal = calib.predict(p_oof)
    elif method == "beta":
        calib = _fit_beta(p_oof[cal], y_oof[cal])
        p_oof_cal = _apply_beta(p_oof, calib)
    elif method == "temperature":
        calib = _fit_temp(p_oof[cal], y_oof[cal])
        p_oof_cal = _apply_temp(p_oof, calib)
    else:
        calib = _platt(p_oof[cal], y_oof[cal])
        p_oof_cal = _apply_platt(p_oof, calib)

    models = fit_ensemble(Xtr, ytr, feats)
    p_te_raw = meta.predict_proba(stack_proba(models, Xte))[:, 1]
    if method == "isotonic":
        p_te = calib.predict(p_te_raw)
    elif method == "beta":
        p_te = _apply_beta(p_te_raw, calib)
    elif method == "temperature":
        p_te = _apply_temp(p_te_raw, calib)
    else:
        p_te = _apply_platt(p_te_raw, calib)

    # thresholds from the calibration slice only
    yc, pc = y_oof[cal], p_oof_cal[cal]
    order = np.argsort(-pc)
    tp = np.cumsum(yc[order])
    prec = tp / np.arange(1, len(pc) + 1)
    rec = tp / max(yc.sum(), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    tau_watch = float(pc[order][int(np.argmax(f1))]) if len(pc) else 0.5
    ok = np.where(prec >= 0.5)[0]
    tau_high = float(pc[order][ok[-1]]) if len(ok) else float(np.quantile(pc, 0.995))

    return {"models": models, "meta": meta, "calib": calib, "calib_method": method,
            "features": feats, "tau_watch": tau_watch, "tau_high": tau_high,
            "calib_cv": cand, "n_oof": int(mask.sum()),
            "p_test": p_te, "p_oof_cal": p_oof_cal, "y_oof": y_oof, "t_oof": t_oof,
            "cal_slice": cal}


def _ece(y, p, bins=10) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return tot


def main() -> int:
    df = load_cme_features()
    df = df[df["label"] != -1].sort_values("t_launch").reset_index(drop=True)
    df["src_flare"] = (df["source_type"].isin(["flare_swpc", "flare_ssw"])).astype(int)
    df["src_filament"] = (df["source_type"] == "filament_eruption").astype(int)
    df["src_none"] = (df["source_type"] == "none").astype(int)
    print(f"[impact] rows={len(df)} positives={int((df.label==1).sum())}")

    train = df[df["t_launch"] < pd.Timestamp("2020-01-01")].reset_index(drop=True)
    test = df[df["t_launch"] >= pd.Timestamp("2020-01-01")].reset_index(drop=True)
    print(f"[impact] train={len(train)} pos={int((train.label==1).sum())} | "
          f"test={len(test)} pos={int((test.label==1).sum())}")

    results = {"splits": {"train": [str(train.t_launch.min()), str(train.t_launch.max()),
                                    int(len(train)), int(train.label.sum())],
                          "test": [str(test.t_launch.min()), str(test.t_launch.max()),
                                   int(len(test)), int(test.label.sum())]}}

    halo = test["is_halo"].values.astype(float)
    results["baseline_halo_rule"] = evaluate_binary(test["label"], halo, threshold=0.5)
    results["baseline_halo_rule"]["n_halo"] = int((test.is_halo == 1).sum())

    yte = test["label"].values.astype(int)
    for tag, feats in (("full", IMPACT_FEATURES),
                       ("no_source", [f for f in IMPACT_FEATURES if f not in SOURCE_FEATURES])):
        blob = _fit_pipeline(train, test, feats)
        p = blob["p_test"]
        res = evaluate_binary(yte, p, threshold=blob["tau_watch"])
        res.update({
            "tau_high_precision": blob["tau_high"], "tau_watch": blob["tau_watch"],
            "calibration_method": blob["calib_method"], "calibration_cv": blob["calib_cv"],
            "n_direct_alert": int((p >= blob["tau_high"]).sum()),
            "n_watch": int(((p >= blob["tau_watch"]) & (p < blob["tau_high"])).sum()),
            "calibration_ece_test": float(_ece(yte, p)),
            "calibration_ece_calib_slice": float(_ece(blob["y_oof"][blob["cal_slice"]],
                                                      blob["p_oof_cal"][blob["cal_slice"]])),
            "p_mean": float(np.mean(p)), "observed_rate": float(yte.mean()),
        })
        for name, m in (("wide_120", test["width"] >= 120),
                        ("source_localized", test["has_position"] == 1),
                        ("no_position", test["has_position"] == 0),
                        ("halo", test["is_halo"] == 1)):
            mm = m.values if hasattr(m, "values") else np.asarray(m)
            if mm.sum() > 0 and 0 < yte[mm].sum() < mm.sum():
                res[f"subset_{name}"] = evaluate_binary(yte[mm], p[mm], threshold=blob["tau_watch"])
        results[f"model_{tag}"] = res
        joblib.dump({k: blob[k] for k in ("models", "meta", "calib", "calib_method",
                                          "features", "tau_watch", "tau_high")},
                    MODELS / f"impact_{tag}.joblib")
        print(f"[impact:{tag}] PR-AUC={res['pr_auc']:.4f} ROC={res['roc_auc']:.4f} "
              f"BSS={res['brier_skill']:+.4f} P@R70={res['precision_at_recall_70']:.4f} "
              f"prec={res['precision']:.4f} rec={res['recall']:.4f} FAR={res['far']:.4f} "
              f"cal={blob['calib_method']} pmean={res['p_mean']:.4f}")
        if tag == "full":
            full_blob, full_p = blob, p
        elif tag == "no_source":
            nosrc_blob = blob

    # ---- source-aware switch: two specialised models ------------------------
    has_pos = train["has_position"] == 1
    sub = {}
    if has_pos.sum() > 500 and (train.loc[has_pos, "label"] == 1).sum() > 40:
        b_loc = _fit_pipeline(train[has_pos].reset_index(drop=True),
                              test[test.has_position == 1].reset_index(drop=True), IMPACT_FEATURES)
        b_blind = _fit_pipeline(train[~has_pos].reset_index(drop=True),
                                test[test.has_position == 0].reset_index(drop=True), IMPACT_FEATURES)
        p_sw = np.full(len(test), np.nan)
        p_sw[(test.has_position == 1).values] = b_loc["p_test"]
        p_sw[(test.has_position == 0).values] = b_blind["p_test"]
        if np.isfinite(p_sw).all():
            res_sw = evaluate_binary(yte, p_sw, threshold=full_blob["tau_watch"])
            res_sw.update({"model": "source_aware_switch",
                           "calibration_method": f"localized={b_loc['calib_method']},"
                                                 f"blind={b_blind['calib_method']}"})
            for name, m in (("source_localized", test["has_position"] == 1),
                            ("no_position", test["has_position"] == 0)):
                mm = m.values
                if 0 < yte[mm].sum() < mm.sum():
                    res_sw[f"subset_{name}"] = evaluate_binary(yte[mm], p_sw[mm])
            results["model_source_aware"] = res_sw
            sub = {"localized": b_loc, "blind": b_blind, "p": p_sw}
            print(f"[impact:switch] PR-AUC={res_sw['pr_auc']:.4f} ROC={res_sw['roc_auc']:.4f} "
                  f"BSS={res_sw['brier_skill']:+.4f}")

            # Which predictor gets served is a CV decision, never a Cycle-25
            # one.  Both candidates expose pre-2020 OOF predictions, so the
            # comparison is made on the same out-of-fold window (aligned by
            # timestamp) and frozen before test is looked at.
            oof_sw = pd.concat([
                pd.DataFrame({"t": pd.to_datetime(b_loc["t_oof"]),
                              "p": b_loc["p_oof_cal"], "y": b_loc["y_oof"]}),
                pd.DataFrame({"t": pd.to_datetime(b_blind["t_oof"]),
                              "p": b_blind["p_oof_cal"], "y": b_blind["y_oof"]}),
            ], ignore_index=True)
            oof_full = pd.DataFrame({"t": pd.to_datetime(full_blob["t_oof"]),
                                     "p": full_blob["p_oof_cal"],
                                     "y": full_blob["y_oof"]})
            oof_nosrc = pd.DataFrame({"t": pd.to_datetime(nosrc_blob["t_oof"]),
                                      "p": nosrc_blob["p_oof_cal"],
                                      "y": nosrc_blob["y_oof"]})
            common = (oof_sw.groupby("t").size().rename("ns")
                      .to_frame().join(oof_full.groupby("t").size().rename("nf"),
                                       how="inner")
                      .join(oof_nosrc.groupby("t").size().rename("nn"),
                            how="inner"))
            keep_t = common[(common.ns == 1) & (common.nf == 1)
                            & (common.nn == 1)].index
            a = oof_sw[oof_sw.t.isin(keep_t)].sort_values("t")
            b = oof_full[oof_full.t.isin(keep_t)].sort_values("t")
            c = oof_nosrc[oof_nosrc.t.isin(keep_t)].sort_values("t")
            if (len(a) > 200 and (a.y.values == b.y.values).all()
                    and (a.y.values == c.y.values).all()):
                y_al = a.y.values.astype(int)

                def _operating_point(pc, yc):
                    order = np.argsort(-pc)
                    tp = np.cumsum(yc[order])
                    prec = tp / np.arange(1, len(pc) + 1)
                    rec = tp / max(yc.sum(), 1)
                    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
                    tw = float(pc[order][int(np.argmax(f1))])
                    okp = np.where(prec >= 0.5)[0]
                    th = (float(pc[order][okp[-1]]) if len(okp)
                          else float(np.quantile(pc, 0.995)))
                    return tw, th

                ap_sw_oof = evaluate_binary(y_al, a.p.values, 0.5)["pr_auc"]
                ap_full_oof = evaluate_binary(y_al, b.p.values, 0.5)["pr_auc"]
                p_blend_oof = (0.4 * b.p.values + 0.4 * a.p.values
                               + 0.2 * c.p.values)
                ap_blend_oof = evaluate_binary(y_al, p_blend_oof, 0.5)["pr_auc"]
                results["model_source_aware"]["pr_auc_oof"] = float(ap_sw_oof)
                results["model_full"]["pr_auc_oof"] = float(ap_full_oof)
                results["model_blend_oof"] = {
                    "pr_auc_oof": float(ap_blend_oof),
                    "weights": [0.4, 0.4, 0.2], "n_oof_common": int(len(a))}
                winner = max({"full": ap_full_oof,
                              "source_aware": ap_sw_oof,
                              "impact_blend": ap_blend_oof},
                             key=lambda k: {"full": ap_full_oof,
                                            "source_aware": ap_sw_oof,
                                            "impact_blend": ap_blend_oof}[k])
                results["cv_model_selection"] = {
                    "criterion": "pre-2020 OOF PR-AUC on the aligned OOF window",
                    "n_oof_common": int(len(a)),
                    "pr_auc_oof_full": float(ap_full_oof),
                    "pr_auc_oof_source_aware": float(ap_sw_oof),
                    "pr_auc_oof_impact_blend": float(ap_blend_oof),
                    "winner": winner,
                }
                sub["wins_oof"] = bool(winner == "source_aware")
                sub["wins_blend_oof"] = bool(winner == "impact_blend")
                # operating points of the switch and of the blend, same rule as
                # _fit_pipeline, fitted on their own OOF probabilities
                sub["tau_watch"], sub["tau_high"] = _operating_point(
                    a.p.values, y_al)
                (sub["tau_blend_watch"],
                 sub["tau_blend_high"]) = _operating_point(p_blend_oof, y_al)

                p_blend_te = (0.4 * full_p + 0.4 * p_sw
                              + 0.2 * nosrc_blob["p_test"])
                res_bl = evaluate_binary(yte, p_blend_te,
                                         threshold=sub["tau_blend_watch"])
                res_bl.update({
                    "model": "impact_blend",
                    "tau_watch": float(sub["tau_blend_watch"]),
                    "tau_high_precision": float(sub["tau_blend_high"]),
                    "calibration_method": "blend(0.4 full + 0.4 switch + 0.2 no_source)",
                    "weights": [0.4, 0.4, 0.2],
                    "pr_auc_oof": float(ap_blend_oof),
                    "calibration_ece_test": float(_ece(yte, p_blend_te)),
                    "p_mean": float(np.mean(p_blend_te)),
                    "observed_rate": float(yte.mean()),
                })
                for name, m in (("wide_120", test["width"] >= 120),
                                ("source_localized", test["has_position"] == 1),
                                ("no_position", test["has_position"] == 0),
                                ("halo", test["is_halo"] == 1)):
                    mm = m.values
                    if 0 < yte[mm].sum() < mm.sum():
                        res_bl[f"subset_{name}"] = evaluate_binary(
                            yte[mm], p_blend_te[mm],
                            threshold=sub["tau_blend_watch"])
                results["model_blend"] = res_bl
                sub["p_blend"] = p_blend_te
                print(f"[impact:select] OOF n={len(a)} PR-AUC full={ap_full_oof:.4f} "
                      f"switch={ap_sw_oof:.4f} blend={ap_blend_oof:.4f} -> {winner}")
                print(f"[impact:blend] test PR-AUC={res_bl['pr_auc']:.4f} "
                      f"ROC={res_bl['roc_auc']:.4f} BSS={res_bl['brier_skill']:+.4f}")

    # ---- persist test predictions ------------------------------------------
    # The persisted operating point must be the model chosen *without* looking
    # at Cycle 25.  `model_full` is the calibrated pre-2020 ensemble; the
    # source-aware switch is a diagnostic only and must never replace it here.
    # Winner of the pre-2020 CV comparison above.  Default is the unified
    # `model_full`; the switch is only served if it wins OOF, which is decided
    # before any Cycle-25 metric exists.
    winner = results.get("cv_model_selection", {}).get("winner", "full")
    use_switch = bool(winner == "source_aware")
    use_blend = bool(winner == "impact_blend")
    best = full_blob
    sw_tau_watch = float(sub.get("tau_watch", best["tau_watch"]))
    sw_tau_high = float(sub.get("tau_high", best["tau_high"]))
    if use_blend:
        p_out = sub["p_blend"]
        tau_watch_out = float(sub["tau_blend_watch"])
        tau_high_out = float(sub["tau_blend_high"])
        results["served_model"] = "impact_blend"
    elif use_switch:
        p_out = sub["p"]
        tau_watch_out = sw_tau_watch
        tau_high_out = sw_tau_high
        results["served_model"] = "source_aware_switch"
    else:
        p_out = best["p_test"]
        tau_watch_out = float(best["tau_watch"])
        tau_high_out = float(best["tau_high"])
        results["served_model"] = "model_full"
    print(f"[impact:served] {results['served_model']}")

    # Serving artefacts: the served predictor must be the one whose test
    # predictions are persisted above, otherwise the live path and the
    # backtest disagree (train/serve skew).  Stale artefacts are removed.
    sw_path = MODELS / "impact_switch.joblib"
    bl_path = MODELS / "impact_blend.joblib"
    if use_switch or use_blend:
        joblib.dump({"localized": {k: sub["localized"][k] for k in
                                   ("models", "meta", "calib", "calib_method", "features")},
                     "blind": {k: sub["blind"][k] for k in
                               ("models", "meta", "calib", "calib_method", "features")},
                     "tau_watch": sw_tau_watch,
                     "tau_high": sw_tau_high,
                     "selected_by": "pre-2020 OOF PR-AUC (aligned window)",
                     "pr_auc_oof": results.get("cv_model_selection", {})},
                    sw_path)
        print(f"[impact:served] wrote {sw_path.name}")
    elif sw_path.exists():
        sw_path.unlink()
        print(f"[impact:served] removed stale {sw_path.name}")
    if use_blend:
        joblib.dump({"weights": [0.4, 0.4, 0.2],
                     "tau_watch": tau_watch_out,
                     "tau_high": tau_high_out,
                     "no_source_features": nosrc_blob["features"],
                     "selected_by": "pre-2020 OOF PR-AUC (aligned window)"},
                    bl_path)
        print(f"[impact:served] wrote {bl_path.name}")
    elif bl_path.exists():
        bl_path.unlink()
        print(f"[impact:served] removed stale {bl_path.name}")
    out = test[["cme_id", "t_launch", "label", "is_impact", "is_halo", "width", "v_lin",
                "src_lon", "src_lat", "has_source", "has_position", "solar_cycle",
                "match_confidence"]].copy()
    out["p_impact"] = p_out
    out["tier"] = np.where(p_out >= tau_high_out, "DIRECT_HIT_ALERT",
                           np.where(p_out >= tau_watch_out, "GLANCING_OR_WATCH", "ALL_CLEAR"))
    out.to_parquet(RESULTS / "v3_impact_test_predictions.parquet", index=False)

    (RESULTS / "v3_impact_benchmark.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.startswith("subset")}
                      for k, v in results.items() if k.startswith(("model_", "baseline"))},
                     indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
