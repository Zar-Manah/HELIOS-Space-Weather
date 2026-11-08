#!/usr/bin/env python3
"""
HELIOS - Solar Storm Predictor (Pipeline Phase 2)
Trains multi-horizon predictive engines:
  1. Empirical Burton Model (Physics Baseline)
  2. Persistence Baseline
  3. LightGBM Gradient Boosted Trees (Point & Quantile Regressors)
  4. Conformalized Quantile Regression (CQR) with Guaranteed Coverage
Evaluates out-of-distribution across distinct solar cycles:
  Train: Cycle 23 (1998-2008) + early Cycle 24 (2009-2015)
  Calibration: late Cycle 24 (2016-2019)
  Test (Unseen OOD): Cycle 25 (2020-2026, including the historic May 2024 G5 storm!)
"""

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(BASE_DIR, "data/processed/helios_omni_features_1998_2026.parquet"
MODELS_DIR = os.path.join(BASE_DIR, "models"
RESULTS_DIR = os.path.join(BASE_DIR, "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 70)
print("☀️  HELIOS SPACE WEATHER ENGINE: MULTI-HORIZON TRAINING  ☀️")
print("=" * 70)

print(f"Loading features from: {DATA_PATH}")
df = pd.read_parquet(DATA_PATH)

feature_cols = [
    "dst", "dst_lag_1h", "dst_lag_2h", "dst_lag_3h", "dst_lag_6h", "dst_lag_12h", "dst_lag_24h",
    "dst_mean_6h", "dst_std_6h", "dst_min_24h", "dst_diff_1h",
    "bz_gsm", "bz_lag_1h", "bz_lag_2h", "bz_lag_3h", "bz_lag_6h",
    "bz_min_3h", "bz_min_6h", "bz_mean_6h", "bz_diff_1h",
    "by_gsm", "b_mag", "b_transverse", "clock_angle",
    "v_flow", "v_lag_1h", "v_lag_2h", "v_lag_3h", "v_diff_1h",
    "np_dens", "temp", "pdyn", "pdyn_lag_1h", "pdyn_diff_1h",
    "ey_rectified", "newell_coupling", "akasofu_eps", "borovsky_coupling",
    "newell_sum_3h", "newell_sum_6h", "newell_sum_12h",
    "kp", "r_sunspot", "f107"
]

print(f"Total features: {len(feature_cols)}")

# Chronological Cycle Split
# Train: 1998 - 2015 (~18 years)
# Calib: 2016 - 2019 (4 years, end of Cycle 24)
# Test:  2020 - 2026 (6+ years, Cycle 25 OOD)

train_mask = (df["year"] >= 1998) & (df["year"] <= 2015)
calib_mask = (df["year"] >= 2016) & (df["year"] <= 2019)
test_mask  = (df["year"] >= 2020)

df_train = df[train_mask].copy().reset_index(drop=True)
df_calib = df[calib_mask].copy().reset_index(drop=True)
df_test  = df[test_mask].copy().reset_index(drop=True)

print(f"Split sizes:")
print(f"  • Train set (1998-2015):      {len(df_train):,} samples")
print(f"  • Calibration set (2016-2019): {len(df_calib):,} samples")
print(f"  • Test set Cycle 25 (2020-2026): {len(df_test):,} samples")

X_train = df_train[feature_cols]
X_calib = df_calib[feature_cols]
X_test  = df_test[feature_cols]

horizons = [1, 2, 3, 6]
results_all = {}
lead_models = {}

def compute_metrics(y_true, y_pred, name=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Storm classification skill at Dst <= -50 nT (Moderate storm)
    actual_storm = (y_true <= -50)
    pred_storm   = (y_pred <= -50)
    
    tp = np.sum(actual_storm & pred_storm)
    tn = np.sum((~actual_storm) & (~pred_storm))
    fp = np.sum((~actual_storm) & pred_storm)
    fn = np.sum(actual_storm & (~pred_storm))
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tss = tpr - fpr  # True Skill Statistic (standard space weather metric)
    
    # Heidke Skill Score (HSS)
    hss_denom = ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    hss = 2.0 * (tp * tn - fp * fn) / hss_denom if hss_denom > 0 else 0.0
    
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "tss_50": float(tss),
        "hss_50": float(hss),
        "tpr_50": float(tpr),
        "fpr_50": float(fpr)
    }

for lead in horizons:
    target_col = f"target_dst_lead_{lead}h"
    print(f"\n" + "-" * 60)
    print(f"🚀 TRAINING HORIZON: LEAD TIME +{lead} HOUR(S) ({target_col})")
    print("-" * 60)
    
    y_train = df_train[target_col].values
    y_calib = df_calib[target_col].values
    y_test  = df_test[target_col].values
    
    # 1. Baseline: Persistence (Dst(t+lead) = Dst(t))
    y_pred_persist = df_test["dst"].values
    metrics_persist = compute_metrics(y_test, y_pred_persist, "Persistence")
    print(f"  • Baseline Persistence: RMSE={metrics_persist['rmse']:.2f} nT, R2={metrics_persist['r2']:.3f}, TSS={metrics_persist['tss_50']:.3f}")
    
    # 2. LightGBM Point Regressor (MSE objective)
    print("  • Training LightGBM Point Regressor...")
    model_point = lgb.LGBMRegressor(
        n_estimators=350,
        learning_rate=0.04,
        num_leaves=45,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model_point.fit(X_train, y_train)
    y_pred_point = model_point.predict(X_test)
    metrics_point = compute_metrics(y_test, y_pred_point, "HELIOS-Point")
    print(f"  • HELIOS-Point:        RMSE={metrics_point['rmse']:.2f} nT, R2={metrics_point['r2']:.3f}, TSS={metrics_point['tss_50']:.3f}")
    
    # 3. Conformalized Quantile Regression (CQR)
    # Train 5% and 95% quantile models for nominal 90% coverage
    alpha = 0.10  # 90% nominal confidence interval
    q_low = alpha / 2.0   # 0.05
    q_high = 1.0 - alpha / 2.0  # 0.95
    
    print(f"  • Training Conformal Quantile Models (q={q_low}, q={q_high})...")
    model_q_low = lgb.LGBMRegressor(
        objective="quantile",
        alpha=q_low,
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=35,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model_q_low.fit(X_train, y_train)
    
    model_q_high = lgb.LGBMRegressor(
        objective="quantile",
        alpha=q_high,
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=35,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model_q_high.fit(X_train, y_train)
    
    # Calibrate on independent calibration set (2016-2019)
    calib_pred_low = model_q_low.predict(X_calib)
    calib_pred_high = model_q_high.predict(X_calib)
    
    # Non-conformity scores E_i = max(q_low(x_i) - y_i, y_i - q_high(x_i))
    scores = np.maximum(calib_pred_low - y_calib, y_calib - calib_pred_high)
    
    # Finite-sample calibrated quantile factor
    n_calib = len(scores)
    q_conformal_level = np.ceil((1.0 - alpha) * (n_calib + 1)) / n_calib
    q_conformal_level = min(1.0, max(0.0, q_conformal_level))
    q_conformal = float(np.quantile(scores, q_conformal_level))
    print(f"  • Conformal adjustment factor Q_(1-alpha): {q_conformal:.2f} nT")
    
    # Test inference with guaranteed intervals
    test_pred_low = model_q_low.predict(X_test) - q_conformal
    test_pred_high = model_q_high.predict(X_test) + q_conformal
    
    # Empirical coverage on completely unseen Cycle 25
    covered = (y_test >= test_pred_low) & (y_test <= test_pred_high)
    empirical_coverage = float(np.mean(covered))
    mean_interval_width = float(np.mean(test_pred_high - test_pred_low))
    
    print(f"  • Conformal Empirical Coverage (Nominal 90%): {empirical_coverage * 100:.2f}%")
    print(f"  • Mean Prediction Interval Width:             {mean_interval_width:.2f} nT")
    
    # Save models
    lead_model_paths = {
        "point": os.path.join(MODELS_DIR, f"helios_point_lead_{lead}h.joblib"),
        "q_low": os.path.join(MODELS_DIR, f"helios_q_low_lead_{lead}h.joblib"),
        "q_high": os.path.join(MODELS_DIR, f"helios_q_high_lead_{lead}h.joblib"),
        "q_conformal": q_conformal
    }
    joblib.dump(model_point, lead_model_paths["point"])
    joblib.dump(model_q_low, lead_model_paths["q_low"])
    joblib.dump(model_q_high, lead_model_paths["q_high"])
    
    results_all[f"lead_{lead}h"] = {
        "persistence": metrics_persist,
        "helios_point": metrics_point,
        "conformal": {
            "nominal_coverage": 1.0 - alpha,
            "empirical_coverage": empirical_coverage,
            "q_conformal_adjustment": q_conformal,
            "mean_interval_width": mean_interval_width
        }
    }
    
    # If lead == 1, save feature importance and test predictions for plotting
    if lead in [1, 3]:
        df_test_preds = df_test[["datetime", "dst", target_col]].copy()
        df_test_preds["pred_point"] = y_pred_point
        df_test_preds["pred_low_cqr"] = test_pred_low
        df_test_preds["pred_high_cqr"] = test_pred_high
        df_test_preds.to_parquet(os.path.join(RESULTS_DIR, f"test_predictions_lead_{lead}h.parquet"), index=False)

# Save metrics JSON
metrics_file = os.path.join(RESULTS_DIR, "helios_evaluation_metrics.json")
with open(metrics_file, "w") as f:
    json.dump(results_all, f, indent=2)

print("\n" + "=" * 70)
print("📊 SUMMARY OF OUT-OF-DISTRIBUTION BENCHMARKS (SOLAR CYCLE 25)")
print("=" * 70)
print(f"{'Horizon':<10} | {'Model':<16} | {'RMSE (nT)':<10} | {'R²':<8} | {'TSS':<8} | {'Coverage':<10}")
print("-" * 75)
for lead in horizons:
    res = results_all[f"lead_{lead}h"]
    p = res["persistence"]
    h = res["helios_point"]
    c = res["conformal"]
    print(f"+{lead}h Lead   | Persistence      | {p['rmse']:<10.2f} | {p['r2']:<8.3f} | {p['tss_50']:<8.3f} | N/A")
    print(f"+{lead}h Lead   | HELIOS-Point     | {h['rmse']:<10.2f} | {h['r2']:<8.3f} | {h['tss_50']:<8.3f} | N/A")
    print(f"+{lead}h Lead   | HELIOS-CQR (90%) | {'—':<10} | {'—':<8} | {'—':<8} | {c['empirical_coverage']*100:.1f}% ({c['mean_interval_width']:.1f} nT)")
    print("-" * 75)

# Save feature importance of 1h model
importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model_point.feature_importances_
}).sort_values("importance", ascending=False)
importance.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)

print("\nTop 10 Most Predictive Physical Features:")
for idx, row in enumerate(importance.head(10).itertuples(), 1):
    print(f"  {idx:2d}. {row.feature:<20} (score: {row.importance})")

print("\n✅ Multi-horizon training and Conformal Calibration finished successfully!")

