#!/usr/bin/env python3
"""
Zar-Manah HELIOS 2.0 - Tactical Metrics Reproducibility Verifier
Recomputes the 1/2/3/6 h CQR coverage, interval width and point-forecast
metrics directly from the published models + published feature parquet,
without retraining. Writes test-prediction parquets for every horizon.

Usage:
    python3 scripts/20_verify_tactical_metrics.py            # report only
    python3 scripts/20_verify_tactical_metrics.py --update   # refresh metrics JSON
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "helios_omni_features_1998_2026.parquet")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
METRICS_FILE = os.path.join(RESULTS_DIR, "helios_evaluation_metrics.json")

FEATURE_COLS = [
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

HORIZONS = [1, 2, 3, 6]
ALPHA = 0.10  # 90% nominal coverage


def compute_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    actual_storm = y_true <= -50
    pred_storm = y_pred <= -50
    tp = int(np.sum(actual_storm & pred_storm))
    tn = int(np.sum(~actual_storm & ~pred_storm))
    fp = int(np.sum(~actual_storm & pred_storm))
    fn = int(np.sum(actual_storm & ~pred_storm))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    hss_denom = ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    hss = 2.0 * (tp * tn - fp * fn) / hss_denom if hss_denom > 0 else 0.0
    return {
        "rmse": rmse, "mae": mae, "r2": r2,
        "tss_50": float(tpr - fpr), "hss_50": float(hss),
        "tpr_50": float(tpr), "fpr_50": float(fpr),
    }


def main():
    update = "--update" in sys.argv

    df = pd.read_parquet(DATA_PATH)
    df_train = df[(df["year"] >= 1998) & (df["year"] <= 2015)].reset_index(drop=True)
    df_calib = df[(df["year"] >= 2016) & (df["year"] <= 2019)].reset_index(drop=True)
    df_test = df[df["year"] >= 2020].reset_index(drop=True)

    print(f"Train {len(df_train):,} | Calib {len(df_calib):,} | Test {len(df_test):,}")

    verified = {}
    for lead in HORIZONS:
        target_col = f"target_dst_lead_{lead}h"
        y_calib = df_calib[target_col].values
        y_test = df_test[target_col].values

        model_point = joblib.load(os.path.join(MODELS_DIR, f"helios_point_lead_{lead}h.joblib"))
        model_q_low = joblib.load(os.path.join(MODELS_DIR, f"helios_q_low_lead_{lead}h.joblib"))
        model_q_high = joblib.load(os.path.join(MODELS_DIR, f"helios_q_high_lead_{lead}h.joblib"))

        X_calib = df_calib[FEATURE_COLS]
        X_test = df_test[FEATURE_COLS]

        calib_low = model_q_low.predict(X_calib)
        calib_high = model_q_high.predict(X_calib)
        scores = np.maximum(calib_low - y_calib, y_calib - calib_high)
        n_calib = len(scores)
        level = min(1.0, max(0.0, np.ceil((1.0 - ALPHA) * (n_calib + 1)) / n_calib))
        q_conformal = float(np.quantile(scores, level))

        pred_point = model_point.predict(X_test)
        raw_low = model_q_low.predict(X_test)
        raw_high = model_q_high.predict(X_test)
        low = raw_low - q_conformal
        high = raw_high + q_conformal

        covered = (y_test >= low) & (y_test <= high)
        raw_covered = (y_test >= raw_low) & (y_test <= raw_high)

        verified[f"lead_{lead}h"] = {
            "persistence": compute_metrics(y_test, df_test["dst"].values),
            "helios_point": compute_metrics(y_test, pred_point),
            "conformal": {
                "nominal_coverage": 1.0 - ALPHA,
                "empirical_coverage": float(np.mean(covered)),
                "q_conformal_adjustment": q_conformal,
                "mean_interval_width": float(np.mean(high - low)),
                "raw_quantile_coverage": float(np.mean(raw_covered)),
                "raw_quantile_width": float(np.mean(raw_high - raw_low)),
            },
        }

        preds = df_test[["datetime", "dst", target_col]].copy()
        preds["pred_point"] = pred_point
        preds["pred_low_cqr"] = low
        preds["pred_high_cqr"] = high
        preds["covered"] = covered
        out_parquet = os.path.join(RESULTS_DIR, f"test_predictions_lead_{lead}h.parquet")
        preds.to_parquet(out_parquet, index=False)

        c = verified[f"lead_{lead}h"]["conformal"]
        print(f"+{lead}h | coverage {c['empirical_coverage']*100:.2f}% "
              f"(raw {c['raw_quantile_coverage']*100:.2f}%) | width {c['mean_interval_width']:.2f} nT "
              f"| q_conf {c['q_conformal_adjustment']:.3f} | rows {len(preds)}")

    out_json = os.path.join(RESULTS_DIR, "tactical_metrics_verified.json")
    with open(out_json, "w") as f:
        json.dump(verified, f, indent=2)
    print(f"\nVerified block written: {out_json}")

    if update:
        with open(METRICS_FILE) as f:
            metrics = json.load(f)
        for key, block in verified.items():
            metrics[key] = block
        metrics.setdefault("verification", {})
        metrics["verification"]["tactical_metrics"] = {
            "script": "scripts/20_verify_tactical_metrics.py",
            "method": "recomputed from published joblib models + published feature parquet (no retraining)",
            "metrics_file": "tactical_metrics_verified.json",
        }
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics JSON refreshed: {METRICS_FILE}")

    # Exit non-zero when the published JSON disagrees with the reproducible values
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as f:
            published = json.load(f)
        # Published values are rounded; compare at the precision actually stored.
        decimals = {"empirical_coverage": 4, "mean_interval_width": 2, "q_conformal_adjustment": 3}
        mismatches = []
        for key, block in verified.items():
            pub = published.get(key, {}).get("conformal", {})
            for field, nd in decimals.items():
                if field in pub and round(float(block["conformal"][field]), nd) != round(float(pub[field]), nd):
                    mismatches.append(f"{key}.{field}: published={pub[field]} verified={block['conformal'][field]}")
        if mismatches:
            print("\nMISMATCHES vs published metrics JSON:")
            for m in mismatches:
                print("  " + m)
            return 1
        print("\nOK: published metrics JSON matches reproducible values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
