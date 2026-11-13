#!/usr/bin/env python3
"""
Zar-Manah HELIOS 2.0 - Coronal Computer Vision vs. Physical Telemetry Ablation Study
Ablates 14-dim physical features vs. 8-dim C2 Coronagraph PCA vs. 16-dim Full Vision PCA (C2 + HMI).
Saves results to results/ablation_study.json.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "helios_vision_matched.parquet")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUT_JSON = os.path.join(RESULTS_DIR, "ablation_study.json")

PHYS_COLS = [
    "v_launch_3d", "width", "accel", "mass", "energy_3d", "is_halo",
    "f107", "r_sunspot", "v_sw_launch", "prior_v",
    "shear_angle_proxy", "free_energy_proxy", "helicity_proxy", "flare_flux_proxy"
]

C2_COLS = [f"pca_c2_{i}" for i in range(8)]
MAG_COLS = [f"pca_mag_{i}" for i in range(8)]
ALL_VIS_COLS = C2_COLS + MAG_COLS


def evaluate_config(df, feature_cols, target_col, train_mask, test_mask):
    X_df = df[feature_cols].copy()
    for c in feature_cols:
        X_df[c] = X_df[c].fillna(X_df[c].median())
    X_arr = X_df.values.astype(np.float32)

    p_mean = X_arr[train_mask].mean(axis=0)
    p_std = X_arr[train_mask].std(axis=0) + 1e-6
    X_norm = (X_arr - p_mean) / p_std

    y_tr = df.loc[train_mask, target_col].values.astype(int)
    y_te = df.loc[test_mask, target_col].values.astype(int)

    clf = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_split=4,
        random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = cross_val_predict(clf, X_norm[train_mask], y_tr, cv=cv, method="predict_proba")[:, 1]
    cv_auc = float(roc_auc_score(y_tr, oof_probs))

    clf.fit(X_norm[train_mask], y_tr)
    te_probs = clf.predict_proba(X_norm[test_mask])[:, 1]
    te_auc = float(roc_auc_score(y_te, te_probs))

    return cv_auc, te_auc


def main():
    print("=" * 80)
    print("🔬 RUNNING SYSTEMATIC ABLATION: PHYSICAL TELEMETRY VS CORONAL VISION")
    print("=" * 80)

    df = pd.read_parquet(DATA_PATH)
    train_mask = df["solar_cycle"].isin(["Cycle 23", "Cycle 24"]).values
    test_mask = (df["solar_cycle"] == "Cycle 25").values

    configs = [
        ("PHYS14 (Operational Model)", PHYS_COLS),
        ("PHYS + C2 Coronagraph PCA", PHYS_COLS + C2_COLS),
        ("PHYS + ALL Vision PCA", PHYS_COLS + ALL_VIS_COLS),
        ("Vision PCA Alone", ALL_VIS_COLS)
    ]

    results = {
        "storm_severity_dst_110": {},
        "chirality_southward_bz_10": {}
    }

    print("\n1. TARGET: SEVERE GEOMAGNETIC STORM VETO (Dst <= -110 nT)")
    print(f"{'Configuration':<32} | {'Dims':<5} | {'Train 5-Fold CV AUC':<20} | {'Test Cycle 25 AUC':<18}")
    print("-" * 85)
    for name, cols in configs:
        cv_auc, te_auc = evaluate_config(df, cols, "is_severe_storm", train_mask, test_mask)
        results["storm_severity_dst_110"][name] = {
            "n_features": len(cols),
            "train_cv_auc": round(cv_auc, 4),
            "test_roc_auc": round(te_auc, 4)
        }
        print(f"{name:<32} | {len(cols):<5} | {cv_auc:<20.4f} | {te_auc:<18.4f}")

    print("\n2. TARGET: SOUTHWARD CHIRALITY THREAT (Bz <= -10 nT)")
    print(f"{'Configuration':<32} | {'Dims':<5} | {'Train 5-Fold CV AUC':<20} | {'Test Cycle 25 AUC':<18}")
    print("-" * 85)
    for name, cols in configs:
        cv_auc, te_auc = evaluate_config(df, cols, "is_dangerous_southward", train_mask, test_mask)
        results["chirality_southward_bz_10"][name] = {
            "n_features": len(cols),
            "train_cv_auc": round(cv_auc, 4),
            "test_roc_auc": round(te_auc, 4)
        }
        print(f"{name:<32} | {len(cols):<5} | {cv_auc:<20.4f} | {te_auc:<18.4f}")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✅ Ablation study completed successfully. Saved to: {OUT_JSON}")
    print("=" * 80)


if __name__ == "__main__":
    main()
