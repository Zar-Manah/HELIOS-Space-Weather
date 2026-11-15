#!/usr/bin/env python3
"""
HELIOS-Transit - Hybrid Dual-Scale Ensemble (v2.5).
Blends the Surgically Edited Mixture-of-Kinematics Neural Network (MoKTransitNN)
and Gradient Boosted Trees with Conformal Quantile Bands.
Runs GBDT first, then PyTorch second to prevent OpenMP memory conflicts.
"""

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
import joblib

DATA_PATH = os.path.join(BASE_DIR, "data/processed/helios_cme_transit_matched.parquet")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

df = pd.read_parquet(DATA_PATH)
L_AU = 1.495978707e8
v_amb = 420.0
gamma = 1.0e-7

df["log_speed"] = np.log(df["cme_speed"])
df["inv_speed"] = 1.0 / (df["cme_speed"] + 10.0)
df["energy_proxy"] = (df["cme_speed"] / 1000.0) ** 2
df["width_norm"] = df["cme_width"] / 360.0

v_eff = df["cme_speed"] * np.exp(-gamma * L_AU * 0.4) + v_amb * (1.0 - np.exp(-gamma * L_AU * 0.4))
df["drag_prior_h"] = L_AU / (v_eff * 3600.0)
df["v_diff_ambient"] = df["cme_speed"] - v_amb
df["drag_accel_proxy"] = -gamma * df["v_diff_ambient"] * np.abs(df["v_diff_ambient"])
df["transit_hours_dbm_prior"] = df["drag_prior_h"]
df["is_fast_cme"] = (df["cme_speed"] >= 1000.0).astype(int)
df["solar_cycle_phase"] = (df["year"] - 1996) % 11 / 11.0
df["sin_cycle"] = np.sin(2 * np.pi * df["solar_cycle_phase"])
df["cos_cycle"] = np.cos(2 * np.pi * df["solar_cycle_phase"])

tree_feats = [
    "cme_speed", "cme_width", "cme_accel", "is_halo",
    "v_diff_ambient", "drag_accel_proxy", "transit_hours_dbm_prior",
    "is_fast_cme", "sin_cycle", "cos_cycle"
]

# 1. Run Tree Models First
print("Executing Tree Inference (GBDT + Quantiles)...")
m_time = joblib.load(os.path.join(MODELS_DIR, "helios_transit_time.joblib"))
m_low  = joblib.load(os.path.join(MODELS_DIR, "helios_transit_q_low.joblib"))
m_high = joblib.load(os.path.join(MODELS_DIR, "helios_transit_q_high.joblib"))
m_dst  = joblib.load(os.path.join(MODELS_DIR, "helios_transit_dst.joblib"))

test_df = df[df["solar_cycle"] == "Cycle 25"].copy().reset_index(drop=True)
preds_tree_time = m_time.predict(test_df[tree_feats])
preds_tree_low  = m_low.predict(test_df[tree_feats])
preds_tree_high = m_high.predict(test_df[tree_feats])
preds_tree_dst  = m_dst.predict(test_df[tree_feats])
print("Tree predictions completed cleanly.")

# 2. Run PyTorch Second (MoKTransitNN)
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

nn_feats = ["cme_speed", "log_speed", "inv_speed", "energy_proxy", "width_norm", "cme_accel", "is_halo", "drag_prior_h"]
train_df = df[df["solar_cycle"].isin(["Cycle 23", "Cycle 24"])].copy().reset_index(drop=True)

scaler = StandardScaler()
scaler.fit(train_df[nn_feats])
X_te = scaler.transform(test_df[nn_feats])
y_te = test_df["transit_hours"].values
fast_mask = (test_df["cme_speed"] >= 1000.0).values

class MoKTransitNN(nn.Module):
    def __init__(self, in_features=8):
        super().__init__()
        self.expert_fast_backbone = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU()
        )
        self.expert_fast_head = nn.Linear(16, 1)

        self.expert_slow_backbone = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU()
        )
        self.expert_slow_head = nn.Linear(16, 1)

        self.gate = nn.Sequential(
            nn.Linear(in_features, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward_experts(self, x):
        h_fast = self.expert_fast_backbone(x)
        h_slow = self.expert_slow_backbone(x)
        y_fast = self.expert_fast_head(h_fast)
        y_slow = self.expert_slow_head(h_slow)
        g = self.gate(x)
        return g, y_fast, y_slow

    def forward(self, x):
        g, y_fast, y_slow = self.forward_experts(x)
        return (g * y_fast + (1.0 - g) * y_slow).squeeze(1)

net = MoKTransitNN(in_features=len(nn_feats))
edited_ckpt = os.path.join(MODELS_DIR, "helios_transit_mok_edited.pt")
net.load_state_dict(torch.load(edited_ckpt, map_location="cpu", weights_only=True))
net.eval()

with torch.no_grad():
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    preds_mok = net(X_te_t).numpy()

# 3. Dynamic Regime Ensemble
# Smooth sigmoid transition between 750 and 1250 km/s
w_dynamic = 0.45 + 0.45 / (1.0 + np.exp(-(test_df["cme_speed"].values - 950.0) / 100.0))
preds_hybrid = w_dynamic * preds_mok + (1.0 - w_dynamic) * preds_tree_time

# Conformal calibration adjustment on hybrid
test_df["pred_hybrid_transit"] = preds_hybrid
test_df["q_low"] = preds_tree_low
test_df["q_high"] = preds_tree_high
test_df["pred_dst_min"] = preds_tree_dst

# -------------------------------------------------------------
# Metrics: global and fast-subset, for every model and baseline
# -------------------------------------------------------------
# Conformal adjustment calibrated on the training split (script 07)
with open(os.path.join(RESULTS_DIR, "helios_transit_metrics.json")) as f:
    q_adjust = float(json.load(f)["q_adjust"])

test_df["pred_gbdt_transit"] = preds_tree_time
test_df["q_low_raw"] = preds_tree_low
test_df["q_high_raw"] = preds_tree_high
test_df["q_low"] = preds_tree_low - q_adjust
test_df["q_high"] = preds_tree_high + q_adjust

fast_mask = test_df["cme_speed"].values >= 1000.0

def _metrics(preds, mask=None):
    m = np.ones(len(y_te), dtype=bool) if mask is None else mask
    p = np.asarray(preds)
    err = np.abs(y_te[m] - p[m])
    return {
        "mae": round(float(err.mean()), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_te[m], p[m]))), 2),
        "median": round(float(np.median(err)), 2),
        "n": int(m.sum()),
    }

def _coverage(low, high, mask=None):
    m = np.ones(len(y_te), dtype=bool) if mask is None else mask
    inside = (y_te >= np.asarray(low)) & (y_te <= np.asarray(high))
    return round(float(inside[m].mean() * 100.0), 2)

mae_global = float(mean_absolute_error(y_te, preds_hybrid))
median_global = float(np.median(np.abs(y_te - preds_hybrid)))
mae_fast   = float(mean_absolute_error(y_te[fast_mask], preds_hybrid[fast_mask])) if fast_mask.sum() > 0 else 0.0
median_fast = float(np.median(np.abs(y_te[fast_mask] - preds_hybrid[fast_mask]))) if fast_mask.sum() > 0 else 0.0
rmse_global = float(np.sqrt(mean_squared_error(y_te, preds_hybrid)))
r2_global  = float(r2_score(y_te, preds_hybrid))
dbm_baseline = float(mean_absolute_error(y_te, test_df["drag_prior_h"].values))

# Analytical 1D Vršnak Drag-Based Model (Vršnak et al., 2013)
def dbm_transit_vrsnak(v0, w=420.0, gamma=0.5e-7, R_target=1.496e8, R0=1.4e7):
    delta_R = R_target - R0
    t = delta_R / (0.5 * (v0 + w))
    for _ in range(10):
        if v0 >= w:
            arg = max(1e-6, 1.0 + gamma * (v0 - w) * t)
            f = w * t + (1.0 / gamma) * np.log(arg) - delta_R
            f_prime = w + (v0 - w) / arg
        else:
            arg = max(1e-6, 1.0 - gamma * (w - v0) * t)
            f = w * t - (1.0 / gamma) * np.log(arg) - delta_R
            f_prime = w - (w - v0) / arg
        dt = f / f_prime
        t = t - dt
        if abs(dt) < 1.0:
            break
    return t / 3600.0

vrsnak_preds = [dbm_transit_vrsnak(v) for v in test_df["cme_speed"].values]
mae_vrsnak_dbm = float(mean_absolute_error(y_te, vrsnak_preds))

# Coverage of the ensemble interval: raw quantile window vs conformalised window
coverage_raw_pct = _coverage(test_df["q_low_raw"], test_df["q_high_raw"])
coverage_pct = _coverage(test_df["q_low"], test_df["q_high"])
mean_window_h = float((test_df["q_high"] - test_df["q_low"]).mean())

metrics = {
    "global": {
        "dbm_prior": _metrics(test_df["drag_prior_h"].values),
        "vrsnak_dbm": _metrics(vrsnak_preds),
        "gbdt": _metrics(preds_tree_time),
        "mok": _metrics(preds_mok),
        "ensemble": _metrics(preds_hybrid),
    },
    "fast": {
        "dbm_prior": _metrics(test_df["drag_prior_h"].values, fast_mask),
        "vrsnak_dbm": _metrics(vrsnak_preds, fast_mask),
        "gbdt": _metrics(preds_tree_time, fast_mask),
        "mok": _metrics(preds_mok, fast_mask),
        "ensemble": _metrics(preds_hybrid, fast_mask),
    },
    "coverage_pct": {
        "global_raw": coverage_raw_pct,
        "global_conformal": coverage_pct,
        "fast_raw": _coverage(test_df["q_low_raw"], test_df["q_high_raw"], fast_mask),
        "fast_conformal": _coverage(test_df["q_low"], test_df["q_high"], fast_mask),
        "mean_window_hours": round(mean_window_h, 2),
    },
}

print("\n" + "=" * 78)
print("🏆 HELIOS-TRANSIT v2.5.19: UNIFIED DUAL-SCALE BENCHMARK (Cycle 25 OOD)")
print("=" * 78)
print("  GLOBAL (all 91 Cycle 25 CMEs)            MAE     RMSE   Median")
for key, label in [("dbm_prior", "Baseline Drag Prior Proxy (DBM-prior)"),
                   ("vrsnak_dbm", "Analytical Drag-Based Model (Vršnak)"),
                   ("gbdt", "HELIOS Single GBDT"),
                   ("mok", "MoK Neural Network"),
                   ("ensemble", "HELIOS Dynamic Ensemble (MoK+GBDT)")]:
    m = metrics["global"][key]
    print(f"    {label:38s} ±{m['mae']:6.2f} {m['rmse']:6.2f} {m['median']:6.2f}")
print(f"  FAST SUBSET (n={metrics['fast']['ensemble']['n']}, >=1000 km/s)")
for key, label in [("dbm_prior", "Baseline Drag Prior Proxy (DBM-prior)"),
                   ("vrsnak_dbm", "Analytical Drag-Based Model (Vršnak)"),
                   ("gbdt", "HELIOS Single GBDT"),
                   ("mok", "MoK Neural Network"),
                   ("ensemble", "HELIOS Dynamic Ensemble (MoK+GBDT)")]:
    m = metrics["fast"][key]
    print(f"    {label:38s} ±{m['mae']:6.2f} {m['rmse']:6.2f} {m['median']:6.2f}")
print("-" * 78)
print("  REFERENCE (literature, NOT re-evaluated here):")
print("    Wold et al. 2018 WSA-ENLIL, Cycle 24: ±10.40 h Earth, ±9.20 h STEREO-A")
print("    Both values are GLOBAL (all CMEs). No published fast-subset baseline exists.")
print("-" * 78)
print(f"  Explained Variance (R², ensemble global): {r2_global:.3f}")
print(f"  Interval coverage  global: raw {coverage_raw_pct:.1f}% -> conformal {coverage_pct:.1f}%")
print(f"  Interval coverage  fast:   raw {metrics['coverage_pct']['fast_raw']:.1f}% -> conformal {metrics['coverage_pct']['fast_conformal']:.1f}%")
print(f"  Mean conformal window: {mean_window_h:.2f} h")
print("=" * 78)

# Save ensemble configuration
ensemble_config = {
    "architecture": "MoKTransitNN + GBDT Dynamic Regime",
    "version": "2.5.19",
    "w_base": 0.45,
    "w_fast_boost": 0.45,
    "transition_v0": 950.0,
    "transition_scale": 100.0,
    "single_gbdt_mae": round(float(mean_absolute_error(y_te, preds_tree_time)), 2),
    "single_gbdt_median": round(float(np.median(np.abs(y_te - preds_tree_time))), 2),
    "global_mae": round(mae_global, 2),
    "global_median": round(median_global, 2),
    "fast_cme_mae": round(mae_fast, 2),
    "global_rmse": round(rmse_global, 2),
    "r2_score": round(r2_global, 3),
    "ensemble_raw_quantile_coverage_pct": coverage_raw_pct,
    "ensemble_raw_quantile_mean_window_hours": round(mean_window_h, 1),
    "benchmark_vrsnak_dbm": round(mae_vrsnak_dbm, 2),
    "benchmark_nasa_enlil_earth": 10.40,
    "benchmark_nasa_enlil_stereo_a_global": 9.20,
    "benchmark_dbm_prior": round(dbm_baseline, 1),
    "metrics": metrics,
}

with open(os.path.join(MODELS_DIR, "helios_transit_ensemble_config.json"), "w") as f:
    json.dump(ensemble_config, f, indent=2)

metrics_file = os.path.join(RESULTS_DIR, "transit_ensemble_benchmark.json")
with open(metrics_file, "w") as f:
    json.dump(ensemble_config, f, indent=2)

test_df.to_parquet(os.path.join(RESULTS_DIR, "transit_cycle25_predictions.parquet"))
print(f"\n✅ Configurations and predictions saved to {MODELS_DIR} & {RESULTS_DIR}")
