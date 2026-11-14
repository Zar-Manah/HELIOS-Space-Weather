#!/usr/bin/env python3
"""
HELIOS - Solar Storm Predictor (Pipeline Phase 3)
Generates publication-quality scientific visualizations and case studies:
  1. Multi-horizon skill benchmarks (RMSE, R², TSS, Coverage).
  2. The May 10-12, 2024 G5 Superstorm (Dst = -406 nT) case study with Conformal Bands.
  3. Feature importance & physical coupling hierarchy.
"""

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA_PATH = os.path.join(BASE_DIR, "data/processed/helios_omni_features_1998_2026.parquet")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_FILE = os.path.join(RESULTS_DIR, "helios_evaluation_metrics.json")
IMPORTANCE_FILE = os.path.join(RESULTS_DIR, "feature_importance.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)

# Set high-quality styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

print("=" * 70)
print("📈 HELIOS SPACE WEATHER: PUBLICATION FIGURE GENERATION 📈")
print("=" * 70)

# -------------------------------------------------------------
# FIGURE 1: MULTI-HORIZON BENCHMARKS
# -------------------------------------------------------------
print("Generating Figure 1: Multi-horizon performance benchmark...")

with open(METRICS_FILE) as f:
    metrics = json.load(f)

horizons = [1, 2, 3, 6]
p_rmse = [metrics[f"lead_{h}h"]["persistence"]["rmse"] for h in horizons]
h_rmse = [metrics[f"lead_{h}h"]["helios_point"]["rmse"] for h in horizons]

p_r2 = [metrics[f"lead_{h}h"]["persistence"]["r2"] for h in horizons]
h_r2 = [metrics[f"lead_{h}h"]["helios_point"]["r2"] for h in horizons]

p_tss = [metrics[f"lead_{h}h"]["persistence"]["tss_50"] for h in horizons]
h_tss = [metrics[f"lead_{h}h"]["helios_point"]["tss_50"] for h in horizons]

c_cov = [metrics[f"lead_{h}h"]["conformal"]["empirical_coverage"] * 100 for h in horizons]
c_width = [metrics[f"lead_{h}h"]["conformal"]["mean_interval_width"] for h in horizons]

fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=300)

# Subplot A: RMSE
ax = axes[0, 0]
ax.plot(horizons, p_rmse, "o--", color="#d9534f", label="Persistence Baseline", linewidth=2, markersize=7)
ax.plot(horizons, h_rmse, "s-", color="#0275d8", label="HELIOS (Ours)", linewidth=2.5, markersize=8)
ax.set_title("(a) Root Mean Square Error (RMSE)", fontsize=12, fontweight="bold")
ax.set_xlabel("Lead Time (Hours)", fontsize=10)
ax.set_ylabel("RMSE (nT)", fontsize=10)
ax.set_xticks(horizons)
ax.legend(frameon=True)
ax.grid(True, linestyle="--", alpha=0.6)

# Subplot B: R² Score
ax = axes[0, 1]
ax.plot(horizons, p_r2, "o--", color="#d9534f", label="Persistence Baseline", linewidth=2, markersize=7)
ax.plot(horizons, h_r2, "s-", color="#0275d8", label="HELIOS (Ours)", linewidth=2.5, markersize=8)
ax.set_title("(b) Coefficient of Determination ($R^2$)", fontsize=12, fontweight="bold")
ax.set_xlabel("Lead Time (Hours)", fontsize=10)
ax.set_ylabel("$R^2$ Score", fontsize=10)
ax.set_xticks(horizons)
ax.legend(frameon=True)
ax.grid(True, linestyle="--", alpha=0.6)

# Subplot C: True Skill Statistic (TSS for Storms <= -50 nT)
ax = axes[1, 0]
ax.plot(horizons, p_tss, "o--", color="#d9534f", label="Persistence Baseline", linewidth=2, markersize=7)
ax.plot(horizons, h_tss, "s-", color="#5cb85c", label="HELIOS Skill", linewidth=2.5, markersize=8)
ax.set_title(r"(c) True Skill Statistic (TSS, $Dst \leq -50$ nT)", fontsize=12, fontweight="bold")
ax.set_xlabel("Lead Time (Hours)", fontsize=10)
ax.set_ylabel("TSS ($TPR - FPR$)", fontsize=10)
ax.set_xticks(horizons)
ax.legend(frameon=True)
ax.grid(True, linestyle="--", alpha=0.6)

# Subplot D: Conformal Coverage & Width
ax = axes[1, 1]
ax.axhline(90.0, color="gray", linestyle=":", label="Nominal 90% Target")
ax.plot(horizons, c_cov, "D-", color="#f0ad4e", label="Empirical Coverage (%)", linewidth=2.5, markersize=8)
ax.set_title("(d) Conformal 90% Empirical Coverage", fontsize=12, fontweight="bold")
ax.set_xlabel("Lead Time (Hours)", fontsize=10)
ax.set_ylabel("Empirical Coverage (%)", fontsize=10)
ax.set_ylim(85, 95)
ax.set_xticks(horizons)
ax.legend(frameon=True)
ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
fig1_path = os.path.join(FIGURES_DIR, "fig1_multi_horizon_benchmarks.png")
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"  Saved: {fig1_path}")

# -------------------------------------------------------------
# FIGURE 2: THE MAY 10-12, 2024 G5 SUPERSTORM CASE STUDY
# -------------------------------------------------------------
print("Generating Figure 2: May 10-12, 2024 G5 Superstorm case study...")

# Load predictions for Lead 1h and Lead 3h
preds_1h = pd.read_parquet(os.path.join(RESULTS_DIR, "test_predictions_lead_1h.parquet"))
preds_3h = pd.read_parquet(os.path.join(RESULTS_DIR, "test_predictions_lead_3h.parquet"))

# Filter to the peak period: May 09, 2024 to May 13, 2024
start_date = pd.to_datetime("2024-05-09 12:00:00")
end_date   = pd.to_datetime("2024-05-13 12:00:00")

mask_1h = (preds_1h["datetime"] >= start_date) & (preds_1h["datetime"] <= end_date)
sub_1h = preds_1h[mask_1h].sort_values("datetime").reset_index(drop=True)

mask_3h = (preds_3h["datetime"] >= start_date) & (preds_3h["datetime"] <= end_date)
sub_3h = preds_3h[mask_3h].sort_values("datetime").reset_index(drop=True)

# Also load raw solar wind drivers for the same period
df_all = pd.read_parquet(DATA_PATH)
mask_drivers = (df_all["datetime"] >= start_date) & (df_all["datetime"] <= end_date)
sub_drivers = df_all[mask_drivers].sort_values("datetime").reset_index(drop=True)

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, dpi=300)

# Panel 1: Solar Wind Forcing (Bz GSM and Flow Speed)
ax0 = axes[0]
color_bz = "#d9534f"
color_v = "#0275d8"

ax0.plot(sub_drivers["datetime"], sub_drivers["bz_gsm"], color=color_bz, linewidth=1.8, label="$B_z$ GSM (nT)")
ax0.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax0.set_ylabel("$B_z$ (nT)", color=color_bz, fontsize=11, fontweight="bold")
ax0.tick_params(axis="y", labelcolor=color_bz)
ax0.set_title("The Historic May 10–12, 2024 G5 Superstorm (Unseen Cycle 25 Test Event)", fontsize=13, fontweight="bold")
ax0.grid(True, linestyle="--", alpha=0.5)

ax0_right = ax0.twinx()
ax0_right.plot(sub_drivers["datetime"], sub_drivers["v_flow"], color=color_v, linewidth=1.8, linestyle="-.", label="Solar Wind Speed (km/s)")
ax0_right.set_ylabel("Speed $V_{sw}$ (km/s)", color=color_v, fontsize=11, fontweight="bold")
ax0_right.tick_params(axis="y", labelcolor=color_v)

# Panel 2: 1-Hour Ahead HELIOS Forecast + Conformal 90% Bands
ax1 = axes[1]
ax1.plot(sub_1h["datetime"], sub_1h["target_dst_lead_1h"], color="black", linewidth=2.2, label="Actual Ground Truth $Dst$")
ax1.plot(sub_1h["datetime"], sub_1h["pred_point"], color="#0275d8", linewidth=1.8, linestyle="--", label="HELIOS Point Forecast (+1h)")
ax1.fill_between(
    sub_1h["datetime"],
    sub_1h["pred_low_cqr"],
    sub_1h["pred_high_cqr"],
    color="#0275d8",
    alpha=0.25,
    label="Conformal 90% Guaranteed Interval"
)
ax1.axhline(-100, color="#f0ad4e", linestyle=":", label="Severe Storm Threshold (-100 nT)")
ax1.axhline(-200, color="#d9534f", linestyle=":", label="Superstorm Threshold (-200 nT)")
ax1.set_ylabel("$Dst$ Index (nT)", fontsize=11, fontweight="bold")
ax1.set_title("+1-Hour Tactical Space Weather Warning", fontsize=11, fontweight="bold")
ax1.legend(loc="lower left", frameon=True, fontsize=9)
ax1.grid(True, linestyle="--", alpha=0.5)

# Panel 3: 3-Hour Ahead Operational Forecast + Conformal 90% Bands
ax2 = axes[2]
ax2.plot(sub_3h["datetime"], sub_3h["target_dst_lead_3h"], color="black", linewidth=2.2, label="Actual Ground Truth $Dst$")
ax2.plot(sub_3h["datetime"], sub_3h["pred_point"], color="#5cb85c", linewidth=1.8, linestyle="--", label="HELIOS Point Forecast (+3h)")
ax2.fill_between(
    sub_3h["datetime"],
    sub_3h["pred_low_cqr"],
    sub_3h["pred_high_cqr"],
    color="#5cb85c",
    alpha=0.25,
    label="Conformal 90% Guaranteed Interval"
)
ax2.axhline(-100, color="#f0ad4e", linestyle=":", label="Severe Storm Threshold (-100 nT)")
ax2.axhline(-200, color="#d9534f", linestyle=":", label="Superstorm Threshold (-200 nT)")
ax2.set_ylabel("$Dst$ Index (nT)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Universal Time (UTC) — May 2024", fontsize=11)
ax2.set_title("+3-Hour Operational Grid Protection Forecast", fontsize=11, fontweight="bold")
ax2.legend(loc="lower left", frameon=True, fontsize=9)
ax2.grid(True, linestyle="--", alpha=0.5)

# Date formatting on X-axis
ax2.xaxis.set_major_locator(mdates.DayLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%Y"))
ax2.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))

plt.tight_layout()
fig2_path = os.path.join(FIGURES_DIR, "fig2_may2024_superstorm_case_study.png")
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"  Saved: {fig2_path}")

# -------------------------------------------------------------
# FIGURE 3: FEATURE IMPORTANCE & PHYSICAL COUPLING
# -------------------------------------------------------------
print("Generating Figure 3: Feature importance and physics hierarchy...")

importance = pd.read_csv(IMPORTANCE_FILE).head(15).iloc[::-1]

fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
bars = ax.barh(importance["feature"], importance["importance"], color="#337ab7", edgecolor="#23527c", height=0.7)

ax.set_title("Top 15 Most Predictive Features in HELIOS Space Weather Architecture", fontsize=12, fontweight="bold")
ax.set_xlabel("Feature Importance Score (Number of Decision Splits)", fontsize=10)
ax.grid(True, axis="x", linestyle="--", alpha=0.6)

# Label values
for bar in bars:
    w = bar.get_width()
    ax.text(w + 10, bar.get_y() + bar.get_height()/2, f"{int(w)}", va="center", fontsize=9, color="#333333")

plt.tight_layout()
fig3_path = os.path.join(FIGURES_DIR, "fig3_feature_importance.png")
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"  Saved: {fig3_path}")

print("✅ All scientific publication figures generated successfully!")
