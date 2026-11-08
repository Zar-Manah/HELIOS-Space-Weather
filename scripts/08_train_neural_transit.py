#!/usr/bin/env python3
"""
HELIOS-Transit - Mixture-of-Kinematics Architecture (MoKTransitNN).
Trains a bifurcated Physics-Informed neural network on Apple Silicon (MPS).
Separates hypersonic decelerating bow-shock dynamics from subsonic ambient entrainment.
Saves base checkpoint to models/helios_transit_mok_mps.pt for weight surgery.
"""

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = os.path.join(BASE_DIR, "data/processed/helios_cme_transit_matched.parquet"
MODELS_DIR = os.path.join(BASE_DIR, "models"
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "helios_transit_mok_mps.pt")

os.makedirs(MODELS_DIR, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using compute acceleration device: {device}")

df = pd.read_parquet(DATA_PATH)

# Feature engineering
L_AU = 1.495978707e8
v_amb = 420.0
gamma = 1.0e-7

df["log_speed"] = np.log(df["cme_speed"])
df["inv_speed"] = 1.0 / (df["cme_speed"] + 10.0)
df["energy_proxy"] = (df["cme_speed"] / 1000.0) ** 2
df["width_norm"] = df["cme_width"] / 360.0

v_eff = df["cme_speed"] * np.exp(-gamma * L_AU * 0.4) + v_amb * (1.0 - np.exp(-gamma * L_AU * 0.4))
df["drag_prior_h"] = L_AU / (v_eff * 3600.0)

FEATURES = [
    "cme_speed", "log_speed", "inv_speed", "energy_proxy",
    "width_norm", "cme_accel", "is_halo", "drag_prior_h"
]

train_df = df[df["solar_cycle"].isin(["Cycle 23", "Cycle 24"])].copy().reset_index(drop=True)
test_df  = df[df["solar_cycle"] == "Cycle 25"].copy().reset_index(drop=True)

scaler = StandardScaler()
X_tr = scaler.fit_transform(train_df[FEATURES])
y_tr = train_df["transit_hours"].values

X_te = scaler.transform(test_df[FEATURES])
y_te = test_df["transit_hours"].values

# Save scaler params for inference
scaler_dict = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "features": FEATURES
}
with open(os.path.join(MODELS_DIR, "transit_scaler.json"), "w") as f:
    json.dump(scaler_dict, f, indent=2)

class MoKTransitNN(nn.Module):
    def __init__(self, in_features=8):
        super().__init__()
        # Hypersonic shock expert (V >= 900 km/s decelerating regime)
        self.expert_fast_backbone = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU()
        )
        self.expert_fast_head = nn.Linear(16, 1)

        # Ambient entrainment expert (V < 500 km/s accelerating regime)
        self.expert_slow_backbone = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU()
        )
        self.expert_slow_head = nn.Linear(16, 1)

        # Smooth kinematic gating router
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


torch.manual_seed(1)
model = MoKTransitNN(in_features=len(FEATURES)).to(device)
criterion = nn.HuberLoss(delta=8.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=350)

X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)
X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)

print(f"Training MoKTransitNN (350 epochs on {device})...")
for epoch in range(1, 351):
    model.train()
    optimizer.zero_grad()
    preds = model(X_tr_t)
    loss = criterion(preds, y_tr_t)
    loss.backward()
    optimizer.step()
    scheduler.step()

model.eval()
with torch.no_grad():
    y_pred_te = model(X_te_t).cpu().numpy()

fast_mask = (test_df["cme_speed"] >= 1000.0).values
mae_base = mean_absolute_error(y_te, y_pred_te)
mae_fast = mean_absolute_error(y_te[fast_mask], y_pred_te[fast_mask])
rmse_base = np.sqrt(mean_squared_error(y_te, y_pred_te))
r2_base = r2_score(y_te, y_pred_te)

print("\n" + "=" * 65)
print("📊 MoKTransitNN Base Checkpoint (Cycle 25 OOD):")
print(f"  • Global MAE:      ±{mae_base:.2f} hours")
print(f"  • Fast CME MAE:    ±{mae_fast:.2f} hours (NASA ENLIL: 9.2h)")
print(f"  • RMSE:             {rmse_base:.2f} hours")
print(f"  • R² Score:         {r2_base:.3f}")
print("=" * 65)

# Save base model
torch.save(model.cpu().state_dict(), CHECKPOINT_PATH)
print(f"Saved base MoK checkpoint to: {CHECKPOINT_PATH}")
