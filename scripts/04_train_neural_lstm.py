#!/usr/bin/env python3
"""
HELIOS - Solar Storm Predictor (Pipeline Phase 4)
Deep Temporal Neural Architecture: Bi-LSTM with Temporal Attention on PyTorch MPS.
Trained on 24-hour multivariate sequence history to forecast Dst index +1h and +3h ahead.
"""

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_PATH = os.path.join(BASE_DIR, "data/processed/helios_omni_features_1998_2026.parquet"
MODELS_DIR = os.path.join(BASE_DIR, "models"
RESULTS_DIR = os.path.join(BASE_DIR, "results"

os.makedirs(MODELS_DIR, exist_ok=True)

# Select device (Apple Silicon Metal MPS)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using compute acceleration device: {device}")

# Core time series features to feed into sequential memory
SEQ_FEATURES = [
    "dst", "bz_gsm", "by_gsm", "b_mag", "v_flow", "np_dens", "temp", "pdyn",
    "ey_rectified", "newell_coupling", "akasofu_eps", "borovsky_coupling", "kp", "f107"
]
SEQ_LEN = 24  # 24-hour historical window

print("Loading dataset for Deep Temporal Bi-LSTM...")
df = pd.read_parquet(DATA_PATH)

# Train on 1998-2019, Test on Cycle 25 (2020-2026)
train_df = df[df["year"] <= 2019].copy().reset_index(drop=True)
test_df  = df[df["year"] >= 2020].copy().reset_index(drop=True)

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df[SEQ_FEATURES])
test_scaled  = scaler.transform(test_df[SEQ_FEATURES])

y_train_1h = train_df["target_dst_lead_1h"].values
y_test_1h  = test_df["target_dst_lead_1h"].values

class SolarTimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len=24):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.X) - self.seq_len
        
    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]
        y_val = self.y[idx + self.seq_len - 1]
        return x_seq, y_val

train_dataset = SolarTimeSeriesDataset(train_scaled, y_train_1h, seq_len=SEQ_LEN)
test_dataset  = SolarTimeSeriesDataset(test_scaled, y_test_1h, seq_len=SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_dataset, batch_size=512, shuffle=False)

print(f"Sequence datasets created: {len(train_dataset):,} train sequences, {len(test_dataset):,} test sequences.")

class HeliosBiLSTM(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.15
        )
        # Temporal attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x: [Batch, Seq_len, Features]
        lstm_out, _ = self.lstm(x) # [Batch, Seq_len, Hidden*2]
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1) # [Batch, Seq_len, 1]
        context = torch.sum(lstm_out * attn_weights, dim=1) # [Batch, Hidden*2]
        out = self.fc(context) # [Batch, 1]
        return out.squeeze(1)

model = HeliosBiLSTM(in_features=len(SEQ_FEATURES), hidden_dim=64).to(device)
criterion = nn.SmoothL1Loss() # Huber loss for extreme storm outlier robustness
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)

print("Training HeliosBiLSTM on Apple Silicon Metal MPS (8 epochs)...")
for epoch in range(1, 9):
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        
    scheduler.step()
    avg_loss = total_loss / len(train_dataset)
    print(f"  Epoch {epoch}/8 - Training Huber Loss: {avg_loss:.4f}")

# Evaluation on Solar Cycle 25 OOD
model.eval()
all_preds = []
all_trues = []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        preds = model(x_batch).cpu().numpy()
        all_preds.extend(preds)
        all_trues.extend(y_batch.numpy())

all_preds = np.array(all_preds)
all_trues = np.array(all_trues)

rmse = np.sqrt(mean_squared_error(all_trues, all_preds))
mae = mean_absolute_error(all_trues, all_preds)
r2 = r2_score(all_trues, all_preds)

actual_storm = (all_trues <= -50)
pred_storm   = (all_preds <= -50)
tp = np.sum(actual_storm & pred_storm)
tn = np.sum((~actual_storm) & (~pred_storm))
fp = np.sum((~actual_storm) & pred_storm)
fn = np.sum(actual_storm & (~pred_storm))
tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
tss = tpr - fpr

print("\n" + "=" * 60)
print(f"🧠 HELIOS Bi-LSTM Neural Performance (+1h Lead, Cycle 25):")
print(f"  • RMSE: {rmse:.2f} nT")
print(f"  • MAE:  {mae:.2f} nT")
print(f"  • R²:   {r2:.3f}")
print(f"  • TSS:  {tss:.3f}")
print("=" * 60)

# Save neural model
torch.save(model.state_dict(), os.path.join(MODELS_DIR, "helios_bilstm_mps.pt"))
print("Saved HeliosBiLSTM weights to:", os.path.join(MODELS_DIR, "helios_bilstm_mps.pt"))

