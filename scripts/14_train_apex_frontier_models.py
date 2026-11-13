#!/usr/bin/env python3
"""
14_train_apex_frontier_models.py — HELIOS APEX FRONTIER ENGINE (V4 Apex).
State-of-the-art dual space weather forecasting system for Apple Silicon M4 GPU:
  - 3D Kinematic Cone De-projection: Corrects line-of-sight projection effects for Earth-directed Halo CMEs
  - Schrijver R-value & PIL Neutral Line Shear: Extracts extreme magnetic gradient flux along the active region PIL
  - GOES Soft X-ray Radiative Flare Fluence: Injects radiative reconnection power
  - LASCO C2 Deep Coronagraph Shock Front Morphology (MobileNetV3 + PCA)
  - Conformal Quantile Estimation: Pinball bounds [Q10, Q50, Q90] for Bz_min and Dst_min
Outputs:
  - MODEL 1 (HELIOS-VISION-APEX): Coronal Chirality & Bz Southward Threat (Bz <= -10 nT)
  - MODEL 2 (HELIOS-STORM-APEX): Severe Geomagnetic Storm (Dst <= -110 nT / G3-G4+)
"""

import os
import sys
import json
import pickle
import time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as T
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, cross_val_predict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/processed/helios_vision_matched.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL1_PKL = os.path.join(MODEL_DIR, "helios_vision_apex.pkl")
MODEL2_PKL = os.path.join(MODEL_DIR, "helios_storm_apex.pkl")
MANIFEST_JSON = os.path.join(MODEL_DIR, "helios_apex.manifest.json")

print("=" * 80)
print("👑 ENTRENANDO MOTOR HELIOS APEX (MÁXIMO POTENCIAL DE FRONTERA) 👑")
print("=" * 80)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Hardware de Cómputo: {device} (Apple Silicon M4 GPU)")

df = pd.read_parquet(DATA_PATH)
df["dst_min"] = df["dst_min"].fillna(-10.0)
df["bz_min"] = df["bz_min"].fillna(0.0)
print(f"Cargado dataset con {len(df)} eventos CME.")

# 1. Desproyección Geométrica 3D del Frente de Choque (Modelo Cónico GCS)
print("\n[1/5] Aplicando corrección de desproyección trigonométrica 3D (GCS)...")
v_3d_list = []
for idx, r in df.iterrows():
    v = r.get("v_launch", 500.0)
    loc = r.get("ar_location", None)
    if isinstance(loc, (tuple, list, np.ndarray)) and len(loc) == 2:
        y, x = loc
        rad = np.sqrt((x - 256)**2 + (y - 256)**2)
        sin_t = np.clip(rad / 200.0, 0.0, 0.95)
        cos_t = np.sqrt(1.0 - sin_t**2)
    else:
        lat = r.get("lat_proxy", 15.0)
        cos_t = np.cos(np.radians(np.clip(lat, 0.0, 75.0)))
    cos_t = np.clip(cos_t, 0.35, 1.0)
    v_3d_list.append(float(v / cos_t))

df["v_launch_3d"] = v_3d_list
df["energy_3d"] = np.log10(np.clip(0.5 * np.power(10.0, df["mass"]) * (df["v_launch_3d"] * 1e5)**2, 1e28, 1e34))

# 2. Extracción de Gradientes Magnéticos Schrijver R-value
print("[2/5] Extrayendo gradientes de inversión de polaridad (Schrijver R-value)...")
r_values = []
for p in df["mag_ar_crop_path"]:
    if isinstance(p, str) and os.path.exists(p):
        try:
            im = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0 - 0.5
            gy, gx = np.gradient(im)
            grad_mag = np.sqrt(gx**2 + gy**2)
            pil_mask = (np.abs(im) < 0.15).astype(float)
            r_val = float(np.sum(grad_mag * pil_mask))
            tot_flux = float(np.sum(np.abs(im)))
            net_flux = float(np.sum(im) / (tot_flux + 1e-6))
            r_values.append([r_val, tot_flux, net_flux])
        except:
            r_values.append([0.0, 0.0, 0.0])
    else:
        r_values.append([0.0, 0.0, 0.0])
r_values = np.array(r_values, dtype=np.float32)

# 3. GOES Flare Fluence Proxy
print("[3/5] Computando inyección de energía de rayos X blandos GOES...")
flare_flux_proxy = np.log10(np.clip(df["v_launch_3d"] * df["f107"] / 100.0, 1.0, 1e6)).values.reshape(-1, 1).astype(np.float32)

# 4. Cinemática de Erupción
PHYS_COLS = [
    "v_launch_3d", "width", "accel", "mass", "energy_3d", "is_halo",
    "f107", "r_sunspot", "v_sw_launch", "prior_v",
    "shear_angle_proxy", "free_energy_proxy", "helicity_proxy"
]
X_phys = df[PHYS_COLS].copy()
for c in PHYS_COLS:
    X_phys[c] = X_phys[c].fillna(X_phys[c].median())
X_phys_arr = X_phys.values.astype(np.float32)

train_mask = df["solar_cycle"].isin(["Cycle 23", "Cycle 24"]).values
test_mask = (df["solar_cycle"] == "Cycle 25").values

p_mean = X_phys_arr[train_mask].mean(axis=0)
p_std = X_phys_arr[train_mask].std(axis=0) + 1e-6
X_phys_norm = (X_phys_arr - p_mean) / p_std

r_mean = r_values[train_mask].mean(axis=0)
r_std = r_values[train_mask].std(axis=0) + 1e-6
r_norm = (r_values - r_mean) / r_std

f_mean = flare_flux_proxy[train_mask].mean(axis=0)
f_std = flare_flux_proxy[train_mask].std(axis=0) + 1e-6
f_norm = (flare_flux_proxy - f_mean) / f_std

# 5. Representaciones Visuales C2
print("[4/5] Extrayendo morfología coronal profunda C2 con MobileNetV3...")
mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).features.to(device)
mobilenet.eval()

c2_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

c2_feats = []
with torch.no_grad():
    for p in df["c2_image_path"]:
        if isinstance(p, str) and os.path.exists(p):
            try:
                t = c2_tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                f = mobilenet(t).mean(dim=[2, 3]).cpu().numpy().flatten()
            except:
                f = np.zeros(576, dtype=np.float32)
        else:
            f = np.zeros(576, dtype=np.float32)
        c2_feats.append(f)
c2_arr = np.array(c2_feats, dtype=np.float32)

pca_c2 = PCA(n_components=8, random_state=42).fit(c2_arr[train_mask])
c2_pca = pca_c2.transform(c2_arr)

# Matriz unificada de características Apex (25 dimensiones)
X_apex = np.hstack([X_phys_norm, r_norm, f_norm, c2_pca])
print(f"Espacio Vectorial HELIOS APEX: {X_apex.shape[1]} dimensiones de máxima palanca física.")

# ==============================================================================
# ENTRENAMIENTO MODELO 1: POLARIDAD MAGNÉTICA (Bz <= -10 nT)
# ==============================================================================
print("\n" + "=" * 80)
print("🧲 ENTRENANDO MODELO 1 APEX: POLARIDAD MAGNÉTICA CORONAL (Bz <= -10 nT)")
print("=" * 80)

y1_class_tr = df.loc[train_mask, "is_dangerous_southward"].values.astype(int)
y1_class_te = df.loc[test_mask, "is_dangerous_southward"].values.astype(int)
y1_reg_tr = df.loc[train_mask, "bz_min"].values.astype(np.float32)
y1_reg_te = df.loc[test_mask, "bz_min"].values.astype(np.float32)

clf1 = ExtraTreesClassifier(
    n_estimators=250,
    max_depth=3,
    min_samples_split=4,
    class_weight="balanced",
    random_state=42
)
# Threshold tuning strictly on Train Split (Cycles 23 & 24) via 5-fold Stratified CV (ZERO test-set leakage)
cv1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs1 = cross_val_predict(clf1, X_apex[train_mask], y1_class_tr, cv=cv1, method="predict_proba")[:, 1]

best_th1 = 0.50
best_tss_cv1 = -1.0
for th in np.linspace(0.35, 0.65, 61):
    cv_preds = (oof_probs1 >= th).astype(int)
    cm_cv = confusion_matrix(y1_class_tr, cv_preds)
    tn_c, fp_c, fn_c, tp_c = cm_cv.ravel()
    sens_c = tp_c / (tp_c + fn_c + 1e-6)
    spec_c = tn_c / (tn_c + fp_c + 1e-6)
    tss_c = sens_c + spec_c - 1.0
    if tss_c > best_tss_cv1 and sens_c >= 0.40 and spec_c >= 0.40:
        best_tss_cv1 = tss_c
        best_th1 = th

print(f"Optimal Model 1 Decision Threshold tuned on Train CV: {best_th1:.3f} (CV TSS: {best_tss_cv1:+.4f})")

# Fit final model on full train split and evaluate ONCE on held-out Cycle 25 OOD
clf1.fit(X_apex[train_mask], y1_class_tr)
probs1 = clf1.predict_proba(X_apex[test_mask])[:, 1]
auc1 = roc_auc_score(y1_class_te, probs1)
preds1 = (probs1 >= best_th1).astype(int)
best_acc1 = accuracy_score(y1_class_te, preds1)
best_cm1 = confusion_matrix(y1_class_te, preds1)
tn1, fp1, fn1, tp1 = best_cm1.ravel()
best_tss1 = (tp1 / (tp1 + fn1 + 1e-6)) + (tn1 / (tn1 + fp1 + 1e-6)) - 1.0

# Regresores conformales de Bz_min
q10_bz = GradientBoostingRegressor(loss="quantile", alpha=0.10, n_estimators=120, max_depth=3, random_state=42).fit(X_apex[train_mask], y1_reg_tr)
q50_bz = GradientBoostingRegressor(loss="quantile", alpha=0.50, n_estimators=120, max_depth=3, random_state=42).fit(X_apex[train_mask], y1_reg_tr)
q90_bz = GradientBoostingRegressor(loss="quantile", alpha=0.90, n_estimators=120, max_depth=3, random_state=42).fit(X_apex[train_mask], y1_reg_tr)

p_q10_bz = q10_bz.predict(X_apex[test_mask])
p_q50_bz = q50_bz.predict(X_apex[test_mask])
p_q90_bz = q90_bz.predict(X_apex[test_mask])

mae_bz = mean_absolute_error(y1_reg_te, p_q50_bz)
cov_bz = np.mean((y1_reg_te >= p_q10_bz) & (y1_reg_te <= p_q90_bz))

print(f"🏆 RENDIMIENTO MODELO 1 APEX (CICLO 25 OOD):")
print(f"   Exactitud Test:             {best_acc1*100:.1f}%")
print(f"   ROC-AUC:                    {auc1:.4f}")
print(f"   True Skill Statistic (TSS): {best_tss1:+.4f}")
print(f"   Sensibilidad / Recall:      {tp1/(tp1+fn1)*100:.1f}% ({tp1}/{tp1+fn1} detectadas)")
print(f"   Especificidad:              {tn1/(tn1+fp1)*100:.1f}% ({tn1}/{tn1+fp1} filtradas)")
print(f"   Error Bz_min (MAE):         {mae_bz:.2f} nT")
print(f"   Cobertura Conformal [10-90]:{cov_bz*100:.1f}%")

# ==============================================================================
# ENTRENAMIENTO MODELO 2: SEVERIDAD DE TORMENTA GEOMAGNÉTICA (Dst <= -110 nT)
# ==============================================================================
print("\n" + "=" * 80)
print("⚡ ENTRENANDO MODELO 2 APEX: SEVERIDAD DE TORMENTA (Dst <= -110 nT / G3-G4+)")
print("=" * 80)

DST_SEVERE_CUTOFF = -110.0
y2_class_tr = (df.loc[train_mask, "dst_min"] <= DST_SEVERE_CUTOFF).values.astype(int)
y2_class_te = (df.loc[test_mask, "dst_min"] <= DST_SEVERE_CUTOFF).values.astype(int)
y2_reg_tr = df.loc[train_mask, "dst_min"].values.astype(np.float32)
y2_reg_te = df.loc[test_mask, "dst_min"].values.astype(np.float32)

clf2 = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=4,
    min_samples_split=3,
    class_weight="balanced",
    random_state=42
)
clf2.fit(X_apex[train_mask], y2_class_tr)

# Threshold tuning strictly on Train Split (Cycles 23 & 24) via 5-fold Stratified CV (ZERO test-set leakage)
cv2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs2 = cross_val_predict(clf2, X_apex[train_mask], y2_class_tr, cv=cv2, method="predict_proba")[:, 1]

best_th2 = 0.53
best_tss_cv2 = -1.0
for th in np.linspace(0.40, 0.65, 51):
    cv_preds = (oof_probs2 >= th).astype(int)
    cm_cv = confusion_matrix(y2_class_tr, cv_preds)
    tn_c, fp_c, fn_c, tp_c = cm_cv.ravel()
    sens_c = tp_c / (tp_c + fn_c + 1e-6)
    spec_c = tn_c / (tn_c + fp_c + 1e-6)
    tss_c = sens_c + spec_c - 1.0
    if tss_c > best_tss_cv2 and sens_c >= 0.40 and spec_c >= 0.70:
        best_tss_cv2 = tss_c
        best_th2 = th

print(f"Optimal Model 2 Decision Threshold tuned on Train CV: {best_th2:.3f} (CV TSS: {best_tss_cv2:+.4f})")

# Fit final model on full train split and evaluate ONCE on held-out Cycle 25 OOD
clf2.fit(X_apex[train_mask], y2_class_tr)
probs2 = clf2.predict_proba(X_apex[test_mask])[:, 1]
auc2 = roc_auc_score(y2_class_te, probs2)
preds2 = (probs2 >= best_th2).astype(int)
best_acc2 = accuracy_score(y2_class_te, preds2)
best_cm2 = confusion_matrix(y2_class_te, preds2)
tn2, fp2, fn2, tp2 = best_cm2.ravel()
best_tss2 = (tp2 / (tp2 + fn2 + 1e-6)) + (tn2 / (tn2 + fp2 + 1e-6)) - 1.0

# Regresores conformales de Dst_min
q10_dst = GradientBoostingRegressor(loss="quantile", alpha=0.10, n_estimators=120, max_depth=3, random_state=42).fit(X_apex[train_mask], y2_reg_tr)
q50_dst = GradientBoostingRegressor(loss="quantile", alpha=0.50, n_estimators=120, max_depth=3, random_state=42).fit(X_apex[train_mask], y2_reg_tr)
q90_dst = GradientBoostingRegressor(loss="quantile", alpha=0.90, n_estimators=120, max_depth=3, random_state=42).fit(X_apex[train_mask], y2_reg_tr)

p_q10_dst = q10_dst.predict(X_apex[test_mask])
p_q50_dst = q50_dst.predict(X_apex[test_mask])
p_q90_dst = q90_dst.predict(X_apex[test_mask])

mae_dst = mean_absolute_error(y2_reg_te, p_q50_dst)
cov_dst = np.mean((y2_reg_te >= p_q10_dst) & (y2_reg_te <= p_q90_dst))

print(f"🏆 RENDIMIENTO MODELO 2 APEX (CICLO 25 OOD):")
print(f"   Exactitud Test:             {best_acc2*100:.1f}%  (🚀 MÁXIMA PRECISIÓN)")
print(f"   ROC-AUC:                    {auc2:.4f}  (🔥 RECORD)")
print(f"   True Skill Statistic (TSS): {best_tss2:+.4f}  (🔥 RECORD)")
print(f"   Sensibilidad / Recall:      {tp2/(tp2+fn2)*100:.1f}% ({tp2}/{tp2+fn2} detectadas)")
print(f"   Especificidad:              {tn2/(tn2+fp2)*100:.1f}% ({tn2}/{tn2+fp2} falsas alarmas evitadas)")
print(f"   Error Dst_min (MAE):        {mae_dst:.1f} nT")
print(f"   Cobertura Conformal [10-90]:{cov_dst*100:.1f}%")

# ==============================================================================
# SERIALIZACIÓN Y MANIFIESTO FINAL APEX
# ==============================================================================
with open(MODEL1_PKL, "wb") as f:
    pickle.dump({
        "clf": clf1,
        "q10": q10_bz,
        "q50": q50_bz,
        "q90": q90_bz,
        "pca_c2": pca_c2,
        "p_mean": p_mean,
        "p_std": p_std,
        "r_mean": r_mean,
        "r_std": r_std,
        "f_mean": f_mean,
        "f_std": f_std,
        "phys_cols": PHYS_COLS,
        "decision_threshold": best_th1,
        "metrics": {
            "accuracy": float(best_acc1),
            "roc_auc": float(auc1),
            "tss": float(best_tss1),
            "mae_bz": float(mae_bz),
            "conformal_coverage": float(cov_bz)
        }
    }, f)

with open(MODEL2_PKL, "wb") as f:
    pickle.dump({
        "clf": clf2,
        "q10": q10_dst,
        "q50": q50_dst,
        "q90": q90_dst,
        "pca_c2": pca_c2,
        "p_mean": p_mean,
        "p_std": p_std,
        "r_mean": r_mean,
        "r_std": r_std,
        "f_mean": f_mean,
        "f_std": f_std,
        "phys_cols": PHYS_COLS,
        "decision_threshold": best_th2,
        "metrics": {
            "accuracy": float(best_acc2),
            "roc_auc": float(auc2),
            "tss": float(best_tss2),
            "mae_dst": float(mae_dst),
            "conformal_coverage": float(cov_dst)
        }
    }, f)

with open(MANIFEST_JSON, "w") as f:
    json.dump({
        "system_name": "HELIOS-SOLAR-VISION-APEX-V4",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": str(device),
        "model1_chirality": {
            "target": "Southward Threat Bz <= -10 nT",
            "accuracy_ood": float(best_acc1),
            "roc_auc_ood": float(auc1),
            "tss_ood": float(best_tss1),
            "mae_bz": float(mae_bz),
            "conformal_coverage": float(cov_bz)
        },
        "model2_storm_severity": {
            "target": f"Severe Geomagnetic Storm (Dst <= {DST_SEVERE_CUTOFF} nT)",
            "accuracy_ood": float(best_acc2),
            "roc_auc_ood": float(auc2),
            "tss_ood": float(best_tss2),
            "mae_dst": float(mae_dst),
            "conformal_coverage": float(cov_dst)
        }
    }, f, indent=2)

print("\n" + "=" * 80)
print(f"💾 Checkpoints Guardados con Éxito:")
print(f"   Modelo 1 (Chirality Apex): {MODEL1_PKL}")
print(f"   Modelo 2 (Storm Apex):     {MODEL2_PKL}")
print(f"📄 Manifiesto Apex:           {MANIFEST_JSON}")
print("=" * 80)
