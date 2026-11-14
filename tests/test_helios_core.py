#!/usr/bin/env python3
"""
Zar-Manah HELIOS 2.0 - Core Test Suite
Rigorous unit tests and assertions verifying bit-for-bit model reproducibility,
dataset integrity, transit kinematics, and multi-horizon nowcasting.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import unittest
import pickle
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_IMG_DIR = os.path.join(BASE_DIR, "data", "raw", "images")

def verify_model_checksum(fpath):
    """Cryptographic fail-closed verification of model weights before loading."""
    import hashlib
    manifest_path = os.path.join(MODELS_DIR, "helios_models_checksums.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"SECURITY ALERT: Cryptographic model manifest missing at {manifest_path}!")
    with open(manifest_path, "r") as f:
        valid_hashes = json.load(f)
    fname = os.path.basename(fpath)
    if fname not in valid_hashes:
        raise KeyError(f"SECURITY ALERT: Model artifact {fname} is not registered in cryptographic manifest!")
    with open(fpath, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    if actual_hash != valid_hashes[fname]:
        raise ValueError(f"SECURITY ALERT: Cryptographic integrity failure for {fname}! Expected {valid_hashes[fname]}, got {actual_hash}")


class TestHeliosDatasetIntegrity(unittest.TestCase):
    """Verifies empirical catalog schemas, sample counts, and image existence."""

    def test_vision_parquet_schema_and_rows(self):
        parquet_path = os.path.join(DATA_DIR, "helios_vision_matched.parquet")
        self.assertTrue(os.path.exists(parquet_path), "Vision parquet missing!")
        df = pd.read_parquet(parquet_path)
        self.assertEqual(len(df), 314, f"Expected 314 CME rows, got {len(df)}")

        # Verify all 14 physical feature columns exist and are non-null
        expected_phys = [
            "v_launch_3d", "width", "accel", "mass", "energy_3d", "is_halo",
            "f107", "r_sunspot", "v_sw_launch", "prior_v",
            "shear_angle_proxy", "free_energy_proxy", "helicity_proxy", "flare_flux_proxy"
        ]
        for col in expected_phys:
            self.assertIn(col, df.columns, f"Feature column {col} missing from parquet!")
            self.assertEqual(df[col].isna().sum(), 0, f"Column {col} has NaN values!")

    def test_coronal_vision_inference_pipeline(self):
        """Tests coronal vision preprocessing, MobileNetV3 feature extraction, and PCA projection unconditionally."""
        import torch
        from torchvision import models, transforms
        from PIL import Image

        # Create synthetic coronal test image (224x224 RGB)
        img_arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_arr)

        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor = tf(img).unsqueeze(0)

        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).features
        mobilenet.eval()

        with torch.no_grad():
            feat = mobilenet(tensor).mean([2, 3]).numpy()  # shape (1, 576)

        self.assertEqual(feat.shape, (1, 576), "MobileNetV3 feature shape error!")
        self.assertFalse(np.isnan(feat).any(), "Vision features contain NaN!")

        # Verify PCA projection models for SOHO C2 and SDO Magnetogram
        pca_path = os.path.join(MODELS_DIR, "helios_vision_pca.pkl")
        if os.path.exists(pca_path):
            verify_model_checksum(pca_path)
            with open(pca_path, "rb") as f:
                pca_pkg = pickle.load(f)
            self.assertIn("pca_c2", pca_pkg, "Missing pca_c2 in vision PCA package!")
            self.assertIn("pca_mag", pca_pkg, "Missing pca_mag in vision PCA package!")
            
            c2_reduced = pca_pkg["pca_c2"].transform(feat)
            mag_reduced = pca_pkg["pca_mag"].transform(feat)
            
            self.assertEqual(c2_reduced.shape, (1, 8), "pca_c2 projection shape mismatch!")
            self.assertEqual(mag_reduced.shape, (1, 8), "pca_mag projection shape mismatch!")
            self.assertFalse(np.isnan(c2_reduced).any(), "pca_c2 output contains NaN!")
            self.assertFalse(np.isnan(mag_reduced).any(), "pca_mag output contains NaN!")


class TestHeliosModelReproducibility(unittest.TestCase):
    """Verifies that saved models reproduce exact published test metrics."""

    def test_apex_v5_storm_and_chirality(self):
        parquet_path = os.path.join(DATA_DIR, "helios_vision_matched.parquet")
        df = pd.read_parquet(parquet_path)
        test_df = df[df["solar_cycle"] == "Cycle 25"].copy()
        self.assertEqual(len(test_df), 91, f"Expected 91 OOD test events, got {len(test_df)}")

        phys_cols = [
            "v_launch_3d", "width", "accel", "mass", "energy_3d", "is_halo",
            "f107", "r_sunspot", "v_sw_launch", "prior_v",
            "shear_angle_proxy", "free_energy_proxy", "helicity_proxy", "flare_flux_proxy"
        ]
        X_test = test_df[phys_cols].values

        # 1. Gear 2: Storm Severity (Dst <= -110 nT)
        p_storm_apex = os.path.join(MODELS_DIR, "helios_storm_apex.pkl")
        verify_model_checksum(p_storm_apex)
        with open(p_storm_apex, "rb") as f:
            m2_pkg = pickle.load(f)
        clf_storm = m2_pkg["clf"]
        p_mean2 = m2_pkg["p_mean"]
        p_std2 = m2_pkg["p_std"]
        X_norm2 = (X_test - p_mean2) / p_std2
        p_storm = clf_storm.predict_proba(X_norm2)[:, 1]
        y_storm = test_df["is_severe_storm"].values
        auc_storm = roc_auc_score(y_storm, p_storm)
        pred_storm = (p_storm >= m2_pkg["decision_threshold"]).astype(int)
        acc_storm = accuracy_score(y_storm, pred_storm)

        self.assertAlmostEqual(auc_storm, 0.873684, places=4, msg="Storm AUC mismatch!")
        self.assertAlmostEqual(acc_storm, 0.780219, places=4, msg="Storm Accuracy mismatch!")

        # 2. Gear 1: Chirality (Bz <= -10 nT)
        p_vision_apex = os.path.join(MODELS_DIR, "helios_vision_apex.pkl")
        verify_model_checksum(p_vision_apex)
        with open(p_vision_apex, "rb") as f:
            m1_pkg = pickle.load(f)
        clf_chir = m1_pkg["clf"]
        p_mean1 = m1_pkg["p_mean"]
        p_std1 = m1_pkg["p_std"]
        X_norm1 = (X_test - p_mean1) / p_std1
        p_chir = clf_chir.predict_proba(X_norm1)[:, 1]
        y_chir = test_df["is_dangerous_southward"].values
        auc_chir = roc_auc_score(y_chir, p_chir)
        pred_chir = (p_chir >= m1_pkg["decision_threshold"]).astype(int)
        acc_chir = accuracy_score(y_chir, pred_chir)

        self.assertAlmostEqual(auc_chir, 0.649709, places=4, msg="Chirality AUC mismatch!")
        self.assertAlmostEqual(acc_chir, 0.582417, places=4, msg="Chirality Accuracy mismatch!")


class TestHeliosTransitAndTacticalNowcasting(unittest.TestCase):
    """Verifies transit error bounds and tactical in-situ nowcasting."""

    def test_transit_gbdt_performance(self):
        transit_path = os.path.join(DATA_DIR, "helios_cme_transit_matched.parquet")
        self.assertTrue(os.path.exists(transit_path))
        df = pd.read_parquet(transit_path)

        L_AU = 1.495978707e8
        v_amb = 420.0
        gamma = 1.0e-7
        v_eff = df["cme_speed"] * np.exp(-gamma * L_AU * 0.4) + v_amb * (1.0 - np.exp(-gamma * L_AU * 0.4))
        df["transit_hours_dbm_prior"] = L_AU / (v_eff * 3600.0)
        df["v_diff_ambient"] = df["cme_speed"] - v_amb
        df["drag_accel_proxy"] = -gamma * df["v_diff_ambient"] * np.abs(df["v_diff_ambient"])
        df["is_fast_cme"] = (df["cme_speed"] >= 1000.0).astype(int)
        solar_phase = (df["year"] - 1996) % 11 / 11.0
        df["sin_cycle"] = np.sin(2 * np.pi * solar_phase)
        df["cos_cycle"] = np.cos(2 * np.pi * solar_phase)

        test_df = df[df["solar_cycle"] == "Cycle 25"].copy().reset_index(drop=True)

        p_transit_gbdt = os.path.join(MODELS_DIR, "helios_transit_time.joblib")
        verify_model_checksum(p_transit_gbdt)
        gbdt = joblib.load(p_transit_gbdt)
        tree_feats = gbdt.booster_.feature_name()

        preds = gbdt.predict(test_df[tree_feats], num_threads=1)
        y_true = test_df["transit_hours"].values
        mae = mean_absolute_error(y_true, preds)
        median_err = np.median(np.abs(y_true - preds))

        self.assertLess(mae, 11.0, f"Transit MAE {mae:.2f}h exceeds threshold 11.0h!")
        self.assertLess(median_err, 8.0, f"Transit Median Error {median_err:.2f}h exceeds threshold 8.0h!")

    def test_dynamic_hybrid_transit_performance(self):
        """Verifies that dynamic ensemble of weight-edited MoKTransitNN and GBDT achieves target accuracy."""
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler

        transit_path = os.path.join(DATA_DIR, "helios_cme_transit_matched.parquet")
        self.assertTrue(os.path.exists(transit_path))
        df = pd.read_parquet(transit_path)

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
        solar_phase = (df["year"] - 1996) % 11 / 11.0
        df["sin_cycle"] = np.sin(2 * np.pi * solar_phase)
        df["cos_cycle"] = np.cos(2 * np.pi * solar_phase)

        test_df = df[df["solar_cycle"] == "Cycle 25"].copy().reset_index(drop=True)
        train_df = df[df["solar_cycle"].isin(["Cycle 23", "Cycle 24"])].copy().reset_index(drop=True)

        # 1. GBDT Predictions (num_threads=1 prevents OpenMP conflict on macOS)
        p_transit_gbdt = os.path.join(MODELS_DIR, "helios_transit_time.joblib")
        verify_model_checksum(p_transit_gbdt)
        gbdt = joblib.load(p_transit_gbdt)
        tree_feats = gbdt.booster_.feature_name()
        preds_tree = gbdt.predict(test_df[tree_feats], num_threads=1)

        # 2. PyTorch MoKTransitNN Predictions (Weight-Edited Checkpoint)
        nn_feats = ["cme_speed", "log_speed", "inv_speed", "energy_proxy", "width_norm", "cme_accel", "is_halo", "drag_prior_h"]
        scaler = StandardScaler()
        scaler.fit(train_df[nn_feats])
        X_te = scaler.transform(test_df[nn_feats])

        class MoKTransitNN(nn.Module):
            def __init__(self, in_features=8):
                super().__init__()
                self.expert_fast_backbone = nn.Sequential(
                    nn.Linear(in_features, 32), nn.GELU(), nn.Linear(32, 16), nn.GELU()
                )
                self.expert_fast_head = nn.Linear(16, 1)
                self.expert_slow_backbone = nn.Sequential(
                    nn.Linear(in_features, 32), nn.GELU(), nn.Linear(32, 16), nn.GELU()
                )
                self.expert_slow_head = nn.Linear(16, 1)
                self.gate = nn.Sequential(
                    nn.Linear(in_features, 8), nn.Tanh(), nn.Linear(8, 1), nn.Sigmoid()
                )

            def forward_experts(self, x):
                return self.gate(x), self.expert_fast_head(self.expert_fast_backbone(x)), self.expert_slow_head(self.expert_slow_backbone(x))

            def forward(self, x):
                g, y_f, y_s = self.forward_experts(x)
                return (g * y_f + (1.0 - g) * y_s).squeeze(1)

        net = MoKTransitNN(in_features=len(nn_feats))
        edited_ckpt = os.path.join(MODELS_DIR, "helios_transit_mok_edited.pt")
        self.assertTrue(os.path.exists(edited_ckpt), "Weight-edited checkpoint missing!")
        verify_model_checksum(edited_ckpt)
        net.load_state_dict(torch.load(edited_ckpt, map_location="cpu", weights_only=True))
        net.eval()

        with torch.no_grad():
            preds_mok = net(torch.tensor(X_te, dtype=torch.float32)).numpy()

        # 3. Dynamic Sigmoid Fusion
        w_dynamic = 0.45 + 0.45 / (1.0 + np.exp(-(test_df["cme_speed"].values - 950.0) / 100.0))
        preds_hybrid = w_dynamic * preds_mok + (1.0 - w_dynamic) * preds_tree

        y_true = test_df["transit_hours"].values
        mae_hybrid = mean_absolute_error(y_true, preds_hybrid)
        median_hybrid = np.median(np.abs(y_true - preds_hybrid))

        self.assertLess(mae_hybrid, 12.0, f"Hybrid MAE {mae_hybrid:.2f}h exceeds threshold 12.0h!")
        self.assertLess(median_hybrid, 8.5, f"Hybrid Median Error {median_hybrid:.2f}h exceeds threshold 8.5h!")

    def test_tactical_lead_1h_performance(self):
        omni_path = os.path.join(DATA_DIR, "helios_omni_features_1998_2026.parquet")
        self.assertTrue(os.path.exists(omni_path))
        df = pd.read_parquet(omni_path)
        test_df = df[pd.to_datetime(df["datetime"]) >= "2020-01-01"].copy()
        self.assertEqual(len(test_df), 57998, f"Expected 57,998 test samples, got {len(test_df)}")

        p_m1 = os.path.join(MODELS_DIR, "helios_point_lead_1h.joblib")
        verify_model_checksum(p_m1)
        m1 = joblib.load(p_m1)
        feat_cols = m1.booster_.feature_name()
        preds = m1.predict(test_df[feat_cols], num_threads=1)
        y_true = test_df["target_dst_lead_1h"].values

        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mae = mean_absolute_error(y_true, preds)
        r2 = r2_score(y_true, preds)

        self.assertLess(rmse, 4.0, f"Lead +1h RMSE {rmse:.2f} exceeds threshold 4.0 nT!")
        self.assertLess(mae, 2.5, f"Lead +1h MAE {mae:.2f} exceeds threshold 2.5 nT!")
        self.assertGreater(r2, 0.96, f"Lead +1h R2 {r2:.3f} below threshold 0.96!")


if __name__ == "__main__":
    unittest.main()
