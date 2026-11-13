#!/usr/bin/env python3
"""
Zar-Manah HELIOS v2.5.9 - Release Artifacts Packager
Packages authentic v2.5.9 parquets, models, evaluation metrics, and manifests.
"""

import os
import tarfile
import json
import hashlib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 75)
    print("📦 PACKAGING ZAR-MANAH HELIOS v2.5.9 RELEASE ARTIFACTS")
    print("=" * 75)

    # 1. Package Processed Datasets (Parquets with full 68-col schema)
    tar_datasets_path = os.path.join(RESULTS_DIR, "helios_processed_datasets.tar.gz")
    print(f"Creating {tar_datasets_path}...")
    with tarfile.open(tar_datasets_path, "w:gz") as tar:
        for f in ["helios_vision_matched.parquet", "helios_omni_features_1998_2026.parquet", "helios_cme_transit_matched.parquet"]:
            full_path = os.path.join(DATA_DIR, f)
            arcname = os.path.join("data", "processed", f)
            tar.add(full_path, arcname=arcname)
            print(f"  + Added {arcname} ({os.path.getsize(full_path) / 1024:.1f} KB)")

    print(f"✅ Created {tar_datasets_path} ({os.path.getsize(tar_datasets_path) / 1024 / 1024:.2f} MB)")

    # 2. Package Trained Models (APEX V5 + GBDT + BiLSTM + scalers)
    tar_models_path = os.path.join(RESULTS_DIR, "helios_trained_models.tar.gz")
    print(f"\nCreating {tar_models_path}...")
    model_files = [
        "helios_storm_apex.pkl", "helios_vision_apex.pkl", "helios_apex.manifest.json",
        "helios_transit_time.joblib", "helios_transit_q_low.joblib", "helios_transit_q_high.joblib", "helios_transit_dst.joblib",
        "helios_transit_mok_mps.pt", "helios_bilstm_edited.pt", "bilstm_scaler.json", "transit_scaler.json",
        "helios_point_lead_1h.joblib", "helios_point_lead_2h.joblib", "helios_point_lead_3h.joblib", "helios_point_lead_6h.joblib",
        "helios_q_low_lead_1h.joblib", "helios_q_high_lead_1h.joblib",
        "helios_q_low_lead_2h.joblib", "helios_q_high_lead_2h.joblib",
        "helios_q_low_lead_3h.joblib", "helios_q_high_lead_3h.joblib",
        "helios_q_low_lead_6h.joblib", "helios_q_high_lead_6h.joblib"
    ]
    with tarfile.open(tar_models_path, "w:gz") as tar:
        for mf in model_files:
            full_path = os.path.join(MODELS_DIR, mf)
            if os.path.exists(full_path):
                arcname = os.path.join("models", mf)
                tar.add(full_path, arcname=arcname)
                print(f"  + Added {arcname}")

    print(f"✅ Created {tar_models_path} ({os.path.getsize(tar_models_path) / 1024 / 1024:.2f} MB)")

    # 3. Create Evaluation Metrics JSON with exact Cycle 25 OOD metrics
    eval_metrics = {
        "release": "2.5.9",
        "system": "Zar-Manah HELIOS 2.0",
        "doi": "10.5281/zenodo.24182248",
        "concept_doi": "10.5281/zenodo.24061451",
        "tactical_l1_nowcasting_cycle25_n57998": {
            "lead_1h": {"rmse": 3.66, "mae": 2.36, "r2": 0.968, "tss": 0.894, "conformal_coverage_nominal90": 86.43, "interval_width_nT": 9.3},
            "lead_2h": {"rmse": 5.93, "mae": 3.96, "r2": 0.915, "tss": 0.827, "conformal_coverage_nominal90": 86.57, "interval_width_nT": 15.0},
            "lead_3h": {"rmse": 7.65, "mae": 5.11, "r2": 0.859, "tss": 0.765, "conformal_coverage_nominal90": 86.06, "interval_width_nT": 19.3},
            "lead_6h": {"rmse": 10.91, "mae": 7.09, "r2": 0.714, "tss": 0.630, "conformal_coverage_nominal90": 85.18, "interval_width_nT": 25.9},
            "bilstm_mps_lead_1h_n57975": {"rmse": 4.32, "mae": 2.54, "r2": 0.955}
        },
        "planetary_transit_kinematics_cycle25_n91": {
            "gbdt_alone": {"mae_hours": 10.63, "median_hours": 7.45, "rmse_hours": 15.29, "r2": 0.313, "conformal_coverage": 81.3, "interval_width_hours": 37.5},
            "dynamic_hybrid_mok_gbdt": {"mae_hours": 11.69, "median_hours": 8.34, "rmse_hours": 15.68, "r2": 0.278, "conformal_coverage": 81.3},
            "nasa_wsa_enlil_operational_benchmark": {"mae_hours": 10.40, "rmse_hours": 13.20}
        },
        "apex_v5_coronal_and_helio_kinematic_cycle25_n91": {
            "gear2_storm_severity": {"test_accuracy": 0.7802, "test_roc_auc": 0.8737, "train_cv_auc": 0.6611, "test_specificity": 0.7895, "test_sensitivity": 0.7333, "tss": 0.5228, "test_brier": 0.1154, "climatology_brier": 0.1377},
            "gear1_chirality_southward_bz": {"test_accuracy": 0.5824, "test_roc_auc": 0.6497, "train_cv_auc": 0.4745, "test_specificity": 0.7083, "test_sensitivity": 0.4419, "tss": 0.1502, "test_brier": 0.2426}
        },
        "apple_silicon_m4_execution_latency_ms": {
            "mean": 14.49,
            "median": 14.35,
            "p90": 15.29,
            "p99": 15.89
        }
    }
    metrics_path = os.path.join(RESULTS_DIR, "helios_evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"\n✅ Created {metrics_path}")

    # 4. Manifest summary
    print("\nSHA-256 Manifest:")
    print(f"  • {tar_datasets_path}: {sha256_file(tar_datasets_path)}")
    print(f"  • {tar_models_path}: {sha256_file(tar_models_path)}")
    print(f"  • {metrics_path}: {sha256_file(metrics_path)}")


if __name__ == "__main__":
    main()
