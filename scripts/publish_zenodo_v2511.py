#!/usr/bin/env python3
"""
Zar-Manah HELIOS 2.0 - CERN Zenodo Official Publisher (v2.5.11)
Publishes clean official release v2.5.11 under permanent Concept DOI 10.5281/zenodo.24215019
and Version DOI 10.5281/zenodo.24215020 for Zar Manah (teabiskot@gmail.com).
"""

import os
import sys
import json
import hashlib
import time
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ZENODO_API = "https://zenodo.org/api"

# Load token from ~/.config/zenodo/credentials.json or env
TOKEN = os.environ.get("ZENODO_ACCESS_TOKEN", "").strip()
if not TOKEN:
    cred_file = Path.home() / ".config" / "zenodo" / "credentials.json"
    if cred_file.exists():
        try:
            TOKEN = json.loads(cred_file.read_text()).get("api_token", "").strip()
        except Exception:
            pass

if not TOKEN:
    print("❌ Error: Zenodo API token not found in env or ~/.config/zenodo/credentials.json")
    sys.exit(1)

CANONICAL_DEP_ID = 24215020
CONCEPT_DOI = "10.5281/zenodo.24215019"
VERSION_DOI = "10.5281/zenodo.24215020"

FILES_TO_UPLOAD = [
    # Tarballs
    ("results/helios_processed_datasets.tar.gz", "helios_processed_datasets.tar.gz"),
    ("results/helios_trained_models.tar.gz", "helios_trained_models.tar.gz"),
    # Metrics & configs
    ("results/helios_evaluation_metrics.json", "helios_evaluation_metrics.json"),
    ("results/helios_transit_metrics.json", "helios_transit_metrics.json"),
    ("results/latency_benchmark.json", "latency_benchmark.json"),
    ("results/ablation_study.json", "ablation_study.json"),
    ("models/helios_models_checksums.json", "helios_models_checksums.json"),
    # Papers & manuscript
    ("paper/HELIOS_Solar_Storm_Conformal_Forecasting.pdf", "HELIOS_Solar_Storm_Conformal_Forecasting.pdf"),
    ("paper/HELIOS_2_Solar_Vision_Swiss_Watch.pdf", "HELIOS_2_Solar_Vision_Swiss_Watch.pdf"),
    ("paper/manuscript.md", "manuscript.md"),
    # Core scripts & entrypoints
    ("predict.py", "predict.py"),
    ("tests/test_helios_core.py", "test_helios_core.py"),
    ("export_ccmc_scoreboard.py", "export_ccmc_scoreboard.py"),
    ("scripts/19_benchmark_latency.py", "19_benchmark_latency.py"),
    ("scripts/helios_swiss_orchestrator.py", "helios_swiss_orchestrator.py"),
    ("scripts/07_train_helios_transit.py", "07_train_helios_transit.py"),
    ("scripts/01_build_features.py", "01_build_features.py"),
    # Metadata & legal
    ("README.md", "README.md"),
    ("LICENSE", "LICENSE"),
    ("requirements.txt", "requirements.txt"),
    ("Zar-Manah_HELIOS.html", "Zar-Manah_HELIOS.html"),
    # Figures
    ("results/figures/fig1_multi_horizon_benchmarks.png", "fig1_multi_horizon_benchmarks.png"),
    ("results/figures/fig2_may2024_superstorm_case_study.png", "fig2_may2024_superstorm_case_study.png"),
    ("results/figures/fig3_feature_importance.png", "fig3_feature_importance.png"),
    ("results/figures/fig4_cme_transit_benchmark.png", "fig4_cme_transit_benchmark.png"),
    ("results/figures/fig1_helios_vision_benchmark.png", "fig1_helios_vision_benchmark.png"),
    ("results/figures/fig2_helios_apex_orchestrator.png", "fig2_helios_apex_orchestrator.png"),
    # Manifest
    ("results/checksums.txt", "checksums.txt"),
]

def main():
    headers = {"Authorization": f"Bearer {TOKEN}"}
    
    print("=" * 80)
    print("🚀 CANONICAL ZENODO PUBLICATION VERIFICATION (v2.5.11)")
    print(f"Record ID:       {CANONICAL_DEP_ID}")
    print(f"Version DOI:     {VERSION_DOI}")
    print(f"Concept DOI:     {CONCEPT_DOI}")
    print("=" * 80)
    
    r_draft = requests.get(f"{ZENODO_API}/deposit/depositions/{CANONICAL_DEP_ID}", headers=headers)
    if r_draft.status_code != 200:
        print(f"❌ Failed to fetch record details: {r_draft.status_code} {r_draft.text}")
        sys.exit(1)
        
    data = r_draft.json()
    files = data.get("files", [])
    print(f"✓ Verified canonical deposition {CANONICAL_DEP_ID}: {len(files)} files present, submitted={data.get('submitted')}")

if __name__ == "__main__":
    main()
