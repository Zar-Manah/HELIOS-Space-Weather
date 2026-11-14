#!/usr/bin/env python3
"""
Zar-Manah HELIOS 2.0 - CERN Zenodo Official Publisher
Publishes clean official release v2.5.11 under Concept DOI 10.5281/zenodo.24215019
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

TOKEN = "NOEMa0J0Wsr7R8r0xKmLqENUzKGrYAw9OPjIUuCtjFJHHWFpkZY4k7dTfM4s"
DRAFT_ID = 24215020
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
    print("🚀 PUBLISHING CLEAN OFFICIAL RELEASE TO CERN ZENODO")
    print(f"Draft ID:        {DRAFT_ID}")
    print(f"Target DOI:      {VERSION_DOI}")
    print(f"Concept DOI:     {CONCEPT_DOI}")
    print("=" * 80)
    
    # 1. Fetch draft info
    r_draft = requests.get(f"{ZENODO_API}/deposit/depositions/{DRAFT_ID}", headers=headers)
    if r_draft.status_code != 200:
        print(f"❌ Failed to fetch draft details: {r_draft.status_code} {r_draft.text}")
        sys.exit(1)
        
    draft_data = r_draft.json()
    bucket_url = draft_data.get("links", {}).get("bucket")
    print(f"✓ Bucket URL: {bucket_url}")
    
    # 2. Upload all 28 clean files
    print(f"\nUploading {len(FILES_TO_UPLOAD)} files to bucket...")
    uploaded_info = {}
    
    for rel_path, remote_name in FILES_TO_UPLOAD:
        local_file = BASE_DIR / rel_path
        if not local_file.exists():
            print(f"❌ Missing local file: {local_file}")
            sys.exit(1)
            
        data = local_file.read_bytes()
        local_md5 = hashlib.md5(data).hexdigest()
        local_sha = hashlib.sha256(data).hexdigest()
        
        target_url = f"{bucket_url}/{remote_name}"
        print(f"  -> Uploading {remote_name} ({len(data):,} bytes)...", end="", flush=True)
        
        up_r = requests.put(
            target_url,
            data=data,
            headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/octet-stream"}
        )
        if up_r.status_code in (200, 201):
            server_md5 = up_r.json().get("checksum", "").replace("md5:", "")
            assert local_md5 == server_md5, f"Checksum mismatch on {remote_name}: {local_md5} vs {server_md5}"
            print(f" [OK: MD5={local_md5}]")
            uploaded_info[remote_name] = {
                "size": len(data),
                "md5": server_md5,
                "sha256": local_sha
            }
        else:
            print(f" ❌ Failed: {up_r.status_code} {up_r.text}")
            sys.exit(1)
            
    print(f"\n✓ All {len(uploaded_info)} files uploaded and verified bit-for-bit.")
    
    # 3. Update metadata
    print("\nUpdating deposition metadata for v2.5.11...")
    with open(BASE_DIR / "paper" / "manuscript.md", "r", errors="ignore") as f:
        manuscript_text = f.read()
        
    abstract_start = manuscript_text.find("## Abstract")
    abstract_end = manuscript_text.find("## 1. Introduction")
    if abstract_start != -1 and abstract_end != -1:
        abstract = manuscript_text[abstract_start+11:abstract_end].strip()
    else:
        abstract = "Zar-Manah HELIOS 2.0: Multi-Gear Architecture for Real-Time Solar Storm Impact & CME Arrival Prediction."

    meta_payload = {
        "metadata": {
            "title": "Zar-Manah HELIOS 2.0: Multi-Gear Architecture for Real-Time Solar Storm Impact & CME Arrival Prediction (v2.5.11)",
            "upload_type": "publication",
            "publication_type": "preprint",
            "description": f"<p>{abstract}</p>",
            "version": "2.5.11",
            "creators": [
                {
                    "name": "Manah, Zar",
                    "affiliation": "Independent Researcher, Madrid, Spain",
                    "orcid": "0009-0007-8443-8999"
                }
            ],
            "access_right": "open",
            "license": "cc-by-4.0",
            "keywords": [
                "Space Weather",
                "Solar Storm Forecasting",
                "Coronal Mass Ejection",
                "Geomagnetic Disturbance",
                "Machine Learning",
                "Conformal Prediction",
                "Apple Silicon M4"
            ],
            "related_identifiers": [
                {
                    "identifier": "https://github.com/Zar-Manah/HELIOS-Space-Weather",
                    "relation": "isSupplementTo",
                    "scheme": "url"
                }
            ]
        }
    }
    
    meta_r = requests.put(f"{ZENODO_API}/deposit/depositions/{DRAFT_ID}", json=meta_payload, headers=headers)
    if meta_r.status_code == 200:
        print("✓ Metadata updated successfully.")
    else:
        print(f"❌ Metadata update failed: {meta_r.status_code} {meta_r.text}")
        sys.exit(1)
        
    # 4. Publish action
    print("\nPublishing deposition officially to CERN Zenodo...")
    pub_r = requests.post(f"{ZENODO_API}/deposit/depositions/{DRAFT_ID}/actions/publish", headers=headers)
    if pub_r.status_code == 202:
        pub_data = pub_r.json()
        doi = pub_data.get("doi")
        conceptdoi = pub_data.get("conceptdoi")
        record_url = pub_data.get("links", {}).get("record_html")
        print("\n" + "=" * 80)
        print("🎉 OFFICIAL CLEAN DEPOSITION PUBLISHED SUCCESSFULLY ON CERN ZENODO!")
        print(f"✓ Version DOI:  {doi}")
        print(f"✓ Concept DOI:  {conceptdoi}")
        print(f"✓ Record URL:   {record_url}")
        print("=" * 80)
        
        # Save results/zenodo_record.json
        record_info = {
            "dep_id": DRAFT_ID,
            "doi": doi,
            "conceptdoi": conceptdoi,
            "record_url": record_url,
            "version": "2.5.11",
            "published_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "status": f"v2.5.11 published on Zenodo CERN OpenAIRE under permanent Concept DOI {conceptdoi}",
            "server_verified_files": uploaded_info
        }
        with open(BASE_DIR / "results" / "zenodo_record.json", "w") as jf:
            json.dump(record_info, jf, indent=2)
        print("✓ Updated results/zenodo_record.json")
    else:
        print(f"❌ Publish failed: {pub_r.status_code} {pub_r.text}")
        sys.exit(1)

if __name__ == "__main__":
    main()
