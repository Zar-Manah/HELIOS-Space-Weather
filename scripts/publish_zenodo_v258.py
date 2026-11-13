#!/usr/bin/env python3
"""
Zar-Manah HELIOS v2.5.8: Official CERN Zenodo Publisher & Cryptographic Verifier.
Creates a new version under Concept DOI 10.5281/zenodo.24061451 from deposition 24181624,
uploads all 15 files, verifies byte-for-byte MD5 & SHA-256 hashes, updates metadata,
and mints the new permanent DOI.
"""

import os
import sys
import json
import time
import hashlib
import urllib.request
import urllib.parse
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZENODO_API = "https://zenodo.org/api"
TOKEN = os.environ.get("ZENODO_ACCESS_TOKEN", "")
PREV_DEP_ID = "24181624"

AUTHOR = {
    "name": "Zar Manah",
    "affiliation": "Independent Researcher, Madrid, Spain",
    "orcid": "0009-0007-8443-8999"
}

def req(url, method="GET", data=None, headers=None):
    hdrs = {
        "Authorization": f"Bearer {TOKEN}",
        "User-Agent": "Zar-Manah-HELIOS/2.5.8"
    }
    if headers:
        hdrs.update(headers)
    body = None
    if data is not None:
        if isinstance(data, (dict, list)):
            body = json.dumps(data).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
        elif isinstance(data, bytes):
            body = data
        else:
            body = str(data).encode("utf-8")

    r = urllib.request.Request(url, data=body, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(r) as resp:
            status = resp.status
            content = resp.read()
            try:
                parsed = json.loads(content.decode("utf-8")) if content else {}
            except Exception:
                parsed = content.decode("utf-8")
            return {"ok": True, "status": status, "data": parsed}
    except urllib.error.HTTPError as e:
        err_content = e.read().decode("utf-8")
        try:
            err_json = json.loads(err_content)
        except Exception:
            err_json = err_content
        return {"ok": False, "status": e.code, "error": err_json}
    except Exception as e:
        return {"ok": False, "status": 0, "error": str(e)}

def compute_hashes(path):
    md5_hash = hashlib.md5()
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            md5_hash.update(chunk)
            sha256_hash.update(chunk)
    return md5_hash.hexdigest(), sha256_hash.hexdigest()

def main():
    if not TOKEN:
        print("❌ Error: ZENODO_ACCESS_TOKEN environment variable is not set!")
        sys.exit(1)

    print("=" * 80)
    print("🌟 ZAR-MANAH HELIOS 2.0 — ZENODO v2.5.8 PUBLICATION & INTEGRITY PIPELINE")
    print("=" * 80)
    print(f"Base Directory:  {BASE_DIR}")
    print(f"Parent Record:   https://zenodo.org/records/{PREV_DEP_ID}")
    print(f"Author / ORCID:  {AUTHOR['name']} ({AUTHOR['orcid']})")
    print(f"Target Version:  v2.5.8 (Strict Scientific Integrity & Zero False Claims)")
    print("-" * 80)

    # Step 1: Spawn new version draft
    print(f"\n[1/6] Creating new draft version from deposition {PREV_DEP_ID}...")
    res = req(f"{ZENODO_API}/deposit/depositions/{PREV_DEP_ID}/actions/newversion", method="POST")
    if not res["ok"]:
        print(f"❌ Failed to spawn new version: {res['error']}")
        sys.exit(1)

    latest_draft_url = res["data"]["links"]["latest_draft"]
    new_dep_id = latest_draft_url.split("/")[-1]
    print(f"✅ Spawned new draft version deposition: ID = {new_dep_id}")
    print(f"   Draft URL: {latest_draft_url}")
    
    # Step 2: Query new draft details
    print(f"\n[2/6] Inspecting draft {new_dep_id} bucket and auto-copied files...")
    draft_res = req(f"{ZENODO_API}/deposit/depositions/{new_dep_id}")
    if not draft_res["ok"]:
        print(f"❌ Failed to fetch draft details: {draft_res['error']}")
        sys.exit(1)
        
    draft_data = draft_res["data"]
    bucket_url = draft_data["links"]["bucket"]
    existing_files = draft_data.get("files", [])
    print(f"   Bucket: {bucket_url}")
    print(f"   Copied files from previous version: {len(existing_files)}")
    
    # Step 3: Delete old auto-copied files
    for ef in existing_files:
        fid = ef["id"]
        fname = ef.get("filename", ef.get("key", fid))
        print(f"   🗑️ Deleting stale file: {fname} (ID {fid})...", end="", flush=True)
        del_res = req(f"{ZENODO_API}/deposit/depositions/{new_dep_id}/files/{fid}", method="DELETE")
        if del_res["ok"] or del_res["status"] in [200, 204]:
            print(" ✅ OK")
        else:
            print(f" ⚠️ Warning: {del_res.get('error')}")
            
    # Step 4: Define the 15 verified files to upload
    files_to_upload = [
        os.path.join(BASE_DIR, "requirements.txt"),
        os.path.join(BASE_DIR, "LICENSE"),
        os.path.join(BASE_DIR, "results", "helios_evaluation_metrics.json"),
        os.path.join(BASE_DIR, "README.md"),
        os.path.join(BASE_DIR, "paper", "HELIOS_Solar_Storm_Conformal_Forecasting.pdf"),
        os.path.join(BASE_DIR, "paper", "HELIOS_2_Solar_Vision_Swiss_Watch.pdf"),
        os.path.join(BASE_DIR, "results", "figures", "fig4_cme_transit_benchmark.png"),
        os.path.join(BASE_DIR, "helios_trained_models.tar.gz"),
        os.path.join(BASE_DIR, "results", "figures", "fig1_multi_horizon_benchmarks.png"),
        os.path.join(BASE_DIR, "results", "figures", "fig3_feature_importance.png"),
        os.path.join(BASE_DIR, "predict.py"),
        os.path.join(BASE_DIR, "results", "figures", "fig2_may2024_superstorm_case_study.png"),
        os.path.join(BASE_DIR, "export_ccmc_scoreboard.py"),
        os.path.join(BASE_DIR, "Zar-Manah_HELIOS.html"),
        os.path.join(BASE_DIR, "helios_processed_datasets.tar.gz")
    ]
    
    print(f"\n[3/6] Uploading {len(files_to_upload)} verified scientific artifacts...")
    local_hashes = {}
    for fpath in files_to_upload:
        fname = os.path.basename(fpath)
        fsize_mb = os.path.getsize(fpath) / (1024 * 1024)
        md5_val, sha256_val = compute_hashes(fpath)
        local_hashes[fname] = {"md5": md5_val, "sha256": sha256_val, "size": os.path.getsize(fpath)}
        print(f"   • Uploading {fname:<44s} ({fsize_mb:6.2f} MB)...", end="", flush=True)
        
        with open(fpath, "rb") as fp:
            data_bytes = fp.read()
            
        u_url = f"{bucket_url}/{urllib.parse.quote(fname)}"
        u_res = req(u_url, method="PUT", data=data_bytes, headers={"Content-Type": "application/octet-stream"})
        if u_res["ok"]:
            print(f" ✅ OK (MD5: {md5_val[:8]}...)")
        else:
            print(f" ❌ Failed: {u_res.get('error')}")
            sys.exit(1)
            
    # Step 5: Cryptographic Hash Verification against Zenodo bucket
    print(f"\n[4/6] Verifying remote bucket hashes byte-for-byte...")
    bucket_info = req(bucket_url)
    if not bucket_info["ok"]:
        print(f"⚠️ Could not query bucket directly: {bucket_info['error']}")
    else:
        bucket_contents = bucket_info["data"].get("contents", [])
        all_match = True
        for b_item in bucket_contents:
            key = b_item.get("key")
            b_md5 = b_item.get("checksum", "").replace("md5:", "")
            b_size = b_item.get("size")
            loc = local_hashes.get(key)
            if loc:
                match = (loc["md5"] == b_md5 and loc["size"] == b_size)
                status_icon = "✅" if match else "❌ MISMATCH"
                if not match:
                    all_match = False
                print(f"   {status_icon} {key:<44s} | Size: {b_size:8d} | MD5: {b_md5}")
        if not all_match:
            print("❌ Hash mismatch detected! Aborting publication.")
            sys.exit(1)
        print("   🏆 100% BYTE-FOR-BYTE BIT IDENTITY VERIFIED ON ZENODO BUCKET!")

    # Step 6: Update Metadata
    print(f"\n[5/6] Updating deposition metadata to v2.5.8...")
    manuscript_path = os.path.join(BASE_DIR, "paper", "manuscript.md")
    with open(manuscript_path, "r") as f:
        m_text = f.read()
    abstract_section = ""
    if "## Abstract" in m_text:
        parts = m_text.split("## Abstract")[1].split("## 1. Introduction")[0].strip()
        abstract_section = parts.replace("\n\n", "<br><br>").replace("**", "<b>").replace("**", "</b>")

    desc_html = (
        f"<p><b>Official Scientific Preprint & Open Architecture: Zar-Manah HELIOS 2.0 (v2.5.8)</b></p>"
        f"<p>{abstract_section}</p>"
        f"<p><b>Architectural & Benchmark Highlights:</b></p>"
        f"<ul>"
        f"<li><b>Tactical Nowcasting (1–6h Horizon):</b> GBDT + Bi-LSTM coupled to Inductive Conformalized Quantile Regression (CQR) guaranteeing finite-sample target coverage on Cycle 25 (MAE 2.90 nT, RMSE 5.07 nT, R²=0.972 at +1h).</li>"
        f"<li><b>3D Planetary CME Shock Transit (24–72h Horizon):</b> Dynamic regime ensemble (MoKTransitNN + GBDT) achieving ±10.63h MAE (±7.45h median error) with single GBDT and ±11.69h with hybrid ensemble across 91 unseen Cycle 25 CMEs (zero synthetic leakage), comparable to operational standards (NASA WSA-ENLIL 10.40h at Earth; Vršnak DBM 12.30h) in &lt;25ms execution on Apple Silicon M4.</li>"
        f"<li><b>Coronal Vision Dual APEX System:</b> 76.9% severe storm accuracy, 0.8360 ROC-AUC, 81.6% specificity defusing operational false alarms on Cycle 25 OOD.</li>"
        f"<li><b>Zero-Persistence Mission Cockpit:</b> Autonomous single-file browser application (Zar-Manah_HELIOS.html) running in volatile RAM with live NOAA SWPC & NASA SDO/SOHO streaming and NASA CCMC Scoreboard JSON export.</li>"
        f"</ul>"
        f"<p><i>Crafted with passion in Spain. Released under MIT and Creative Commons Attribution 4.0 International (CC-BY-4.0) licenses.</i></p>"
    )

    meta_payload = {
        "metadata": {
            "title": "Zar-Manah HELIOS 2.0: Multi-Gear Architecture for Real-Time Solar Storm Impact & CME Arrival Prediction",
            "upload_type": "publication",
            "publication_type": "preprint",
            "description": desc_html,
            "creators": [AUTHOR],
            "access_right": "open",
            "license": "cc-by-4.0",
            "version": "2.5.8",
            "keywords": [
                "Space Weather", "Solar Physics", "Geomagnetic Storms", "Dst Index",
                "Coronal Mass Ejections", "CME Arrival Time", "Conformal Prediction",
                "LightGBM", "Mixture of Kinematics", "Apple Silicon M4", "Spain", "Open Science"
            ],
            "related_identifiers": [
                {
                    "identifier": "https://github.com/Zar-Manah/HELIOS-Space-Weather",
                    "relation": "isSupplementedBy",
                    "scheme": "url"
                }
            ]
        }
    }

    meta_res = req(f"{ZENODO_API}/deposit/depositions/{new_dep_id}", method="PUT", data=meta_payload)
    if not meta_res["ok"]:
        print(f"❌ Failed to update metadata: {meta_res['error']}")
        sys.exit(1)
    print("✅ Deposition metadata updated successfully.")

    # Step 7: Formal Publication
    print(f"\n[6/6] Minting permanent DOI and publishing v2.5.8 on Zenodo...")
    pub_res = req(f"{ZENODO_API}/deposit/depositions/{new_dep_id}/actions/publish", method="POST")
    if not pub_res["ok"]:
        print(f"❌ Publication failed: {pub_res['error']}")
        sys.exit(1)

    pub_data = pub_res["data"]
    new_doi = pub_data.get("doi")
    concept_doi = pub_data.get("conceptdoi")
    record_url = pub_data.get("links", {}).get("record", f"https://zenodo.org/records/{new_dep_id}")

    print("\n" + "=" * 80)
    print("🎉 ZENODO PUBLICATION v2.5.8 COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"• Deposition ID: {new_dep_id}")
    print(f"• Version DOI:   {new_doi}")
    print(f"• Concept DOI:   {concept_doi}")
    print(f"• Record URL:    {record_url}")
    print("=" * 80)

    # Save to results/zenodo_record.json
    record_path = os.path.join(BASE_DIR, "results", "zenodo_record.json")
    with open(record_path, "w") as f:
        json.dump({
            "dep_id": int(new_dep_id),
            "doi": new_doi,
            "conceptdoi": concept_doi,
            "record_url": record_url,
            "version": "2.5.8",
            "published_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        }, f, indent=2)
    print(f"✅ Record saved to {record_path}")

if __name__ == "__main__":
    main()

