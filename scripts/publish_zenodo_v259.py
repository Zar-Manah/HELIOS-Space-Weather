#!/usr/bin/env python3
"""
Zar-Manah HELIOS v2.5.9: Official CERN Zenodo Publisher & Cryptographic Verifier.
Creates and publishes an official deposition under Zar Manah identity,
uploads all verified scientific artifacts, verifies byte-for-byte hashes against Zenodo's server,
and mints the permanent DOI.
"""

import os
import sys
import json
import time
import hashlib
import argparse
import urllib.request
import urllib.parse
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZENODO_API = "https://zenodo.org/api"
TOKEN = os.environ.get("ZENODO_ACCESS_TOKEN", "").strip()

AUTHOR = {
    "name": "Zar Manah",
    "affiliation": "Independent Researcher, Madrid, Spain",
    "orcid": "0009-0007-8443-8999"
}

def req(url, method="GET", data=None, headers=None, timeout=180):
    hdrs = {
        "Authorization": f"Bearer {TOKEN}",
        "User-Agent": "Zar-Manah-HELIOS/2.5.9"
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
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            status = resp.status
            content = resp.read()
            try:
                parsed = json.loads(content.decode("utf-8")) if content else {}
            except Exception:
                parsed = content.decode("utf-8")
            return {"ok": True, "status": status, "data": parsed}
    except urllib.error.HTTPError as e:
        err_content = e.read().decode("utf-8", errors="replace")
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

def build_metadata():
    manuscript_path = os.path.join(BASE_DIR, "paper", "manuscript.md")
    with open(manuscript_path, "r") as f:
        m_text = f.read()
    abstract_section = ""
    if "## Abstract" in m_text:
        parts = m_text.split("## Abstract")[1].split("## 1. Introduction")[0].strip()
        abstract_section = parts.replace("\n\n", "<br><br>").replace("**", "<b>").replace("**", "</b>")

    desc_html = (
        f"<p><b>Official Scientific Preprint & Open Architecture: Zar-Manah HELIOS 2.0 (v2.5.9)</b></p>"
        f"<p>{abstract_section}</p>"
        f"<p><b>Architectural & Benchmark Highlights:</b></p>"
        f"<ul>"
        f"<li><b>Tactical Nowcasting (1–6h Horizon):</b> Dual-stream GBDT + Bi-LSTM coupled to Inductive Conformalized Quantile Regression (CQR) guaranteeing 89.26% empirical coverage (MAE 2.36 nT, RMSE 3.66 nT, R²=0.968 at +1h) evaluated across all 57,998 hourly samples of Solar Cycle 25 (comprising 2,104 storm hours across 237 distinct storm days).</li>"
        f"<li><b>3D Planetary CME Shock Transit (24–72h Horizon):</b> Dynamic regime ensemble (MoKTransitNN + GBDT) achieving 10.63h MAE (7.45h median error) with single GBDT and 11.69h with hybrid ensemble across 91 unseen Cycle 25 CMEs (zero synthetic leakage), with 92.3% empirical conformal coverage on nominal 90% window (59.7h width), comparable to literature benchmarks (NASA WSA-ENLIL 10.40h on Cycle 24; Vršnak DBM 12.30h) in sub-second execution.</li>"
        f"<li><b>Coronal Vision Dual APEX System:</b> 78.02% severe storm accuracy, 0.8737 ROC-AUC, 78.95% specificity defusing operational false alarms on Cycle 25 OOD. Selective classification / abstention zone enforced when 0.40 &lt; P_south &lt; 0.60 to avoid false certainty.</li>"
        f"<li><b>Empirical Latency Benchmark:</b> Measured across 1,000 full inference cycles on Apple Silicon M4 with an end-to-end latency of 17.59 ms (median 17.32 ms, p99 21.27 ms; sub-22 ms latency).</li>"
        f"<li><b>Interactive Visualization Companion:</b> Autonomous single-file browser cockpit (Zar-Manah_HELIOS.html) providing interactive telemetry visualization with client-side kinematics and NASA CCMC Scoreboard JSON export.</li>"
        f"</ul>"
        f"<p><i>Crafted in Spain under identity Zar Manah (ORCID: 0009-0007-8443-8999). Released under dual MIT and Creative Commons Attribution 4.0 International (CC-BY-4.0) licenses.</i></p>"
    )

    return {
        "metadata": {
            "title": "Zar-Manah HELIOS 2.0: Multi-Gear Architecture for Real-Time Solar Storm Impact & CME Arrival Prediction",
            "upload_type": "publication",
            "publication_type": "preprint",
            "description": desc_html,
            "creators": [AUTHOR],
            "access_right": "open",
            "license": "cc-by-4.0",
            "version": "2.5.9",
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

def main():
    if not TOKEN:
        print("❌ Error: ZENODO_ACCESS_TOKEN environment variable is not set!")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Zenodo v2.5.9 Publisher")
    parser.add_argument("--draft-id", type=str, default="", help="Existing draft ID if already spawned")
    parser.add_argument("--publish", action="store_true", help="Formally mint DOI and publish on Zenodo")
    args = parser.parse_args()

    meta_payload = build_metadata()

    print("=" * 80)
    print("🌟 ZAR-MANAH HELIOS 2.0 — ZENODO v2.5.9 PUBLICATION & INTEGRITY PIPELINE")
    print("=" * 80)
    print(f"Base Directory:  {BASE_DIR}")
    print(f"Author / ORCID:  {AUTHOR['name']} ({AUTHOR['orcid']})")
    print(f"Target Version:  v2.5.9 (Strict Scientific Integrity & Zero False Claims)")
    print(f"Publish Action:  {'MINT PERMANENT DOI' if args.publish else 'PREPARE & VERIFY DRAFT ONLY'}")
    print("-" * 80)

    # Step 1: Locate or Spawn draft
    new_dep_id = args.draft_id
    if not new_dep_id:
        print(f"\n[1/6] Creating new deposition under official Zar Manah account...")
        res = req(f"{ZENODO_API}/deposit/depositions", method="POST", data=meta_payload)
        if not res["ok"]:
            print(f"❌ Failed to create deposition: {res['error']}")
            sys.exit(1)
        new_dep_id = str(res["data"]["id"])
        print(f"✅ Created deposition ID: {new_dep_id}")
        draft_data = res["data"]
    else:
        print(f"\n[1/6] Using existing draft deposition: ID = {new_dep_id}")
        draft_res = req(f"{ZENODO_API}/deposit/depositions/{new_dep_id}")
        if not draft_res["ok"]:
            print(f"❌ Failed to fetch draft details: {draft_res['error']}")
            sys.exit(1)
        draft_data = draft_res["data"]

    bucket_url = draft_data["links"]["bucket"]
    existing_files = draft_data.get("files", [])
    print(f"\n[2/6] Inspecting draft {new_dep_id} bucket...")
    print(f"   Bucket URL: {bucket_url}")
    print(f"   Existing files in draft: {len(existing_files)}")

    # Step 3: Delete any stale files in draft bucket
    for ef in existing_files:
        fid = ef["id"]
        fname = ef.get("filename", ef.get("key", fid))
        print(f"   🗑️ Deleting stale file: {fname} (ID {fid})...", end="", flush=True)
        del_res = req(f"{ZENODO_API}/deposit/depositions/{new_dep_id}/files/{fid}", method="DELETE")
        if del_res["ok"] or del_res["status"] in [200, 204]:
            print(" ✅ OK")
        else:
            print(f" ⚠️ Warning: {del_res.get('error')}")

    # Step 4: Define all 23 verified scientific artifacts to upload
    files_to_upload = [
        os.path.join(BASE_DIR, "requirements.txt"),
        os.path.join(BASE_DIR, "LICENSE"),
        os.path.join(BASE_DIR, "README.md"),
        os.path.join(BASE_DIR, "Zar-Manah_HELIOS.html"),
        os.path.join(BASE_DIR, "predict.py"),
        os.path.join(BASE_DIR, "paper", "HELIOS_Solar_Storm_Conformal_Forecasting.pdf"),
        os.path.join(BASE_DIR, "paper", "HELIOS_2_Solar_Vision_Swiss_Watch.pdf"),
        os.path.join(BASE_DIR, "paper", "manuscript.md"),
        os.path.join(BASE_DIR, "tests", "test_helios_core.py"),
        os.path.join(BASE_DIR, "scripts", "19_benchmark_latency.py"),
        os.path.join(BASE_DIR, "scripts", "export_ccmc_scoreboard.py"),
        os.path.join(BASE_DIR, "results", "helios_evaluation_metrics.json"),
        os.path.join(BASE_DIR, "results", "latency_benchmark.json"),
        os.path.join(BASE_DIR, "results", "ablation_study.json"),
        os.path.join(BASE_DIR, "results", "helios_transit_metrics.json"),
        os.path.join(BASE_DIR, "results", "checksums.txt"),
        os.path.join(BASE_DIR, "models", "helios_models_checksums.json"),
        os.path.join(BASE_DIR, "results", "figures", "fig1_multi_horizon_benchmarks.png"),
        os.path.join(BASE_DIR, "results", "figures", "fig2_may2024_superstorm_case_study.png"),
        os.path.join(BASE_DIR, "results", "figures", "fig3_feature_importance.png"),
        os.path.join(BASE_DIR, "results", "figures", "fig4_cme_transit_benchmark.png"),
        os.path.join(BASE_DIR, "results", "helios_trained_models.tar.gz"),
        os.path.join(BASE_DIR, "results", "helios_processed_datasets.tar.gz")
    ]

    print(f"\n[3/6] Uploading {len(files_to_upload)} verified scientific artifacts to Zenodo bucket...")
    local_hashes = {}
    for fpath in files_to_upload:
        if not os.path.exists(fpath):
            print(f"❌ Error: Required file missing: {fpath}")
            sys.exit(1)
        fname = os.path.basename(fpath)
        fsize_mb = os.path.getsize(fpath) / (1024 * 1024)
        md5_h, sha_h = compute_hashes(fpath)
        local_hashes[fname] = {"md5": md5_h, "sha256": sha_h, "size": os.path.getsize(fpath)}
        print(f"   • Uploading {fname:45s} ({fsize_mb:6.2f} MB)...", end="", flush=True)

        with open(fpath, "rb") as fp:
            data = fp.read()
        target_url = f"{bucket_url}/{urllib.parse.quote(fname)}"
        up_res = req(target_url, method="PUT", data=data, headers={"Content-Type": "application/octet-stream"})
        if up_res["ok"]:
            print(f" ✅ OK (MD5: {md5_h[:8]}...)")
        else:
            print(f" ❌ FAILED: {up_res.get('error')}")
            sys.exit(1)

    # Step 5: Verification of all uploaded files in bucket
    print(f"\n[4/6] Cryptographic bit-identity verification against Zenodo server...")
    verify_draft = req(f"{ZENODO_API}/deposit/depositions/{new_dep_id}")
    if not verify_draft["ok"]:
        print(f"❌ Failed to verify draft: {verify_draft['error']}")
        sys.exit(1)

    uploaded_files = verify_draft["data"].get("files", [])
    print(f"   Total files present in draft bucket: {len(uploaded_files)}/{len(files_to_upload)}")
    all_match = True
    server_hashes = {}
    for uf in uploaded_files:
        fn = uf.get("filename", uf.get("key"))
        server_size = int(uf.get("filesize", uf.get("size", 0)))
        server_checksum = uf.get("checksum", "")
        clean_server_md5 = server_checksum.replace("md5:", "").strip()
        server_hashes[fn] = {"size": server_size, "md5": clean_server_md5}
        if fn not in local_hashes:
            print(f"   ⚠️ Unexpected file in bucket: {fn}")
            continue
        loc = local_hashes[fn]
        size_match = (server_size == loc["size"])
        md5_match = (clean_server_md5 == loc["md5"])
        if not size_match or not md5_match:
            print(f"   ❌ Hash/Size MISMATCH on {fn}: server_size={server_size}, loc_size={loc['size']}, server_md5={clean_server_md5}, loc_md5={loc['md5']}")
            all_match = False
        else:
            print(f"   ✅ Bit-exact: {fn:45s} ({server_size} bytes, MD5 {clean_server_md5})")

    if not all_match or len(uploaded_files) != len(files_to_upload):
        print("❌ Integrity verification failed! Aborting.")
        sys.exit(1)
    print("   🏆 100% BYTE-FOR-BYTE BIT IDENTITY VERIFIED ON ZENODO BUCKET!")

    # Step 6: Update Metadata
    print(f"\n[5/6] Updating deposition metadata to v2.5.9...")
    meta_res = req(f"{ZENODO_API}/deposit/depositions/{new_dep_id}", method="PUT", data=meta_payload)
    if not meta_res["ok"]:
        print(f"❌ Failed to update metadata: {meta_res['error']}")
        sys.exit(1)
    print("✅ Deposition metadata updated successfully.")

    if not args.publish:
        print("\n" + "=" * 80)
        print(f"⏸️ DRAFT PREPARED & VERIFIED ON ZENODO: ID = {new_dep_id}")
        print(f"Run with --publish to formally mint permanent DOI.")
        print("=" * 80)
        return

    # Step 7: Formal Publication
    print(f"\n[6/6] Minting permanent DOI and publishing v2.5.9 on Zenodo...")
    pub_res = req(f"{ZENODO_API}/deposit/depositions/{new_dep_id}/actions/publish", method="POST")
    if not pub_res["ok"]:
        print(f"❌ Publication failed: {pub_res['error']}")
        sys.exit(1)

    pub_data = pub_res["data"]
    new_doi = pub_data.get("doi")
    concept_doi = pub_data.get("conceptdoi")
    record_url = pub_data.get("links", {}).get("record", f"https://zenodo.org/records/{new_dep_id}")

    print("\n" + "=" * 80)
    print("🎉 ZENODO PUBLICATION v2.5.9 COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"• Deposition ID: {new_dep_id}")
    print(f"• Version DOI:   {new_doi}")
    print(f"• Concept DOI:   {concept_doi}")
    print(f"• Record URL:    {record_url}")
    print("=" * 80)

    # Save to results/zenodo_record.json with REAL server-verified hashes
    record_path = os.path.join(BASE_DIR, "results", "zenodo_record.json")
    with open(record_path, "w") as f:
        json.dump({
            "dep_id": int(new_dep_id),
            "doi": new_doi,
            "conceptdoi": concept_doi,
            "record_url": record_url,
            "version": "2.5.9",
            "published_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "status": f"v2.5.9 published on Zenodo CERN OpenAIRE under permanent Concept DOI {concept_doi}",
            "server_verified_files": server_hashes,
            "local_sha256": {k: v["sha256"] for k, v in local_hashes.items()}
        }, f, indent=2)
    print(f"✅ Record saved to {record_path} with authentic server-verified checksums.")

if __name__ == "__main__":
    main()
