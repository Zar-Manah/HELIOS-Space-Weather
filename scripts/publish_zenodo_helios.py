#!/usr/bin/env python3
"""
Zar-Manah HELIOS: Official Zenodo Publisher (CERN / OpenAIRE)
Author: Zar Manah (ORCID: 0009-0007-8443-8999)
"""

import os
import sys
import json
import urllib.request
import urllib.parse
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZENODO_API_BASE = "https://zenodo.org/api"
ZENODO_TOKEN = "" + os.environ.get("ZENODO_ACCESS_TOKEN", "") + ""

AUTHOR = {
    "name": "Zar Manah",
    "affiliation": "Independent Researcher, Madrid, Spain",
    "orcid": "0009-0007-8443-8999"
}

def http_request(url, method="GET", data=None, headers=None):
    hdrs = {
        "Authorization": f"Bearer {ZENODO_TOKEN}",
        "User-Agent": "Zar-Manah-HELIOS/2.5"
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
        elif isinstance(data, str):
            body = data.encode("utf-8")

    req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            content = resp.read()
            try:
                res_json = json.loads(content.decode("utf-8"))
            except Exception:
                res_json = {"raw": content}
            return {"ok": True, "status": resp.status, "data": res_json}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": e.code, "error": err_body}
    except Exception as e:
        return {"ok": False, "status": 0, "error": str(e)}

def main():
    print("=" * 78)
    print("🚀 ZAR-MANAH HELIOS: ZENODO (CERN) OFFICIAL SCIENTIFIC DEPOSIT")
    print("=" * 78)
    
    # 1. Read abstract from manuscript.md
    manuscript_path = os.path.join(BASE_DIR, "paper", "manuscript.md")
    with open(manuscript_path, "r") as f:
        m_text = f.read()
        
    abstract_section = ""
    if "## Abstract" in m_text:
        parts = m_text.split("## Abstract")[1].split("## 1. Introduction")[0].strip()
        abstract_section = parts.replace("\n\n", "<br><br>").replace("**", "<b>").replace("**", "</b>")
    
    desc_html = (
        f"<p><b>Official Scientific Preprint & Open Architecture: Zar-Manah HELIOS v2.5</b></p>"
        f"<p>{abstract_section}</p>"
        f"<p><b>Primary Highlights:</b></p>"
        f"<ul>"
        f"<li><b>Tactical Nowcasting (1–6h):</b> GBDT + Bi-LSTM coupled to Inductive Conformalized Quantile Regression (CQR) with 89.2% empirical finite-sample coverage (MAE 2.90 nT at +1h).</li>"
        f"<li><b>Planetary CME Transit (24–72h):</b> Empirical Kinematic Regression (GBDT & MoK hybrid) achieving 10.63h MAE and 7.45h median on unseen Solar Cycle 25 (91 clean CMEs), on par with NASA WSA-ENLIL (10.40h at Earth) in sub-second execution (<0.05s on consumer hardware).</li>"
        f"<li><b>Real-Event Validation:</b> Evaluated on historical superstorms (May 2024, Halloween 2003, Nov 2003) and Solar Cycle 25 out-of-distribution eruptions with zero synthetic leakage.</li>"
        f"<li><b>Universal Cockpit:</b> Autonomous cross-platform HTML5/Canvas cockpit with real-time NOAA SWPC L1 telemetry and NASA CCMC Scoreboard JSON export.</li>"
        f"</ul>"
        f"<p><i>Crafted with passion in Spain. Released under MIT and CC-BY-4.0 licenses.</i></p>"
    )

    # 2. Create deposition draft
    print("1. Creating deposition draft on Zenodo...")
    res = http_request(f"{ZENODO_API_BASE}/deposit/depositions", method="POST", data={})
    if not res["ok"]:
        print("❌ Error creating deposition:", res["error"])
        sys.exit(1)
        
    dep = res["data"]
    dep_id = dep["id"]
    bucket_url = dep["links"]["bucket"]
    print(f"✅ Deposition draft created: ID = {dep_id}")
    print(f"   Bucket URL: {bucket_url}")

    # 3. Upload metadata
    print("2. Assigning formal scientific metadata...")
    meta_payload = {
        "metadata": {
            "title": "Zar-Manah HELIOS: Dual-Scale Space Weather Intelligence Coupling Tactical Conformal Nowcasting (1–6h) and Planetary CME Shock Transit (24–72h) across Solar Cycles 23–25",
            "upload_type": "publication",
            "publication_type": "preprint",
            "description": desc_html,
            "creators": [AUTHOR],
            "access_right": "open",
            "license": "cc-by-4.0",
            "keywords": [
                "Space Weather", "Geomagnetic Storms", "Dst Index",
                "Coronal Mass Ejections", "CME Transit Time", "Conformal Prediction",
                "Mixture-of-Kinematics", "Weight-Space Surgery", "Apple Silicon",
                "Spain", "NASA OMNI2", "NOAA SWPC", "Open Science"
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
    
    m_res = http_request(f"{ZENODO_API_BASE}/deposit/depositions/{dep_id}", method="PUT", data=meta_payload)
    if not m_res["ok"]:
        print("❌ Error updating metadata:", m_res["error"])
        sys.exit(1)
    print("✅ Metadata uploaded successfully.")

    # 4. Upload key files
    files_to_upload = [
        os.path.join(BASE_DIR, "requirements.txt"),
        os.path.join(BASE_DIR, "LICENSE"),
        os.path.join(BASE_DIR, "paper", "HELIOS_2_Solar_Vision_Swiss_Watch.pdf"),
        os.path.join(BASE_DIR, "Zar-Manah_HELIOS.html"),
        os.path.join(BASE_DIR, "predict.py"),
        os.path.join(BASE_DIR, "scripts", "export_ccmc_scoreboard.py"),
        os.path.join(BASE_DIR, "README.md"),
        os.path.join(BASE_DIR, "results", "helios_evaluation_metrics.json"),
        os.path.join(BASE_DIR, "results", "figures", "fig1_multi_horizon_benchmarks.png"),
        os.path.join(BASE_DIR, "results", "figures", "fig2_may2024_superstorm_case_study.png"),
        os.path.join(BASE_DIR, "results", "figures", "fig3_feature_importance.png"),
        os.path.join(BASE_DIR, "results", "figures", "fig4_cme_transit_benchmark.png"),
        os.path.join(BASE_DIR, "helios_trained_models.tar.gz"),
        os.path.join(BASE_DIR, "helios_processed_datasets.tar.gz")
    ]

    print("3. Uploading scientific artifacts to Zenodo bucket...")
    for fpath in files_to_upload:
        if not os.path.exists(fpath):
            print(f"⚠️ Warning: {fpath} not found, skipping.")
            continue
        fname = os.path.basename(fpath)
        fsize_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"   • Uploading {fname} ({fsize_mb:.2f} MB)...", end="", flush=True)
        
        with open(fpath, "rb") as fp:
            fdata = fp.read()
            
        u_url = f"{bucket_url}/{urllib.parse.quote(fname)}"
        u_res = http_request(u_url, method="PUT", data=fdata, headers={"Content-Type": "application/octet-stream"})
        if u_res["ok"]:
            print(" ✅ OK")
        else:
            print(f" ❌ Failed: {u_res.get('error')}")

    # 5. Publish deposition
    print("4. Formal Publication & DOI Minting on Zenodo...")
    pub_url = f"{ZENODO_API_BASE}/deposit/depositions/{dep_id}/actions/publish"
    pub_res = http_request(pub_url, method="POST")
    if not pub_res["ok"]:
        print("❌ Error publishing deposition:", pub_res["error"])
        sys.exit(1)
        
    pub_data = pub_res["data"]
    doi = pub_data.get("doi")
    record_url = pub_data.get("links", {}).get("record_html")
    
    print("=" * 78)
    print("🎉 PUBLICATION SUCCESSFUL ON ZENODO (CERN)!")
    print("=" * 78)
    print(f"• Assigned DOI:   {doi}")
    print(f"• Record URL:     {record_url}")
    print(f"• Deposition ID:  {dep_id}")
    print("=" * 78)
    
    # Save DOI info to local state
    zenodo_meta = {
        "dep_id": dep_id,
        "doi": doi,
        "record_url": record_url,
        "published_at": pub_data.get("created")
    }
    with open(os.path.join(BASE_DIR, "results", "zenodo_record.json"), "w") as f:
        json.dump(zenodo_meta, f, indent=2)
        
    return doi, record_url

if __name__ == "__main__":
    main()
