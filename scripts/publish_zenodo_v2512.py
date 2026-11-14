#!/usr/bin/env python3
"""
Zar-Manah HELIOS 2.0 - Zenodo Publisher v2.5.12
Publishes a new version under Concept DOI 10.5281/zenodo.24215019.

Credentials are NEVER stored in this file:
  - env ZENODO_ACCESS_TOKEN, or
  - ~/.config/zenodo/credentials.json  {"api_token": "..."}

Usage:
    python3 scripts/publish_zenodo_v2512.py --dry-run
    python3 scripts/publish_zenodo_v2512.py --publish
"""

import os
import sys
import json
import time
import hashlib
import argparse
import urllib.request
import urllib.error
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
API = "https://zenodo.org/api"
DEP_ID = 24224088                 # draft created from 24215020
CONCEPT_DOI = "10.5281/zenodo.24215019"
VERSION_DOI = "10.5281/zenodo.24224088"
PREVIOUS_DEP_ID = 24215020
VERSION = "2.5.12"
PUBLICATION_DATE = "2026-11-14"

FILES_TO_UPLOAD = [
    ("results/helios_processed_datasets.tar.gz", "helios_processed_datasets.tar.gz"),
    ("results/helios_trained_models.tar.gz", "helios_trained_models.tar.gz"),
    ("results/helios_evaluation_metrics.json", "helios_evaluation_metrics.json"),
    ("results/tactical_metrics_verified.json", "tactical_metrics_verified.json"),
    ("results/helios_transit_metrics.json", "helios_transit_metrics.json"),
    ("results/latency_benchmark.json", "latency_benchmark.json"),
    ("results/ablation_study.json", "ablation_study.json"),
    ("models/helios_models_checksums.json", "helios_models_checksums.json"),
    ("paper/HELIOS_Solar_Storm_Conformal_Forecasting.pdf", "HELIOS_Solar_Storm_Conformal_Forecasting.pdf"),
    ("paper/HELIOS_2_Solar_Vision_Swiss_Watch.pdf", "HELIOS_2_Solar_Vision_Swiss_Watch.pdf"),
    ("paper/manuscript.md", "manuscript.md"),
    ("predict.py", "predict.py"),
    ("tests/test_helios_core.py", "test_helios_core.py"),
    ("export_ccmc_scoreboard.py", "export_ccmc_scoreboard.py"),
    ("scripts/19_benchmark_latency.py", "19_benchmark_latency.py"),
    ("scripts/20_verify_tactical_metrics.py", "20_verify_tactical_metrics.py"),
    ("scripts/helios_swiss_orchestrator.py", "helios_swiss_orchestrator.py"),
    ("scripts/07_train_helios_transit.py", "07_train_helios_transit.py"),
    ("scripts/01_build_features.py", "01_build_features.py"),
    ("README.md", "README.md"),
    ("LICENSE", "LICENSE"),
    ("CITATION.cff", "CITATION.cff"),
    ("requirements.txt", "requirements.txt"),
    ("Zar-Manah_HELIOS.html", "Zar-Manah_HELIOS.html"),
    ("results/figures/fig1_multi_horizon_benchmarks.png", "fig1_multi_horizon_benchmarks.png"),
    ("results/figures/fig2_may2024_superstorm_case_study.png", "fig2_may2024_superstorm_case_study.png"),
    ("results/figures/fig3_feature_importance.png", "fig3_feature_importance.png"),
    ("results/figures/fig4_cme_transit_benchmark.png", "fig4_cme_transit_benchmark.png"),
    ("results/figures/fig1_helios_vision_benchmark.png", "fig1_helios_vision_benchmark.png"),
    ("results/figures/fig2_helios_apex_orchestrator.png", "fig2_helios_apex_orchestrator.png"),
    ("results/checksums.txt", "checksums.txt"),
]

CHANGELOG = (
    "<p><strong>Changes in v2.5.12 (supersedes v2.5.11):</strong></p><ul>"
    "<li>Published PDFs, metrics and code are now byte-consistent with the GitHub tag v2.5.12; "
    "the v2.5.11 deposit mixed 81.3% and 92.3% transit-coverage figures in different files.</li>"
    "<li>Model checksum manifest regenerated and verified: every sha256 in "
    "<code>helios_models_checksums.json</code> now validates its file.</li>"
    "<li>Ablation table clarified: the PHYS14 row is an ExtraTrees reference, not the deployed "
    "LightGBM APEX estimator; both score near chance under cross-validation.</li>"
    "<li>Tactical coverage/width are now reproducible from the published models and feature "
    "parquet via <code>scripts/20_verify_tactical_metrics.py</code>, which regenerates "
    "test-prediction parquets for all four horizons.</li>"
    "<li>Latency benchmark artifact refreshed with measurement protocol and run-to-run variance note.</li>"
    "<li><code>predict.py --live</code> now reports <code>data_source</code> and "
    "<code>fallback_reason</code> in JSON, so a synthetic fallback can never be mistaken for live data.</li>"
    "<li><code>--dashboard</code> opens the cockpit correctly on macOS; <code>--version</code> reports 2.5.12.</li>"
    "<li>Added <code>CITATION.cff</code>; removed a legacy publisher script that contained a "
    "credential in plain text (the token has been rotated).</li>"
    "</ul>"
)


def get_token():
    token = os.environ.get("ZENODO_ACCESS_TOKEN", "").strip()
    if token:
        return token
    cred = Path.home() / ".config" / "zenodo" / "credentials.json"
    if cred.exists():
        try:
            return json.loads(cred.read_text()).get("api_token", "").strip()
        except Exception:
            pass
    print("ERROR: no Zenodo token in ZENODO_ACCESS_TOKEN or ~/.config/zenodo/credentials.json")
    sys.exit(1)


def api(path, token, method="GET", payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        API + path, method=method, data=data,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code} on {method} {path}: {e.read().decode()[:800]}")
        raise


def upload_file(bucket_url, token, local_path, remote_name):
    body = Path(local_path).read_bytes()
    req = urllib.request.Request(
        f"{bucket_url}/{remote_name}", method="PUT", data=body,
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/octet-stream"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        if r.status not in (200, 201):
            raise RuntimeError(f"upload failed {remote_name}: HTTP {r.status}")
    return hashlib.sha256(body).hexdigest()


def build_metadata(token):
    prev = api(f"/records/{PREVIOUS_DEP_ID}", token)
    description = prev["metadata"]["description"]
    for a, b in [("17.59", "16.55"), ("17.32", "16.54"), ("21.27", "17.34"),
                 ("sub-22 ms", "sub-18 ms"), ("24215020", "24224088"), ("2.5.11", "2.5.12")]:
        description = description.replace(a, b)
    description = description.replace("</p><p><em>Lineage Note", CHANGELOG + "</p><p><em>Lineage Note", 1)
    if "Changes in v2.5.12" not in description:
        description += CHANGELOG

    return {
        "metadata": {
            "title": ("Zar-Manah HELIOS 2.0: Multi-Gear Architecture for Real-Time Solar Storm "
                      f"Impact & CME Arrival Prediction (v{VERSION})"),
            "upload_type": "publication",
            "publication_type": "preprint",
            "description": description,
            "creators": [{
                "name": "Manah, Zar",
                "affiliation": "Independent Researcher, Madrid, Spain",
                "orcid": "0009-0007-8443-8999",
            }],
            "keywords": ["Space Weather", "Solar Storm Forecasting", "Coronal Mass Ejection",
                         "Geomagnetic Disturbance", "Machine Learning", "Conformal Prediction",
                         "Apple Silicon M4"],
            "access_right": "open",
            "license": "cc-by-4.0",
            "version": VERSION,
            "publication_date": PUBLICATION_DATE,
            "related_identifiers": [
                {"identifier": "https://github.com/Zar-Manah/HELIOS-Space-Weather",
                 "relation": "isSupplementTo", "scheme": "url"},
                {"identifier": "10.5281/zenodo.24061451", "relation": "continues", "scheme": "doi"},
                {"identifier": "10.5281/zenodo.24215020", "relation": "isNewVersionOf", "scheme": "doi"},
            ],
        }
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()
    if not (args.dry_run or args.publish):
        ap.error("choose --dry-run or --publish")

    token = get_token()

    print("=" * 78)
    print(f"ZENODO v{VERSION} | draft {DEP_ID} | version DOI {VERSION_DOI}")
    print("=" * 78)

    missing = [src for src, _ in FILES_TO_UPLOAD if not (BASE_DIR / src).exists()]
    if missing:
        print("MISSING LOCAL FILES:")
        for m in missing:
            print("  " + m)
        sys.exit(1)
    print(f"Local files present: {len(FILES_TO_UPLOAD)}")

    # checksums.txt must cover every published file except itself
    published = {remote for _, remote in FILES_TO_UPLOAD} - {"checksums.txt"}
    listed = set()
    for line in (BASE_DIR / "results/checksums.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            listed.add(line.split(None, 1)[1].strip())
    if listed != published:
        print("checksums.txt mismatch")
        print("  missing in checksums:", sorted(published - listed))
        print("  extra in checksums:", sorted(listed - published))
        sys.exit(1)
    print(f"checksums.txt covers {len(listed)} files: OK")

    draft = api(f"/deposit/depositions/{DEP_ID}", token)
    if draft.get("submitted"):
        print("Draft already submitted; aborting.")
        sys.exit(1)
    print(f"Draft state={draft['state']} files={len(draft.get('files', []))}")

    if args.dry_run:
        meta = build_metadata(token)
        print("Metadata keys:", sorted(meta["metadata"].keys()))
        print("Title:", meta["metadata"]["title"])
        print("Description length:", len(meta["metadata"]["description"]))
        print("DRY RUN OK - nothing uploaded or published.")
        return

    for f in draft.get("files", []):
        api(f"/deposit/depositions/{DEP_ID}/files/{f['id']}", token, method="DELETE")
    print(f"Deleted {len(draft.get('files', []))} pre-existing draft files")

    bucket = draft["links"]["bucket"]
    for src, remote in FILES_TO_UPLOAD:
        digest = upload_file(bucket, token, BASE_DIR / src, remote)
        print(f"  uploaded {remote:<52} sha256={digest[:12]}...")

    api(f"/deposit/depositions/{DEP_ID}", token, method="PUT", payload=build_metadata(token))
    print("Metadata set")

    rec = api(f"/deposit/depositions/{DEP_ID}/actions/publish", token, method="POST")
    print(f"PUBLISHED id={rec['id']} doi={rec.get('doi')} state={rec['state']}")

    time.sleep(5)
    pub = api(f"/records/{rec['id']}", token)
    print(f"Verified: version={pub['metadata'].get('version')} files={len(pub.get('files', []))}")
    print(f"DOI: {pub.get('doi')} | Concept: {pub['metadata'].get('conceptdoi')}")


if __name__ == "__main__":
    main()
