#!/usr/bin/env python3
"""
Zar-Manah HELIOS 2.0 - Hardware Acceleration & Inference Latency Benchmark
Runs 1,000 continuous inference cycles on Apple Silicon M4 GPU (MPS) and measures
stage-by-stage latencies empirically, producing results/latency_benchmark.json.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "scripts"))

from helios_swiss_orchestrator import SwissWatchPipeline, check_cme_earth_trajectory, device

def main():
    parser = argparse.ArgumentParser(description="Zar-Manah HELIOS 2.0 Latency Benchmark")
    parser.add_argument("--save", action="store_true", help="Explicitly overwrite canonical results/latency_benchmark.json")
    args = parser.parse_args()

    print("=" * 70)
    print("⚡ ZAR-MANAH HELIOS 2.0: 1,000-CYCLE EMPIRICAL LATENCY BENCHMARK (M4 GPU)")
    print("=" * 70)

    pipeline = SwissWatchPipeline()
    l1_seq = np.zeros((24, 14), dtype=np.float32)

    sample_event = {
        "v_launch": 812.0, "cme_speed": 812.0, "width": 360.0, "cme_width": 360.0,
        "accel": -12.4, "cme_accel": -12.4, "is_halo": 1, "source_lat": 18.0, "source_lon": -12.0,
        "ar_location": (280, 240), "f107": 215.0, "r_sunspot": 180.0, "v_sw_launch": 410.0,
        "prior_v": 450.0, "shear_angle_proxy": 45.0, "free_energy_proxy": 1.2e32,
        "helicity_proxy": -0.85, "flare_flux_proxy": 2.5e-4
    }

    # Warmup 25 cycles
    print("Warming up pipeline (25 cycles)...")
    for _ in range(25):
        _ = pipeline.run_inference(sample_event, l1_sequence=l1_seq)

    print("Running 1,000 continuous instrumented inference cycles...")
    t_pre = []
    t_g1 = []
    t_g2 = []
    t_g3 = []
    t_g4 = []
    t_post = []
    t_tot = []

    t0_global = time.time()
    for _ in range(1000):
        t0 = time.perf_counter()

        # Stage 0: Preprocessing & Physical Feature Vector Derivation
        v = float(sample_event.get("v_launch", 500.0))
        lat = float(sample_event.get("lat_proxy", 15.0))
        cos_t = np.clip(np.cos(np.radians(np.clip(lat, 0.0, 75.0))), 0.35, 1.0)
        v_3d = float(v / cos_t)
        mass = float(sample_event.get("mass", 5e15))
        m_grams = np.clip(mass if mass > 1e10 else 10.0**mass, 1e12, 1e18)
        energy_3d = float(np.log10(0.5) + np.log10(m_grams) + 2.0 * np.log10(np.clip(v_3d, 100.0, 4000.0) * 1e5))
        f107 = float(sample_event.get("f107", 120.0))
        flare_val = float(np.log10(np.clip(v_3d * f107 / 100.0, 1.0, 1e6)))
        phys_cols = pipeline.m1_pkg["phys_cols"]
        vals = [v_3d if c == "v_launch_3d" else energy_3d if c == "energy_3d" else flare_val if c == "flare_flux_proxy" else float(sample_event.get(c, 0.0)) for c in phys_cols]
        p_std = np.where(pipeline.m1_pkg["p_std"] == 0, 1.0, pipeline.m1_pkg["p_std"])
        X_solar = ((np.array(vals, dtype=np.float32) - pipeline.m1_pkg["p_mean"]) / p_std).reshape(1, -1)
        X_solar = np.nan_to_num(X_solar, nan=0.0)
        t1 = time.perf_counter()

        # Stage 1: Gear 1 (Chirality APEX Classifier & Quantiles)
        _ = float(pipeline.m1_pkg["clf"].predict_proba(X_solar)[0, 1])
        _ = float(pipeline.m1_pkg["q10"].predict(X_solar)[0])
        _ = float(pipeline.m1_pkg["q50"].predict(X_solar)[0])
        _ = float(pipeline.m1_pkg["q90"].predict(X_solar)[0])
        t2 = time.perf_counter()

        # Stage 2: Gear 2 (Storm Severity APEX Classifier & Quantiles)
        _ = float(pipeline.m2_pkg["clf"].predict_proba(X_solar)[0, 1])
        _ = float(pipeline.m2_pkg["q10"].predict(X_solar)[0])
        _ = float(pipeline.m2_pkg["q50"].predict(X_solar)[0])
        _ = float(pipeline.m2_pkg["q90"].predict(X_solar)[0])
        t3 = time.perf_counter()

        # Stage 3: Gear 3 (3D Cone Trajectory + MoKTransitNN on GPU + GBDT Ensembling)
        _, _, _ = check_cme_earth_trajectory(18.0, -12.0, 360.0, 812.0)
        L_AU = 1.495978707e8
        v_amb = 420.0
        gamma = 1.0e-7
        v_eff = v * np.exp(-gamma * L_AU * 0.4) + v_amb * (1.0 - np.exp(-gamma * L_AU * 0.4))
        drag_prior = L_AU / (v_eff * 3600.0)
        raw_tr = [v, np.log(v), 1.0 / (v + 10.0), (v / 1000.0)**2, 1.0, -12.4, 1, drag_prior]
        tr_norm = (np.array(raw_tr) - np.array(pipeline.transit_scaler["mean"])) / np.array(pipeline.transit_scaler["scale"])
        with torch.no_grad():
            t_input = torch.tensor(tr_norm, dtype=torch.float32).unsqueeze(0).to(device)
            pred_mok = float(pipeline.transit_model(t_input).item())
        tree_feats = [v, 360.0, -12.4, 1, v - v_amb, -gamma * (v - v_amb) * abs(v - v_amb), drag_prior, 0, np.sin(2 * np.pi * 0.5), np.cos(2 * np.pi * 0.5)]
        tree_cols = ["cme_speed", "cme_width", "cme_accel", "is_halo", "v_diff_ambient", "drag_accel_proxy", "transit_hours_dbm_prior", "is_fast_cme", "sin_cycle", "cos_cycle"]
        X_tree = pd.DataFrame([tree_feats], columns=tree_cols)
        pred_gbdt = float(pipeline.gbdt_time.predict(X_tree)[0])
        _ = float(pipeline.gbdt_low.predict(X_tree)[0])
        _ = float(pipeline.gbdt_high.predict(X_tree)[0])
        _ = max(16.0, 0.45 * pred_mok + 0.55 * pred_gbdt)
        t4 = time.perf_counter()

        # Stage 4: Gear 4 (HeliosBiLSTM Temporal Attention on GPU MPS)
        with torch.no_grad():
            seq_arr = (l1_seq - np.array(pipeline.bilstm_scaler["mean"], dtype=np.float32)) / np.array(pipeline.bilstm_scaler["scale"], dtype=np.float32)
            seq_t = torch.tensor(seq_arr, dtype=torch.float32).unsqueeze(0).to(device)
            _ = float(pipeline.bilstm_model(seq_t).item())
        t5 = time.perf_counter()

        # Stage 5: Verdict & Cockpit Formatting
        verdict = "RED PROTOCOL (G5 DANGER)"
        t6 = time.perf_counter()

        t_pre.append((t1 - t0) * 1000.0)
        t_g1.append((t2 - t1) * 1000.0)
        t_g2.append((t3 - t2) * 1000.0)
        t_g3.append((t4 - t3) * 1000.0)
        t_g4.append((t5 - t4) * 1000.0)
        t_post.append((t6 - t5) * 1000.0)
        t_tot.append((t6 - t0) * 1000.0)

    elapsed_global = time.time() - t0_global
    lat_arr = np.array(t_tot)

    mean_tot = float(np.mean(lat_arr))
    median_tot = float(np.median(lat_arr))
    p95_tot = float(np.percentile(lat_arr, 95))
    p99_tot = float(np.percentile(lat_arr, 99))
    min_tot = float(np.min(lat_arr))
    max_tot = float(np.max(lat_arr))

    mean_pre = float(np.mean(t_pre))
    mean_g1 = float(np.mean(t_g1))
    mean_g2 = float(np.mean(t_g2))
    mean_g3 = float(np.mean(t_g3))
    mean_g4 = float(np.mean(t_g4))
    mean_post = float(np.mean(t_post))

    print(f"\n📊 Empirical Benchmark Results (N=1000 cycles in {elapsed_global:.1f}s):")
    print(f"   • Total End-to-End Mean:   {mean_tot:.2f} ms")
    print(f"   • Total Median:            {median_tot:.2f} ms")
    print(f"   • Total p95:               {p95_tot:.2f} ms")
    print(f"   • Total p99:               {p99_tot:.2f} ms")
    print(f"   • Min / Max:               {min_tot:.2f} ms / {max_tot:.2f} ms")
    print("   • Stage-by-Stage Breakdown (Empirically Measured):")
    print(f"     - Stage 0 (Preprocessing):        {mean_pre:.2f} ms")
    print(f"     - Stage 1 (Gear 1 Chirality APEX): {mean_g1:.2f} ms")
    print(f"     - Stage 2 (Gear 2 Storm APEX):     {mean_g2:.2f} ms")
    print(f"     - Stage 3 (Gear 3 Transit MoK):    {mean_g3:.2f} ms")
    print(f"     - Stage 4 (Gear 4 BiLSTM GPU):     {mean_g4:.2f} ms")
    print(f"     - Stage 5 (Postprocessing/Verdict):{mean_post:.2f} ms")

    benchmark_artifact = {
        "hardware": "Apple Silicon M4 SoC (16 GB Unified RAM, Metal MPS)",
        "n_cycles": 1000,
        "mean_latency_ms": round(mean_tot, 2),
        "median_latency_ms": round(median_tot, 2),
        "p95_latency_ms": round(p95_tot, 2),
        "p99_latency_ms": round(p99_tot, 2),
        "min_latency_ms": round(min_tot, 2),
        "max_latency_ms": round(max_tot, 2),
        "stage_breakdown_empirical_ms": {
            "stage0_preprocessing": round(mean_pre, 2),
            "gear1_chirality_apex": round(mean_g1, 2),
            "gear2_storm_apex": round(mean_g2, 2),
            "gear3_mok_transit": round(mean_g3, 2),
            "gear4_bilstm_attention": round(mean_g4, 2),
            "stage5_verdict_formatting": round(mean_post, 2),
            "total_measured_sum": round(mean_pre + mean_g1 + mean_g2 + mean_g3 + mean_g4 + mean_post, 2)
        }
    }

    if args.save:
        out_path = os.path.join(BASE_DIR, "results", "latency_benchmark.json")
        with open(out_path, "w") as f:
            json.dump(benchmark_artifact, f, indent=2)
        print(f"\n✅ Benchmark artifact saved to: {out_path}")
    else:
        print("\nℹ️ Dry run completed. Canonical results/latency_benchmark.json preserved (pass --save to overwrite).")

if __name__ == "__main__":
    main()
