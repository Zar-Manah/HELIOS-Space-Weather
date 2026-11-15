#!/usr/bin/env python3
"""
23_random_audit.py — NEW randomized end-to-end audit (v2.5.19).

Nothing here reuses the 5 landmark battery events or the fixed benchmarks.
Every test draws a fresh random sample (seed logged) from REAL observations.

Sections run in separate processes because mixing PyTorch and scikit-learn
in one interpreter triggers the known OpenMP conflict (SIGSEGV):

    python3 scripts/23_random_audit.py ACD    # orchestrator: A, C, D
    python3 scripts/23_random_audit.py B1     # GBDT predictions (sklearn)
    python3 scripts/23_random_audit.py B2     # Bi-LSTM predictions (torch)
    python3 scripts/23_random_audit.py merge  # -> results/random_audit.json

  A. 20 random real CME events (battery events excluded)
  B. 40 random real OMNI Cycle-25 timestamps
  C. 5 fastest + 5 slowest real CMEs -> physical sanity
  D. 500 random draws inside the real feature envelope -> robustness
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "scripts"))

RESULTS = os.path.join(BASE_DIR, "results")
DATA = os.path.join(BASE_DIR, "data/processed")
SEED = int(os.environ.get("HELIOS_AUDIT_SEED", str(int(time.time()) % 100000)))
rng = np.random.default_rng(SEED)
BATTERY_IDX = {272, 97, 96, 4, 61}
SEC = sys.argv[1] if len(sys.argv) > 1 else "ACD"

SEQ = ["dst", "bz_gsm", "by_gsm", "b_mag", "v_flow", "np_dens", "temp", "pdyn",
       "ey_rectified", "newell_coupling", "akasofu_eps", "borovsky_coupling", "kp", "f107"]


def w(name, obj):
    with open(os.path.join(RESULTS, f"_audit_{name}.json"), "w") as f:
        json.dump(obj, f, indent=2)


# =========================================================== A / C / D
if SEC in ("ACD",):
    from helios_swiss_orchestrator import SwissWatchPipeline
    pipe = SwissWatchPipeline()
    cat = pd.read_parquet(os.path.join(DATA, "helios_vision_matched.parquet"))

    print("=" * 78)
    print(f"[A] 20 RANDOM real CME events (seed {SEED}, battery events excluded)")
    print("=" * 78)
    pool = [i for i in range(len(cat)) if i not in BATTERY_IDX]
    pick = sorted(rng.choice(pool, size=20, replace=False).tolist())
    rows = []
    for i in pick:
        r = cat.iloc[i].to_dict()
        res = pipe.run_inference(r)
        err = abs(res["transit_hours"] - float(r["transit_hours"]))
        sev_ok = int(bool(res["is_severe_storm"])) == int(bool(r["is_severe_storm"]))
        bz = "ABSTAIN" if res["is_bz_south"] is None else (
            "OK" if int(bool(res["is_bz_south"])) == int(bool(r["is_dangerous_southward"])) else "WRONG")
        rows.append({"idx": int(i), "cycle": r["solar_cycle"], "speed": round(float(r["cme_speed"]), 1),
                     "gt_transit": round(float(r["transit_hours"]), 2),
                     "pred_transit": round(res["transit_hours"], 2), "abs_err_h": round(err, 2),
                     "p_severe": round(float(res["storm_prob_severe"]), 4),
                     "gt_severe": bool(r["is_severe_storm"]), "sev_class_ok": sev_ok,
                     "p_south": round(float(res["bz_prob_south"]), 4),
                     "gt_dangerous_south": bool(r["is_dangerous_southward"]), "bz_verdict": bz,
                     "latency_ms": round(res["latency_ms"], 2)})
        print(f"  {r['solar_cycle']:9s} {str(r['t_launch_dt'])[:10]} v={r['cme_speed']:6.0f} "
              f"transit {res['transit_hours']:5.1f}/{r['transit_hours']:5.1f} (|d| {err:4.1f}h) "
              f"sev {'OK ' if sev_ok else 'BAD'} bz {bz}")
    e = np.array([x["abs_err_h"] for x in rows])
    sec_a = {"n": len(rows), "mae_h": round(float(e.mean()), 2), "median_h": round(float(np.median(e)), 2),
             "max_h": round(float(e.max()), 2),
             "sev_class_correct": sum(x["sev_class_ok"] for x in rows),
             "bz_ok": sum(x["bz_verdict"] == "OK" for x in rows),
             "bz_wrong": sum(x["bz_verdict"] == "WRONG" for x in rows),
             "bz_abstain": sum(x["bz_verdict"] == "ABSTAIN" for x in rows),
             "latency_ms_mean": round(float(np.mean([x["latency_ms"] for x in rows])), 2),
             "events": rows}
    w("A", sec_a)
    print(f"  -> MAE {sec_a['mae_h']} h | median {sec_a['median_h']} h | sev {sec_a['sev_class_correct']}/20 | "
          f"Bz ok {sec_a['bz_ok']} wrong {sec_a['bz_wrong']} abstain {sec_a['bz_abstain']}")

    print("=" * 78)
    print("[C] Extreme real CMEs: 5 fastest + 5 slowest -> physical sanity")
    print("=" * 78)
    srt = cat.sort_values("cme_speed")
    rows = []
    for i in list(srt.index[:5]) + list(srt.index[-5:]):
        r = cat.iloc[i].to_dict()
        res = pipe.run_inference(r)
        t = res["transit_hours"]
        lo, hi = res["transit_conformal_window"]
        rows.append({"idx": int(i), "speed": round(float(r["cme_speed"]), 1),
                     "pred_transit": round(t, 2), "gt_transit": round(float(r["transit_hours"]), 2),
                     "window": [round(lo, 2), round(hi, 2)],
                     "finite": bool(np.isfinite(t)), "clamped_ok": bool(t >= 16.0),
                     "window_ordered": bool(lo < hi)})
        print(f"  v={r['cme_speed']:6.0f} pred {t:6.1f}h gt {r['transit_hours']:6.1f}h "
              f"win [{lo:5.1f},{hi:5.1f}] finite={np.isfinite(t)} clamp={t>=16.0}")
    w("C", {"n": len(rows), "all_finite": all(x["finite"] for x in rows),
            "all_clamped": all(x["clamped_ok"] for x in rows),
            "all_windows_ordered": all(x["window_ordered"] for x in rows), "events": rows})
    print(f"  -> finite {sum(x['finite'] for x in rows)}/10 | clamp {sum(x['clamped_ok'] for x in rows)}/10 | "
          f"ventanas ordenadas {sum(x['window_ordered'] for x in rows)}/10")

    print("=" * 78)
    print("[D] 500 random draws inside the real feature envelope -> robustness")
    print("=" * 78)
    cols = ["v_launch", "width", "accel", "mass", "energy", "prior_v", "f107", "r_sunspot",
            "v_sw_launch", "shear_angle_proxy", "free_energy_proxy", "helicity_proxy",
            "flare_flux_proxy", "cme_speed", "cme_width", "cme_accel", "cme_mass",
            "cme_energy", "dst_min", "bz_min"]
    env = {c: (float(cat[c].quantile(0.01)), float(cat[c].quantile(0.99))) for c in cols}
    bad, lats = [], []
    for _ in range(500):
        row = {c: float(rng.uniform(*env[c])) for c in cols}
        row["is_halo"] = int(rng.integers(0, 2))
        row["solar_cycle"] = "Cycle 25"
        row["year"] = 2024
        res = pipe.run_inference(row)
        lats.append(res["latency_ms"])
        for k in ("transit_hours", "storm_prob_severe", "bz_prob_south"):
            if not np.isfinite(res[k]):
                bad.append([k, str(res[k])])
        if not (0.0 <= res["storm_prob_severe"] <= 1.0) or not (0.0 <= res["bz_prob_south"] <= 1.0):
            bad.append(["probability_out_of_range", res["storm_prob_severe"], res["bz_prob_south"]])
    w("D", {"n": 500, "non_finite_or_out_of_range": len(bad), "examples": bad[:5],
            "latency_ms_mean": round(float(np.mean(lats)), 2),
            "latency_ms_p99": round(float(np.percentile(lats, 99)), 2)})
    print(f"  -> 500/500 ejecutados | no finitos/fuera de rango: {len(bad)} | "
          f"latencia media {np.mean(lats):.2f} ms | p99 {np.percentile(lats, 99):.2f} ms")

# ================================================================ A2 (OOD-only)
if SEC in ("A2",):
    from helios_swiss_orchestrator import SwissWatchPipeline
    pipe = SwissWatchPipeline()
    cat = pd.read_parquet(os.path.join(DATA, "helios_vision_matched.parquet"))
    cyc = cat[cat["solar_cycle"] == "Cycle 25"]
    pool = [i for i in cyc.index if i not in BATTERY_IDX]
    pick = sorted(rng.choice(pool, size=30, replace=False).tolist())
    rows = []
    print(f"[A2] 30 RANDOM Cycle-25 (OOD) events, seed {SEED}")
    for i in pick:
        r = cat.loc[i].to_dict()
        res = pipe.run_inference(r)
        err = abs(res["transit_hours"] - float(r["transit_hours"]))
        sev_ok = int(bool(res["is_severe_storm"])) == int(bool(r["is_severe_storm"]))
        bz = "ABSTAIN" if res["is_bz_south"] is None else (
            "OK" if int(bool(res["is_bz_south"])) == int(bool(r["is_dangerous_southward"])) else "WRONG")
        rows.append({"idx": int(i), "speed": round(float(r["cme_speed"]), 1),
                     "gt_transit": round(float(r["transit_hours"]), 2),
                     "pred_transit": round(res["transit_hours"], 2), "abs_err_h": round(err, 2),
                     "gbdt_transit": round(float(res["transit_gbdt"]), 2),
                     "p_severe": round(float(res["storm_prob_severe"]), 4),
                     "gt_severe": bool(r["is_severe_storm"]), "sev_class_ok": sev_ok,
                     "p_south": round(float(res["bz_prob_south"]), 4),
                     "gt_dangerous_south": bool(r["is_dangerous_southward"]), "bz_verdict": bz})
        print(f"  {str(r['t_launch_dt'])[:10]} v={r['cme_speed']:6.0f} "
              f"ens {res['transit_hours']:5.1f} gbdt {res['transit_gbdt']:5.1f} "
              f"gt {r['transit_hours']:5.1f} (|d| {err:4.1f}h) sev {'OK ' if sev_ok else 'BAD'} bz {bz}")
    e = np.array([x["abs_err_h"] for x in rows])
    eg = np.array([abs(x["gbdt_transit"] - x["gt_transit"]) for x in rows])
    w("A2", {"n": len(rows), "ensemble_mae_h": round(float(e.mean()), 2),
             "ensemble_median_h": round(float(np.median(e)), 2),
             "gbdt_mae_h": round(float(eg.mean()), 2),
             "sev_class_correct": sum(x["sev_class_ok"] for x in rows),
             "bz_ok": sum(x["bz_verdict"] == "OK" for x in rows),
             "bz_wrong": sum(x["bz_verdict"] == "WRONG" for x in rows),
             "bz_abstain": sum(x["bz_verdict"] == "ABSTAIN" for x in rows), "events": rows})
    print(f"  -> ensemble MAE {e.mean():.2f} h | median {np.median(e):.2f} h | GBDT MAE {eg.mean():.2f} h | "
          f"sev {sum(x['sev_class_ok'] for x in rows)}/30 | Bz ok {sum(x['bz_verdict']=='OK' for x in rows)} "
          f"wrong {sum(x['bz_verdict']=='WRONG' for x in rows)} abstain {sum(x['bz_verdict']=='ABSTAIN' for x in rows)}")

# ================================================================ B1 (sklearn)
if SEC in ("B1",):
    import joblib
    omni = pd.read_parquet(os.path.join(DATA, "helios_omni_features_1998_2026.parquet"))
    te = omni[omni["year"] >= 2020].reset_index(drop=True)
    m1 = joblib.load(os.path.join(BASE_DIR, "models", "helios_point_lead_1h.joblib"))
    m3 = joblib.load(os.path.join(BASE_DIR, "models", "helios_point_lead_3h.joblib"))
    f1 = list(getattr(m1, "feature_names_in_", SEQ))
    f3 = list(getattr(m3, "feature_names_in_", SEQ))
    valid = [i for i in range(24, len(te)) if np.isfinite(te["target_dst_lead_3h"].iloc[i])]
    pick = sorted(rng.choice(valid, size=40, replace=False).tolist())
    rows = []
    for i in pick:
        p1 = float(m1.predict(te.iloc[[i]][f1])[0])
        p3 = float(m3.predict(te.iloc[[i]][f3])[0])
        rows.append({"row": int(i), "gbdt_1h": round(p1, 3), "gbdt_3h": round(p3, 3),
                     "obs_1h": round(float(te["target_dst_lead_1h"].iloc[i]), 2),
                     "obs_3h": round(float(te["target_dst_lead_3h"].iloc[i]), 2)})
    w("B1", {"rows": rows})
    print(f"[B1] GBDT sobre {len(rows)} filas aleatorias reales de Cycle 25 (seed {SEED})")

# ================================================================ B2 (torch)
if SEC in ("B2",):
    from sklearn.preprocessing import StandardScaler
    omni = pd.read_parquet(os.path.join(DATA, "helios_omni_features_1998_2026.parquet"))
    tr = omni[omni["year"] <= 2019]
    te = omni[omni["year"] >= 2020].reset_index(drop=True)
    scaler = StandardScaler().fit(tr[SEQ])
    Xs = scaler.transform(te[SEQ]).astype(np.float32)
    import torch
    from helios_swiss_orchestrator import HeliosBiLSTM
    net = HeliosBiLSTM(in_features=14, hidden_dim=64)
    net.load_state_dict(torch.load(os.path.join(BASE_DIR, "models", "helios_bilstm_edited.pt"),
                                   map_location="cpu", weights_only=True))
    net.eval()
    b1 = json.load(open(os.path.join(RESULTS, "_audit_B1.json")))["rows"]
    for r in b1:
        i = r["row"]
        # window Xs[i-24:i] ends at row i-1, so the Bi-LSTM target is that row
        seq = torch.tensor(Xs[i - 24:i][None], dtype=torch.float32)
        with torch.no_grad():
            r["lstm_1h"] = round(float(net(seq).item()), 3)
        r["obs_1h_at_window_end"] = round(float(te["target_dst_lead_1h"].iloc[i - 1]), 2)
        r["e_lstm_1h"] = round(abs(r["lstm_1h"] - r["obs_1h_at_window_end"]), 2)
        r["e_gbdt_1h"] = round(abs(r["gbdt_1h"] - r["obs_1h"]), 2)
        r["e_gbdt_3h"] = round(abs(r["gbdt_3h"] - r["obs_3h"]), 2)
    m = lambda k: round(float(np.mean([x[k] for x in b1])), 2)  # noqa: E731
    s = lambda k: round(float(np.sqrt(np.mean([x[k] ** 2 for x in b1]))), 2)  # noqa: E731
    w("B", {"n": len(b1), "lstm_1h_mae": m("e_lstm_1h"), "gbdt_1h_mae": m("e_gbdt_1h"),
            "gbdt_3h_mae": m("e_gbdt_3h"), "lstm_1h_rmse": s("e_lstm_1h"),
            "gbdt_1h_rmse": s("e_gbdt_1h"), "gbdt_3h_rmse": s("e_gbdt_3h"), "rows": b1})
    print(f"[B2] Bi-LSTM +1h MAE {m('e_lstm_1h')} nT | GBDT +1h MAE {m('e_gbdt_1h')} nT | "
          f"GBDT +3h MAE {m('e_gbdt_3h')} nT (n={len(b1)})")

# =================================================================== merge
if SEC == "merge":
    out = {"seed": SEED, "seed_a2": int(os.environ.get("HELIOS_AUDIT_SEED_A2", "777001")),
           "version": "2.5.19", "sections": {}}
    for name, key in [("A", "A_random_cme"), ("A2", "A2_random_cycle25"), ("B", "B_random_omni"),
                      ("C", "C_extremes"), ("D", "D_random_envelope")]:
        p = os.path.join(RESULTS, f"_audit_{name}.json")
        if os.path.exists(p):
            sec = json.load(open(p))
            sec["seed"] = out["seed_a2"] if name == "A2" else SEED
            out["sections"][key] = sec
    with open(os.path.join(RESULTS, "random_audit.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"✅ results/random_audit.json (seed {SEED}) -> {list(out['sections'])}")
