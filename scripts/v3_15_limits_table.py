#!/usr/bin/env python3
"""Four-cell limits table: {served, bench} x {full n=126, frozen 1:1 n=91}.

Read-only.  Loads the two frozen transit artefacts and the two frozen cohorts,
never retrains and never rewrites a model.  Writes:

  results/v3_limits_table.json
  paper/limits_study/RESULTS_TABLE.md

Why this exists
---------------
G2.4 certifies the served model on the full cohort (n=126) and G2.5 certifies
the bench model on the frozen 1:1 cohort (n=91).  Reporting one cell per gate
invites cherry-picking, so all four cells are published together, including
the one that had never been measured: the bench model on the full cohort.

Both models are frozen artefacts; this script only evaluates them.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from v3_common import (PROC, RESULTS, ROOT, add_transit_features,
                       design_matrix, load_cme_features)

MODELS = ROOT / "models" / "v3"
CLIP_LOG = (1.5, 5.1)


def full_cohort() -> pd.DataFrame:
    """Exactly the cohort of v3_06_train_transit.py without --bench."""
    df = load_cme_features().drop_duplicates("cme_id")
    df = df[(df["label"] == 1) & df["transit_hours"].between(12, 150)].copy()
    df = df[df["transit_hours"].notna() & df["v_lin"].notna()]
    df = add_transit_features(df.sort_values("t_launch").reset_index(drop=True))
    return df[df["t_launch"] >= pd.Timestamp("2020-01-01")].reset_index(drop=True)


def bench_cohort() -> pd.DataFrame:
    """Exactly the frozen 1:1 association cohort used by G2.5 (n=91)."""
    bench = pd.read_parquet(PROC / "v3_transit_bench.parquet").drop_duplicates("cme_id")
    bench["t_launch"] = pd.to_datetime(bench["t_launch"])
    ref = bench.set_index("cme_id")["transit_hours"]
    df = load_cme_features().drop_duplicates("cme_id")
    df = df[df["cme_id"].isin(set(ref.index))].copy()
    df["transit_hours"] = df["cme_id"].map(ref)
    df = df[df["transit_hours"].notna() & df["v_lin"].notna()]
    df = add_transit_features(df.sort_values("t_launch").reset_index(drop=True))
    return df[df["t_launch"] >= pd.Timestamp("2020-01-01")].reset_index(drop=True)


def member_pred(m, X, is_log: bool) -> np.ndarray:
    if isinstance(m, list):
        if is_log:
            return np.mean([np.exp(np.clip(x.predict(X), *CLIP_LOG)) for x in m], axis=0)
        return np.mean([x.predict(X) for x in m], axis=0)
    p = m.predict(X)
    return np.exp(np.clip(p, *CLIP_LOG)) if is_log else p


def predict(art: dict, df: pd.DataFrame) -> pd.DataFrame:
    feats = art["features"]
    X = design_matrix(df, feats)
    is_log = art.get("target_space") == "log"
    out = np.zeros(len(X))
    for m, w in zip(art["models"], art["model_weights"]):
        out = out + w * member_pred(m, X, is_log)

    d = df.copy()
    d["pred"] = out
    edges = tuple(art["regime_edges_v_lin"])
    reg = np.digitize(d["v_lin"].values, edges)
    mode = art["band_mode"]

    if mode == "abs_conformal_normalized":
        sg = np.array([art["regime_sigma"][str(r)] for r in reg])
        qn = np.array([art["q_norm"][str(r)] for r in reg])
        d["lo"] = d["pred"].values - qn * sg
        d["hi"] = d["pred"].values + qn * sg
    elif mode == "cqr":
        qr = np.array([art["q"][str(r)] for r in reg])
        d["lo"] = art["lo_model"].predict(X) - qr
        d["hi"] = art["hi_model"].predict(X) + qr
    else:  # abs_conformal
        qa = np.array([art["q_abs"][str(r)] for r in reg])
        d["lo"] = d["pred"].values - qa
        d["hi"] = d["pred"].values + qa

    d["covered"] = ((d["transit_hours"] >= d["lo"]) & (d["transit_hours"] <= d["hi"])).astype(int)
    return d


def metrics(d: pd.DataFrame) -> dict:
    e = d["pred"].values - d["transit_hours"].values
    out = {
        "n": int(len(d)),
        "mae_h": float(np.mean(np.abs(e))),
        "rmse_h": float(np.sqrt(np.mean(e ** 2))),
        "median_ae_h": float(np.median(np.abs(e))),
        "bias_h": float(np.mean(e)),
        "coverage90": float(d["covered"].mean()),
        "mean_width_h": float((d["hi"] - d["lo"]).mean()),
    }
    fast = d["v_lin"].values >= 1000
    if fast.sum() >= 3:
        out["fast_n"] = int(fast.sum())
        out["fast_mae_h"] = float(np.mean(np.abs(e[fast])))
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 4000, seed: int = 0) -> dict:
    """Paired bootstrap of MAE(a) - MAE(b) on the same events."""
    rng = np.random.default_rng(seed)
    n = len(a)
    d = np.abs(a) - np.abs(b)  # signed per-event MAE difference
    obs = float(np.mean(d))
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"delta_mae_h": obs, "ci95_h": [float(lo), float(hi)],
            "p_two_sided": float(2.0 * min((boot <= 0).mean(), (boot >= 0).mean())),
            "n": n, "n_boot": n_boot}


def main() -> int:
    served = joblib.load(MODELS / "transit_v3.joblib")
    bench = joblib.load(MODELS / "transit_v3_bench.joblib")

    cohorts = {"full_n126": full_cohort(), "frozen_1to1_n91": bench_cohort()}
    models = {"served_transit_v3": served, "bench_transit_v3_bench": bench}

    table: dict[str, dict] = {}
    errs: dict[str, np.ndarray] = {}
    for cname, cdf in cohorts.items():
        for mname, art in models.items():
            key = f"{cname}|{mname}"
            d = predict(art, cdf)
            table[key] = metrics(d)
            errs[key] = (d["pred"].values - d["transit_hours"].values)
            print(f"[limits] {key:44s} n={table[key]['n']:3d} "
                  f"MAE={table[key]['mae_h']:.4f} h", flush=True)

    contrasts = {
        c: paired_bootstrap(errs[f"{c}|served_transit_v3"],
                            errs[f"{c}|bench_transit_v3_bench"])
        for c in cohorts
    }

    payload = {
        "note": ("All four cells are reported. Neither model is retrained; "
                 "the cohort rules are those of v3_06_train_transit.py."),
        "served_artifact": "models/v3/transit_v3.joblib",
        "bench_artifact": "models/v3/transit_v3_bench.joblib",
        "bench_is_served": False,
        "cohorts": {k: int(len(v)) for k, v in cohorts.items()},
        "cells": table,
        "model_contrast_served_minus_bench": contrasts,
    }
    (RESULTS / "v3_limits_table.json").write_text(json.dumps(payload, indent=2) + "\n")

    order = [("full_n126", "served_transit_v3"), ("full_n126", "bench_transit_v3_bench"),
             ("frozen_1to1_n91", "served_transit_v3"), ("frozen_1to1_n91", "bench_transit_v3_bench")]
    lines = [
        "# Transit limits table (four cells)",
        "",
        "| Cohort | Model | n | MAE (h) | Median AE (h) | RMSE (h) | Bias (h) | Coverage@90 | Mean width (h) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c, m in order:
        r = table[f"{c}|{m}"]
        lines.append(
            f"| {c} | {m} | {r['n']} | {r['mae_h']:.4f} | {r['median_ae_h']:.4f} | "
            f"{r['rmse_h']:.4f} | {r['bias_h']:.4f} | {r['coverage90']:.4f} | {r['mean_width_h']:.4f} |")
    lines += [
        "",
        "`bench_transit_v3_bench` is **not served**. It is the artefact that G2.5 certifies.",
        "`served_transit_v3` is the production model (`models/v3/transit_v3.joblib`).",
        "",
        "## Paired model contrast (served − bench), 4,000 bootstrap resamples",
        "",
        "| Cohort | ΔMAE (h) | 95 % CI | p (two-sided) |",
        "|---|---:|---|---:|",
    ]
    for c in ("full_n126", "frozen_1to1_n91"):
        k = contrasts[c]
        lines.append(
            f"| {c} | {k['delta_mae_h']:+.4f} | "
            f"[{k['ci95_h'][0]:+.4f}, {k['ci95_h'][1]:+.4f}] | {k['p_two_sided']:.4f} |")
    lines += [""]
    md = "\n".join(lines)
    paper_copy = ROOT / "paper" / "limits_study" / "RESULTS_TABLE.md"
    paper_copy.parent.mkdir(parents=True, exist_ok=True)
    paper_copy.write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
