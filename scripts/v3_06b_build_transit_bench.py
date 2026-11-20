#!/usr/bin/env python3
"""HELIOS v3.4 - Stage 3.0: clean 1:1 transit benchmark set.

Rebuilds, from the same raw CDAW + Richardson & Cane inputs and with the same
explicit rule as the frozen v2.5.19 transit benchmark (331 events, GBDT MAE
10.63 h), the comparison set used for the 10.63 h / 10.4 h claims.  The rebuild
yields 327 events: the rule is identical, but the two sets are not row-for-row
identical, so the comparison is like-for-like by rule and cohort definition,
not a byte-identical cohort.

Rule (identical to v2.5.19 scripts/06_crossmatch_cme_dataset.py):
  * R&C disturbance with a candidate-CME list; launch time = FIRST timestamp
    listed in the candidate cell (column 18).
  * 15 h <= transit <= 130 h.
  * CDAW CME inside +-6 h of that launch time; the FASTEST one is the driver.
  * No CME in the window -> event dropped (never synthesised).

Output
  data/processed/v3_transit_bench.parquet
  results/v3_transit_bench_audit.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_01_build_cme_catalog import CDAW_TXT, RC_HTM, parse_cdaw, parse_rc
from v3_common import PROC, RESULTS, load_cme_features


def main() -> int:
    cdaw = parse_cdaw(CDAW_TXT)
    rc = parse_rc(RC_HTM)
    print(f"[bench] cdaw={len(cdaw)} rc={len(rc)}")

    ev = rc[rc["t_lasco"].notna()].copy()
    ev["transit_hours"] = (ev["t_disturbance"] - ev["t_lasco"]).dt.total_seconds() / 3600.0
    ev = ev[ev["transit_hours"].between(15.0, 130.0)].copy()
    print(f"[bench] R&C events with candidate CME list in 15-130 h: {len(ev)}")

    t_cdaw = cdaw["t_launch"].values
    rows, dropped = [], 0
    for r in ev.itertuples():
        dt_h = (t_cdaw - np.datetime64(r.t_lasco)) / np.timedelta64(1, "h")
        win = np.where(np.abs(dt_h) <= 6.0)[0]
        if len(win) == 0:
            dropped += 1
            continue
        best = win[np.argmax(cdaw["v_lin"].values[win])]
        rows.append(dict(
            rc_id=r.rc_id,
            t_launch=cdaw["t_launch"].values[best],
            t_shock=r.t_disturbance,
            transit_hours=float(r.transit_hours),
            cme_id=cdaw["cme_id"].values[best],
            v_lin=float(cdaw["v_lin"].values[best]),
            width=float(cdaw["width"].values[best]) if np.isfinite(cdaw["width"].values[best]) else np.nan,
            match_dt_h=float(dt_h[best]),
            n_window=int(len(win)),
            mc_flag=r.mc_flag,
            v_icme_rc=r.v_icme,
            dst_rc=r.dst_rc,
        ))
    bench = pd.DataFrame(rows).sort_values("t_launch").reset_index(drop=True)
    # strict 1:1 — a single CDAW CME can appear in two R&C candidate lists; keep
    # the disturbance whose own candidate time is closest to the CME.
    n0 = len(bench)
    bench = (bench.assign(_a=bench["match_dt_h"].abs())
                  .sort_values("_a").drop_duplicates("cme_id", keep="first")
                  .drop(columns="_a").sort_values("t_launch").reset_index(drop=True))
    if len(bench) != n0:
        print(f"[bench] 1:1 dedupe dropped {n0 - len(bench)} double-associated CME(s)")
    bench["solar_cycle"] = np.select(
        [bench["t_launch"] < pd.Timestamp("2009-01-01"),
         bench["t_launch"] < pd.Timestamp("2020-01-01")],
        [23, 24], default=25)
    print(f"[bench] matched={len(bench)} dropped(no CME in +-6h)={dropped}")

    # attach the full v3 feature matrix (SHARP, sources, OMNI context, pre-conditioning)
    feats = load_cme_features()
    keep = [c for c in feats.columns if c not in ("t_launch", "transit_hours")]
    out = bench.merge(feats[keep], on="cme_id", how="left", suffixes=("", "_v3"))
    out.to_parquet(PROC / "v3_transit_bench.parquet", index=False)

    _c = out[["transit_hours", "v_lin"]].dropna()
    corr = float(np.corrcoef(_c["transit_hours"], np.log1p(_c["v_lin"]))[0, 1])
    audit = {
        "n": int(len(out)),
        "dropped": int(dropped),
        "by_cycle": out["solar_cycle"].value_counts().sort_index().to_dict(),
        "transit_median_h": float(out["transit_hours"].median()),
        "transit_min_h": float(out["transit_hours"].min()),
        "transit_max_h": float(out["transit_hours"].max()),
        "corr_transit_logspeed": corr,
        "n_train_lt_2020": int((out["t_launch"] < pd.Timestamp("2020-01-01")).sum()),
        "n_test_cycle25": int((out["t_launch"] >= pd.Timestamp("2020-01-01")).sum()),
        "has_source": int((out.get("has_source", pd.Series(dtype=float)) == 1).sum()),
        "has_sharp": int(out.get("sharp_usflux", pd.Series(dtype=float)).notna().sum()),
        "legacy_n": 331,
    }
    (RESULTS / "v3_transit_bench_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))

    # Optional cross-check against the v2.5.19 association table.  That table is
    # part of the superseded release and is not shipped, so the check runs only
    # when the local archive is present.
    legacy = PROC / "helios_cme_transit_matched.parquet"
    if not legacy.exists():
        print(f"[bench] legacy comparison skipped: {legacy.name} is not in this package")
        print(f"[OK] {PROC / 'v3_transit_bench.parquet'} shape={out.shape}")
        return 0
    lg = pd.read_parquet(legacy)
    lg["t_launch"] = pd.to_datetime(lg["t_launch"])
    j = out[["t_launch", "transit_hours"]].merge(
        lg[["t_launch", "transit_hours"]], on="t_launch", how="inner",
        suffixes=("", "_legacy"))
    print(f"[bench] exact t_launch overlap with legacy 331: {len(j)}/{len(lg)}")
    if len(j):
        print(f"[bench] transit median diff vs legacy: "
              f"{(j['transit_hours'] - j['transit_hours_legacy']).abs().median():.4f} h")
    print(f"[OK] {PROC / 'v3_transit_bench.parquet'} shape={out.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
