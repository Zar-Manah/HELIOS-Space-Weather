#!/usr/bin/env python3
"""
HELIOS 2.0 - Historical End-to-End Backtest (real OMNI data, no synthetic input).

Runs the tactical (1/2/3/6 h) ring-current pipeline on real historical epochs:
seven geomagnetic storms and five quiet periods. For every epoch it compares the
forecast against the observed Dst and evaluates the severe-storm alert
(Dst <= -110 nT) at each lead time.

Usage:
    python3 scripts/21_backtest_historical.py [--save]

Outputs results/backtest_historical.json when --save is passed.
"""

import argparse
import json
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import predict as helios  # noqa: E402

DATA_FILE = os.path.join(BASE_DIR, "data", "processed", "helios_omni_features_1998_2026.parquet")
OUT_FILE = os.path.join(BASE_DIR, "results", "backtest_historical.json")
SEVERE_THRESHOLD = -110.0

EVENTS = [
    ("Halloween 2003", "2003-10-29 12:00", "storm"),
    ("November 2003", "2003-11-20 12:00", "storm"),
    ("November 2004", "2004-11-07 18:00", "storm"),
    ("St. Patrick 2015", "2015-03-17 12:00", "storm"),
    ("September 2017", "2017-09-08 00:00", "storm"),
    ("May 2024", "2024-05-10 18:00", "storm"),
    ("October 2024", "2024-10-10 18:00", "storm"),
    ("Solar minimum 2009", "2009-06-15 12:00", "quiet"),
    ("Quiet 2019", "2019-07-01 00:00", "quiet"),
    ("Quiet 2020", "2020-06-01 06:00", "quiet"),
    ("Quiet 2021", "2021-06-15 00:00", "quiet"),
    ("Quiet 2022", "2022-06-01 12:00", "quiet"),
]

LEADS = [1, 2, 3, 6]


def run_backtest():
    df = pd.read_parquet(DATA_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    models, metrics = helios.load_tactical_models()

    records = []
    for name, timestamp, kind in EVENTS:
        target = pd.Timestamp(timestamp)
        idx = (df["datetime"] - target).abs().idxmin()
        row = df.iloc[[idx]]
        forecast = helios.predict_tactical_storm(row, models, metrics)

        record = {
            "event": name,
            "kind": kind,
            "timestamp": str(row["datetime"].iloc[0])[:16],
            "dst_observed_at_t": round(float(row["dst"].iloc[0]), 1),
            "leads": {},
        }
        for lead in LEADS:
            predicted = float(forecast[lead]["pred"])
            observed = float(row[f"target_dst_lead_{lead}h"].iloc[0])
            record["leads"][f"{lead}h"] = {
                "predicted_dst_nT": round(predicted, 1),
                "observed_dst_nT": round(observed, 1),
                "abs_error_nT": round(abs(predicted - observed), 1),
                "severe_alert": predicted <= SEVERE_THRESHOLD,
                "severe_observed": observed <= SEVERE_THRESHOLD,
            }
        records.append(record)

    summary = {}
    for lead in LEADS:
        key = f"{lead}h"
        errors = [r["leads"][key]["abs_error_nT"] for r in records]
        hits = sum(1 for r in records if r["leads"][key]["severe_alert"] and r["leads"][key]["severe_observed"])
        false_alarms = sum(1 for r in records if r["leads"][key]["severe_alert"] and not r["leads"][key]["severe_observed"])
        misses = sum(1 for r in records if not r["leads"][key]["severe_alert"] and r["leads"][key]["severe_observed"])
        correct_quiet = sum(1 for r in records if not r["leads"][key]["severe_alert"] and not r["leads"][key]["severe_observed"])
        summary[key] = {
            "mae_nT": round(sum(errors) / len(errors), 1),
            "storm_hits": hits,
            "false_alarms": false_alarms,
            "misses": misses,
            "correct_quiet": correct_quiet,
        }

    return {
        "n_events": len(records),
        "n_storms": sum(1 for r in records if r["kind"] == "storm"),
        "n_quiet": sum(1 for r in records if r["kind"] == "quiet"),
        "severe_threshold_nT": SEVERE_THRESHOLD,
        "data_source": "OMNI hourly features (1998-2026), real observations",
        "events": records,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="HELIOS 2.0 historical backtest on real OMNI epochs")
    parser.add_argument("--save", action="store_true", help="Write results/backtest_historical.json")
    args = parser.parse_args()

    result = run_backtest()
    print(f"Events: {result['n_events']} ({result['n_storms']} storms, {result['n_quiet']} quiet)\n")
    header = f"{'Event':22s} {'Type':7s} {'t':17s} " + " ".join(f"{l}h pred/obs  " for l in LEADS)
    print(header)
    for r in result["events"]:
        cells = " ".join(
            f"{r['leads'][f'{l}h']['predicted_dst_nT']:7.1f}/{r['leads'][f'{l}h']['observed_dst_nT']:7.1f}" for l in LEADS
        )
        print(f"{r['event']:22s} {r['kind']:7s} {r['timestamp']:17s} {cells}")

    print("\nSummary (severe = Dst <= -110 nT):")
    for lead in LEADS:
        s = result["summary"][f"{lead}h"]
        print(
            f"  +{lead}h  MAE={s['mae_nT']:5.1f} nT | hits={s['storm_hits']} "
            f"false_alarms={s['false_alarms']} misses={s['misses']} correct_quiet={s['correct_quiet']}"
        )

    if args.save:
        with open(OUT_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()
