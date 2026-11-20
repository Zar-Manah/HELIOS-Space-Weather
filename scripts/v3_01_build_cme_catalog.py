#!/usr/bin/env python3
"""HELIOS v3.4 - Stage 1.1: clean CME catalog (CDAW) + Richardson & Cane ICMEs + real OMNI targets.

Fixes vs v2.5.19:
  * CDAW column 8 is v_20R, column 9 is Accel (legacy read v_20R as accel).
  * No manual target overrides: every dst_min / bz_min / b_max / v_max / pdyn_max is
    measured inside the real R&C ICME window from OMNI observations.
  * Explicit CME<->ICME match provenance and ambiguous (-1) labels.

Outputs
  data/processed/v3_cme_catalog.parquet
  results/v3_catalog_audit.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
PROC.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

CDAW_TXT = RAW / "cdaw_univ_all.txt"
RC_HTM = RAW / "icmetable2.htm"
OMNI_PARQUET = PROC / "helios_omni_features_1998_2026.parquet"


# ----------------------------------------------------------------------------
# CDAW parser
# ----------------------------------------------------------------------------
CDAW_RE = re.compile(
    r"^(?P<date>\d{4}/\d{2}/\d{2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"(?P<cpa>Halo|[\d.]+)\s+(?P<width>[\d.]+)\s+"
    r"(?P<v_lin>\d+|----)\s+(?P<v2i>\d+|----)\s+(?P<v2f>\d+|----)\s+"
    r"(?P<v20r>[\d.]+|----)\s+(?P<accel>[-+]?[\d.]+|------)\*?\s+"
    r"(?P<mass>[-\d.e+]+|-------)\*?\s+(?P<energy>[-\d.e+]+|-------)\*?\s+"
    r"(?P<mpa>\d+|----)\s*(?P<remarks>.*)$"
)


def _num(tok: str, star_ok: bool = True) -> float:
    tok = tok.strip().rstrip("*")
    if tok in {"", "-", "----", "------", "-------", "-----"}:
        return np.nan
    try:
        return float(tok)
    except ValueError:
        return np.nan


def parse_cdaw(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = CDAW_RE.match(line.strip())
        if not m:
            continue
        d = m.groupdict()
        dt = pd.to_datetime(f"{d['date']} {d['time']}", format="%Y/%m/%d %H:%M:%S", errors="coerce")
        if pd.isna(dt):
            continue
        remarks = d["remarks"].strip()
        low = remarks.lower()
        rows.append(
            dict(
                cme_id=f"CDAW-{d['date'].replace('/', '')}-{d['time'].replace(':', '')}",
                t_launch=dt,
                cpa=_num(d["cpa"]),
                width=_num(d["width"]),
                v_lin=_num(d["v_lin"]),
                v_2nd_init=_num(d["v2i"]),
                v_2nd_final=_num(d["v2f"]),
                v_20R=_num(d["v20r"]),
                accel_true=_num(d["accel"]),
                mass=_num(d["mass"]),
                energy=_num(d["energy"]),
                mpa=_num(d["mpa"]),
                remarks=remarks,
                flag_poor="poor event" in low,
                flag_very_poor="very poor" in low,
                flag_only_c2="only c2" in low,
                flag_only_c3="only c3" in low,
                flag_partial_halo="partial halo" in low,
                flag_halo=bool(re.search(r"\bhalo\b", low)) and "partial" not in low,
            )
        )
    df = pd.DataFrame(rows).sort_values("t_launch").reset_index(drop=True)
    # CDAW lists distinct CMEs at the same timestamp (different CPA / different
    # analysis).  Without a per-timestamp ordinal the ids collide and every
    # downstream cme_id merge silently duplicates rows.
    dup = df["cme_id"].duplicated(keep=False)
    if dup.any():
        ordn = df.groupby("cme_id").cumcount() + 1
        df.loc[dup, "cme_id"] = df.loc[dup, "cme_id"] + "-" + ordn[dup].astype(str)
    df["dv_2nd"] = df["v_2nd_final"] - df["v_2nd_init"]
    df["v_20r_ratio"] = np.where(
        (df["v_20R"] > 0) & (df["v_lin"] > 0), df["v_20R"] / df["v_lin"], np.nan
    )
    df["accel_missing"] = df["accel_true"].isna()
    df["is_halo"] = ((df["width"] >= 360) | df["flag_halo"]).astype(int)
    df["is_partial_halo"] = ((df["width"] >= 120) & (df["width"] < 360)).astype(int)
    df["is_wide"] = (df["width"] >= 120).astype(int)
    df["is_narrow"] = (df["width"] < 30).astype(int)
    df["log_speed"] = np.log1p(df["v_lin"])
    df["log_mass"] = np.log10(df["mass"].where(df["mass"] > 0))
    df["log_energy"] = np.log10(df["energy"].where(df["energy"] > 0))
    return df


# ----------------------------------------------------------------------------
# Richardson & Cane parser
# ----------------------------------------------------------------------------
RC_DATE = re.compile(r"^(\d{4})/(\d{2})/(\d{2})\s+(\d{2})(\d{2})")
RC_SHORT = re.compile(r"^(\d{2})(\d{2})\s*([A-Za-z]*)")


def _rc_dt(tok: str) -> pd.Timestamp | None:
    tok = tok.strip()
    m = RC_DATE.match(tok)
    if not m:
        return None
    y, mo, d, hh, mm = m.groups()
    ts = pd.Timestamp(f"{y}-{mo}-{d} {hh}:{mm}:00")
    return ts if 1990 <= ts.year <= 2030 else None


def parse_lasco_list(cell: str) -> list[tuple[pd.Timestamp, str]]:
    """Parse the 18th R&C cell: a comma-separated list of candidate CMEs.

    Examples
      '2024/05/07 0312 D,  0524 D, 2024/05/08 0536 HD, 1224 HD, 1912 D, ...'
      '1999/04/13 0330 H'
    Short entries ('0524 D') inherit year/month from the previous full date.
    """
    out: list[tuple[pd.Timestamp, str]] = []
    last: pd.Timestamp | None = None
    for part in re.split(r",", cell):
        part = part.strip()
        if not part:
            continue
        m = RC_DATE.match(part)
        if m:
            ts = _rc_dt(part)
            if ts is not None:
                last = ts
                tag = part[16:].strip()
                out.append((ts, tag))
            continue
        m = RC_SHORT.match(part)
        if m and last is not None:
            hh, mm, tag = m.groups()
            try:
                ts = last.replace(hour=int(hh), minute=int(mm))
            except ValueError:
                continue
            out.append((ts, tag.strip()))
    # dedupe keeping order
    seen, uniq = set(), []
    for ts, tag in out:
        if ts in seen:
            continue
        seen.add(ts)
        uniq.append((ts, tag))
    return uniq


AU_KM = 1.495978707e8


def dbm_transit_hours(v0: float, w_amb: float, gamma: float) -> float:
    """Exact 1-D drag-based-model transit time to 1 AU (v0 > w_amb)."""
    if not np.isfinite(v0) or v0 <= w_amb:
        return np.inf
    if gamma <= 0:
        return AU_KM / (v0 - w_amb) / 3600.0
    # w*T + (1/gamma)*ln(1 + gamma*(v0-w)*T) = D  -> solve for T by bisection
    lo, hi = 0.0, 400.0 * 3600.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        dist = w_amb * mid + np.log1p(gamma * (v0 - w_amb) * mid) / gamma
        if dist < AU_KM:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) / 3600.0


def dbm_speed_at_1au(v0: float, w_amb: float, gamma: float) -> float:
    t = dbm_transit_hours(v0, w_amb, gamma) * 3600.0
    if not np.isfinite(t):
        return np.nan
    return w_amb + 1.0 / (1.0 / (v0 - w_amb) + gamma * t)


def gamma_from_transit(v0: float, w_amb: float, transit_h: float) -> float:
    """Invert the drag parameter implied by an observed transit (bisection)."""
    if not np.isfinite(v0) or v0 <= w_amb or transit_h <= 0:
        return np.nan
    t = transit_h * 3600.0
    if v0 * t < AU_KM:  # even with zero drag the CME cannot reach 1 AU
        return np.nan
    lo, hi = 0.0, 3e-7
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if dbm_transit_hours(v0, w_amb, mid) * 3600.0 > t:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)



def _rc_num(tok: str) -> float:
    tok = tok.strip()
    m = re.search(r"[-+]?\d+(?:\.\d+)?", tok)
    return float(m.group(0)) if m else np.nan


def parse_rc(path: Path) -> pd.DataFrame:
    # BeautifulSoup, not a regex: 100 of the 644 data rows contain a <td> that
    # a naive regex pass merges with its neighbour, silently dropping the
    # candidate-CME list (column 18) and therefore the whole R&C event.
    from bs4 import BeautifulSoup

    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    table = soup.find("table")
    trs = table.find_all("tr") if table is not None else soup.find_all("tr")
    rows = []
    for tr in trs:
        cells = [td.get_text(strip=True).replace("\xa0", " ") for td in tr.find_all("td")]
        while cells and cells[-1] == "":
            cells.pop()
        if not cells or _rc_dt(cells[0]) is None:
            continue
        # 18 cells => last one is the candidate CME list; 17 => no list column.
        lasco_cell = ""
        if len(cells) >= 18:
            lasco_cell = cells[17]
            cells = cells[:17]
        if len(cells) < 16:
            continue
        lasco_list = parse_lasco_list(lasco_cell)
        rec = dict(
            rc_id=f"RC-{_rc_dt(cells[0]).strftime('%Y%m%d%H%M')}",
            t_disturbance=_rc_dt(cells[0]),
            t_icme_start=_rc_dt(cells[1]),
            t_icme_end=_rc_dt(cells[2]),
            comp_start=_rc_num(cells[3]),
            comp_end=_rc_num(cells[4]),
            mc_start_h=_rc_num(cells[5]),
            mc_end_h=_rc_num(cells[6]),
            bde=cells[7],
            bif=cells[8],
            quality=cells[9],
            dv=_rc_num(cells[10]),
            v_icme=_rc_num(cells[11]),
            v_max=_rc_num(cells[12]),
            b_mean=_rc_num(cells[13]),
            mc_flag_raw=cells[14],
            dst_rc=_rc_num(cells[15]),
            v_transit_rc=_rc_num(cells[16]) if len(cells) > 16 else np.nan,
            v_transit_raw=cells[16] if len(cells) > 16 else "",
            lasco_list=lasco_list,
            n_lasco_candidates=len(lasco_list),
            t_lasco=lasco_list[0][0] if lasco_list else pd.NaT,
        )
        rec["mc_flag"] = _rc_num(rec["mc_flag_raw"])
        rows.append(rec)

    df = pd.DataFrame(rows).sort_values("t_disturbance").reset_index(drop=True)
    # propagate the merged "ICME start/end" when the cell is a continuation marker
    df["t_icme_end"] = df["t_icme_end"].fillna(df["t_disturbance"] + pd.Timedelta(hours=24))
    df["icme_duration_h"] = (df["t_icme_end"] - df["t_icme_start"]).dt.total_seconds() / 3600.0
    return df


# ----------------------------------------------------------------------------
# OMNI targets
# ----------------------------------------------------------------------------
def load_omni() -> pd.DataFrame:
    df = pd.read_parquet(OMNI_PARQUET, columns=[
        "datetime", "dst", "bz_gsm", "b_mag", "v_flow", "np_dens", "pdyn", "e_field", "temp",
    ])
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.set_index("datetime").sort_index()


def measure_window(omni: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> dict:
    win = omni.loc[t0:t1]
    if len(win) < 3:
        return {}
    dst = win["dst"]
    bz = win["bz_gsm"]
    out = dict(
        dst_min_obs=dst.min(),
        dst_t0=float(omni["dst"].asof(t0)) if pd.notna(omni["dst"].asof(t0)) else np.nan,
        dst_mean_obs=dst.mean(),
        bz_min_obs=bz.min(),
        bz_min_6h_mean_obs=bz.rolling(6, min_periods=3).mean().min(),
        b_mag_max_obs=win["b_mag"].max(),
        v_max_obs=win["v_flow"].max(),
        pdyn_max_obs=win["pdyn"].max(),
        ey_max_obs=win["e_field"].max(),
        storm_duration_dst_le_50_h=float((dst <= -50).sum()),
        n_omni_hours=int(len(win)),
    )
    return out


# ----------------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------------
def match_catalog(cdaw: pd.DataFrame, rc: pd.DataFrame) -> pd.DataFrame:
    """Attach R&C events to CDAW CMEs with explicit provenance.

    rc_list  : the R&C table itself lists the candidate CME times for the event
               (column 18, e.g. '2024/05/08 0536 HD, 1224 HD, ...').  The fastest
               listed halo is selected as the driver (standard multi-CME logic).
    inferred : no list -> best physically plausible CME in the preceding 120 h,
               validated with the exact DBM drag inversion and a projection
               allowance of at most 2.5x the plane-of-sky speed.
    """
    cdaw = cdaw.copy()
    cdaw["rc_id"] = None
    cdaw["match_method"] = None
    cdaw["match_confidence"] = None
    cdaw["match_score"] = np.nan
    cdaw["t_shock"] = pd.NaT
    cdaw["t_icme_start"] = pd.NaT
    cdaw["t_icme_end"] = pd.NaT
    cdaw["mc_flag"] = np.nan
    cdaw["dst_rc"] = np.nan
    cdaw["v_icme_rc"] = np.nan

    used: set[int] = set()
    launch_times = cdaw["t_launch"].values
    speeds = cdaw["v_lin"].fillna(0).values
    widths = cdaw["width"].fillna(0).values
    halos = cdaw["is_halo"].values

    def assign(i: int, ev: pd.Series, method: str, conf: str, score: float) -> None:
        used.add(i)
        cdaw.at[i, "rc_id"] = ev["rc_id"]
        cdaw.at[i, "match_method"] = method
        cdaw.at[i, "match_confidence"] = conf
        cdaw.at[i, "match_score"] = score
        cdaw.at[i, "t_shock"] = ev["t_disturbance"]
        cdaw.at[i, "t_icme_start"] = ev["t_icme_start"]
        cdaw.at[i, "t_icme_end"] = ev["t_icme_end"]
        cdaw.at[i, "mc_flag"] = ev["mc_flag"]
        cdaw.at[i, "dst_rc"] = ev["dst_rc"]
        cdaw.at[i, "v_icme_rc"] = ev["v_icme"]

    # Pass 1: explicit R&C candidate list
    for _, ev in rc.iterrows():
        if not ev["lasco_list"]:
            continue
        best = None
        for ts, tag in ev["lasco_list"]:
            delta_h = np.abs((launch_times - np.datetime64(ts)) / np.timedelta64(1, "h"))
            close = np.where((delta_h <= 2.5) & (~np.isin(np.arange(len(cdaw)), list(used))))[0]
            for i in close:
                tag_bonus = 0.6 if ("H" in tag.upper()) else 0.0
                score = (
                    np.log1p(speeds[i]) + 0.5 * widths[i] / 360.0 + tag_bonus - 0.2 * delta_h[i]
                )
                if best is None or score > best[0]:
                    best = (score, i)
        if best is not None:
            assign(best[1], ev, "rc_list", "high", float(best[0]))

    # Pass 2: inferred association for events without a usable list
    for _, ev in rc.iterrows():
        if cdaw["rc_id"].eq(ev["rc_id"]).any():
            continue
        t_shock = ev["t_disturbance"]
        lo, hi = t_shock - pd.Timedelta(hours=120), t_shock - pd.Timedelta(hours=1)
        cand = cdaw[(cdaw["t_launch"] >= lo) & (cdaw["t_launch"] <= hi) & (~cdaw.index.isin(used))]
        if cand.empty:
            continue
        transit = (t_shock - cand["t_launch"]).dt.total_seconds() / 3600.0
        v0 = cand["v_lin"].where(cand["v_lin"] > 0)
        # projection allowance: CDAW speeds are plane-of-sky lower bounds
        gamma = [
            gamma_from_transit(float(v) * 1.8, 400.0, float(t)) if np.isfinite(v) else np.nan
            for v, t in zip(v0.values, transit.values)
        ]
        gamma = pd.Series(gamma, index=cand.index)
        plausible = gamma.notna() & (gamma <= 3.0e-7) & (cand["v_lin"] * 2.5 * transit * 3600.0 >= AU_KM)
        if not plausible.any():
            continue
        score = (
            np.log1p(cand["v_lin"].fillna(0))
            + 0.5 * cand["width"].fillna(0) / 360.0
            + 0.4 * cand["is_halo"].fillna(0)
        ).where(plausible, -np.inf)
        i = int(score.idxmax())
        assign(i, ev, "inferred", "medium", float(score.loc[i]))

    cdaw["is_impact"] = cdaw["rc_id"].notna().astype(int)
    cdaw["transit_hours"] = (cdaw["t_shock"] - cdaw["t_launch"]).dt.total_seconds() / 3600.0
    # Ambiguous: another CME launched within +-24 h of a matched CME (competing candidate)
    matched_t = cdaw.loc[cdaw["is_impact"] == 1, "t_launch"].sort_values().values
    label = cdaw["is_impact"].copy()
    if len(matched_t):
        for i, t in zip(cdaw.index, cdaw["t_launch"].values):
            if label.loc[i] == 1:
                continue
            delta = np.min(np.abs((matched_t - t) / np.timedelta64(1, "h")))
            if delta <= 24.0:
                label.loc[i] = -1
    cdaw["label"] = label
    return cdaw


def main() -> None:
    cdaw = parse_cdaw(CDAW_TXT)
    rc = parse_rc(RC_HTM)
    print(f"[CDAW] parsed {len(cdaw)} CMEs  {cdaw.t_launch.min()} .. {cdaw.t_launch.max()}")
    print(f"[R&C ] parsed {len(rc)} disturbances  {rc.t_disturbance.min()} .. {rc.t_disturbance.max()}")
    print(f"[CDAW] accel_true corr(speed): {cdaw[['v_lin','accel_true']].corr().iloc[0,1]:.4f} "
          f"(legacy 0.8069 was v_20R)")

    cat = match_catalog(cdaw, rc)

    omni = load_omni()
    targets = []
    for _, r in cat.iterrows():
        if pd.isna(r["t_shock"]):
            targets.append({})
            continue
        t1 = max(r["t_icme_end"], r["t_shock"] + pd.Timedelta(hours=36))
        t1 = min(t1, r["t_shock"] + pd.Timedelta(hours=96))
        targets.append(measure_window(omni, r["t_shock"] - pd.Timedelta(hours=2), t1))
    tdf = pd.DataFrame(targets, index=cat.index)
    for c in tdf.columns:
        cat[c] = tdf[c]

    # Target provenance: OMNI observation first; if the OMNI window contains a data
    # gap that swallowed the main phase (e.g. the Halloween 2003 ACE outage) the R&C
    # tabulated Dst minimum is used instead.  Never imputed, never overridden.
    use_rc = (
        cat["dst_min_obs"].notna() & cat["dst_rc"].notna() & (cat["dst_rc"] < cat["dst_min_obs"] - 15.0)
    )
    cat["dst_target_source"] = np.select(
        [use_rc, cat["dst_min_obs"].notna(), cat["dst_rc"].notna()],
        ["rc_gap", "omni", "rc"],
        default="missing",
    )
    cat["dst_min_target"] = np.where(use_rc, cat["dst_rc"], cat["dst_min_obs"])
    cat["dst_min_target"] = cat["dst_min_target"].fillna(cat["dst_rc"])

    cat["year"] = cat["t_launch"].dt.year
    cat["solar_cycle"] = np.where(cat["year"] <= 2008, 23, np.where(cat["year"] <= 2019, 24, 25))

    # Pre-conditioning in the CME sequence (strictly causal, launch times only)
    cat = cat.sort_values("t_launch").reset_index(drop=True)
    for win_h in (6, 24, 36):
        cnt, vmax, wmax = [], [], []
        for t in cat["t_launch"]:
            prior = cat[(cat["t_launch"] >= t - pd.Timedelta(hours=win_h)) & (cat["t_launch"] < t)]
            cnt.append(len(prior))
            vmax.append(prior["v_lin"].max() if len(prior) else np.nan)
            wmax.append(prior["width"].max() if len(prior) else np.nan)
        cat[f"prior_{win_h}h_count"] = cnt
        cat[f"prior_{win_h}h_max_speed"] = vmax
        cat[f"prior_{win_h}h_max_width"] = wmax

    out = PROC / "v3_cme_catalog.parquet"
    cat.to_parquet(out, index=False)

    pos = int((cat["label"] == 1).sum())
    amb = int((cat["label"] == -1).sum())
    audit = {
        "n_cme": int(len(cat)),
        "n_rc_events": int(len(rc)),
        "n_rc_with_candidate_list": int((rc["n_lasco_candidates"] > 0).sum()),
        "n_matched": pos,
        "n_ambiguous": amb,
        "match_method": cat.loc[cat.is_impact == 1, "match_method"].value_counts().to_dict(),
        "match_confidence": cat.loc[cat.is_impact == 1, "match_confidence"].value_counts().to_dict(),
        "accel_corr_speed": float(cat[["v_lin", "accel_true"]].corr().iloc[0, 1]),
        "v20r_corr_speed": float(cat[["v_lin", "v_20R"]].corr().iloc[0, 1]),
        "accel_true_median": float(cat["accel_true"].median()),
        "v20r_median": float(cat["v_20R"].median()),
        "targets_present": int(cat["dst_min_target"].notna().sum()),
        "dst_target_source": cat["dst_target_source"].value_counts().to_dict(),
        "halo_precision_rule": float(cat.loc[cat.is_halo == 1, "is_impact"].mean()),
        "halo_precision_rule_high_conf": float(
            cat.loc[(cat.is_halo == 1) & (cat.match_confidence == "high"), "is_impact"].mean()
        ),
        "n_halo": int(cat["is_halo"].sum()),
        "cycle25_pos": int(((cat.solar_cycle == 25) & (cat.label == 1)).sum()),
        "transit_median_h": float(cat.loc[cat.is_impact == 1, "transit_hours"].median()),
        "dst_min_obs_median": float(cat["dst_min_target"].median()),
        "positives_by_cycle": cat.loc[cat.label == 1].groupby("solar_cycle").size().to_dict(),
    }
    (RESULTS / "v3_catalog_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    print(f"[OK] {out}  shape={cat.shape}")


if __name__ == "__main__":
    main()
