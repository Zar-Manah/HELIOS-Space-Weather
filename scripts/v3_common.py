#!/usr/bin/env python3
"""HELIOS v3.4 shared feature assembly (strictly causal at t_launch)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

CYCLE_START = {23: pd.Timestamp("1996-05-01"), 24: pd.Timestamp("2008-12-01"), 25: pd.Timestamp("2019-12-01")}


def load_cme_features() -> pd.DataFrame:
    df = pd.read_parquet(PROC / "v3_solar_sources.parquet")
    df["t_launch"] = pd.to_datetime(df["t_launch"])

    sharp_path = PROC / "v3_sharp_features.parquet"
    if sharp_path.exists():
        sh = pd.read_parquet(sharp_path)
        df = df.merge(sh, on="cme_id", how="left")
    ctx_path = PROC / "v3_omni_context.parquet"
    if ctx_path.exists():
        ctx = pd.read_parquet(ctx_path)
        ctx["datetime"] = pd.to_datetime(ctx["datetime"])
        ctx = ctx.sort_values("datetime")
        df = pd.merge_asof(
            df.sort_values("t_launch"), ctx, left_on="t_launch", right_on="datetime",
            direction="backward", tolerance=pd.Timedelta(hours=2),
        )

    # cycle phase (causal, based on published cycle start dates)
    phase = []
    for t, c in zip(df["t_launch"], df["solar_cycle"]):
        start = CYCLE_START.get(int(c), pd.Timestamp("2019-12-01"))
        yrs = (t - start).days / 365.25
        phase.append(2 * np.pi * yrs / 11.0)
    df["cycle_phase_sin"] = np.sin(phase)
    df["cycle_phase_cos"] = np.cos(phase)

    # derived physics features
    df["log_momentum"] = np.log10((df["mass"].where(df["mass"] > 0) * df["v_lin"].where(df["v_lin"] > 0)))
    df["cpa_sin"] = np.sin(np.radians(df["cpa"].fillna(0)))
    df["cpa_cos"] = np.cos(np.radians(df["cpa"].fillna(0)))
    df["abs_lon"] = df["src_lon"].abs()
    df["east_west"] = np.sign(df["src_lon"])
    df["is_west"] = (df["src_lon"] > 0).astype(int)
    df["is_east"] = (df["src_lon"] < 0).astype(int)
    df["log_sharp_usflux"] = np.log10(df["sharp_usflux"].where(df["sharp_usflux"] > 0)) \
        if "sharp_usflux" in df else np.nan
    df["log_sharp_totpot"] = np.log10(df["sharp_totpot"].where(df["sharp_totpot"] > 0)) \
        if "sharp_totpot" in df else np.nan
    df["log_sharp_totusjh"] = np.log10(df["sharp_totusjh"].where(df["sharp_totusjh"] > 0)) \
        if "sharp_totusjh" in df else np.nan
    df["sharp_available"] = (df["sharp_usflux"].notna().astype(int)
                             if "sharp_usflux" in df else 0)
    return df


IMPACT_FEATURES = [
    # CDAW morphology & kinematics
    "width", "is_halo", "is_partial_halo", "is_narrow", "cpa_sin", "cpa_cos",
    "v_lin", "log_speed", "accel_true", "accel_missing", "v_20R", "v_20r_ratio",
    "dv_2nd", "log_mass", "log_energy", "log_momentum",
    "flag_poor", "flag_very_poor", "flag_only_c2", "flag_only_c3",
    # source geometry: the flag is a no-witness halo candidate, not a proven backside
    # origin; has_position / unknown_source keep the "frontside but no witness" case apart
    "has_source", "has_position", "backside_confirmed", "unknown_source",
    "src_lon", "src_lat", "abs_lon", "is_west", "is_east",
    "d_earth_deg", "cos_deprojection", "v_3d_est", "cone_margin_deg",
    "flank_margin_deg", "no_source_halo",
    # flare driver
    "log_goes_flux", "flare_cme_dt_min", "pos_angle_consistency", "ar_num",
    # preconditioning
    "prior_6h_count", "prior_6h_max_speed", "prior_24h_count", "prior_24h_max_speed",
    "prior_36h_count", "prior_36h_max_speed", "prior_36h_max_width",
    # ambient heliosphere
    "t0_dst", "t0_v_flow", "t0_bz_gsm", "t0_pdyn", "v_mean_24h", "v_max_24h",
    "bz_min_24h", "dst_min_48h", "pdyn_mean_24h", "carr_v_flow", "carr_np_dens",
    # SHARP
    "sharp_available", "log_sharp_usflux", "log_sharp_totpot", "log_sharp_totusjh",
    "sharp_r_value", "sharp_meanshr", "sharp_shrgt45", "sharp_area_acr",
    "d_totpot_24h", "d_totusjh_24h", "d_r_value_24h",
    # cycle
    "solar_cycle", "cycle_phase_sin", "cycle_phase_cos",
]

MONOTONE = {
    "d_earth_deg": -1,
    "width": 1,
    "v_3d_est": 1,
    "has_source": 1,
    "has_position": 1,
    "backside_confirmed": -1,
    "unknown_source": -1,
    "cone_margin_deg": 1,
}


def design_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df.reindex(columns=features).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    return X.astype(float)


def precision_at_recall(y: np.ndarray, p: np.ndarray, target: float) -> float:
    order = np.argsort(-p)
    y = np.asarray(y)[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    rec = tp / max(y.sum(), 1)
    idx = np.where(rec >= target)[0]
    return float(prec[idx[0]]) if len(idx) else float("nan")


def evaluate_binary(y, p, threshold=None) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    out = {
        "n": int(len(y)),
        "pos": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "brier_skill": float(1 - brier_score_loss(y, p) / max(brier_score_loss(y, np.full_like(p, y.mean())), 1e-12)),
    }
    for r in (0.5, 0.7, 0.8):
        out[f"precision_at_recall_{int(r*100)}"] = precision_at_recall(y, p, r)
    if threshold is None:
        # choose threshold maximising F1 on this set (report-only, never used for fitting)
        order = np.argsort(-p)
        tp = np.cumsum(y[order])
        prec = tp / np.arange(1, len(y) + 1)
        rec = tp / max(y.sum(), 1)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
        threshold = float(p[order[int(np.argmax(f1))]])
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    out.update({
        "threshold": float(threshold),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "far": fp / max(fp + tn, 1) if (tn := int(((~pred) & (y == 0)).sum())) >= 0 else float("nan"),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    })
    return out


# ---------------------------------------------------------------------------
# Exact drag-based-model solver (Vrsnak 2001), shared by transit and apex.
# ---------------------------------------------------------------------------
AU_KM = 1.495978707e8


def dbm_transit_vec(v0: np.ndarray, w: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Vectorised exact DBM transit time (hours) to 1 AU.  inf if unreachable."""
    v0 = np.asarray(v0, float)
    w = np.asarray(w, float)
    gamma = np.asarray(gamma, float)
    out = np.full(v0.shape, np.inf)
    ok = np.isfinite(v0) & np.isfinite(w) & np.isfinite(gamma) & (v0 > w) & (gamma >= 0)
    if not ok.any():
        return out
    v0, w, g = v0[ok], w[ok], gamma[ok]
    lo = np.zeros_like(v0)
    hi = np.full_like(v0, 500.0 * 3600.0)
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            dist = w * mid + np.where(
                g > 0, np.log1p(g * (v0 - w) * mid) / np.maximum(g, 1e-15), (v0 - w) * mid
            )
        lo = np.where(dist < AU_KM, mid, lo)
        hi = np.where(dist < AU_KM, hi, mid)
    out[ok] = 0.5 * (lo + hi) / 3600.0
    return out


def dbm_speed_vec(v0, w, gamma):
    t = dbm_transit_vec(v0, w, gamma) * 3600.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return w + 1.0 / (1.0 / (v0 - w) + gamma * t)


def effective_launch_speed(df: pd.DataFrame) -> np.ndarray:
    """De-projected launch speed (plane-of-sky speeds are lower bounds)."""
    v = df["v_lin"].values.astype(float)
    cosd = df["cos_deprojection"].values.astype(float) if "cos_deprojection" in df else np.ones(len(df))
    has = df["has_source"].values.astype(float) if "has_source" in df else np.zeros(len(df))
    return np.where(has == 1, v / np.clip(cosd, 0.45, 1.0), v)


AU_KM = 1.495978707e8

TRANSIT_BASE_FEATURES = ["v_lin", "width", "v_20R", "is_halo", "v_diff_ambient",
                         "drag_accel_proxy", "transit_hours_dbm_prior",
                         "is_fast_cme", "sin_cycle", "cos_cycle"]
TRANSIT_DBM_GRID = [(350.0, 1e-8), (400.0, 1e-8), (450.0, 3e-8),
                    (500.0, 3e-8), (350.0, 3e-8), (450.0, 1e-8)]
TRANSIT_GRID_COLS = [f"dbm{i}_w{int(w)}_g{int(round(-np.log10(g)))}"
                     for i, (w, g) in enumerate(TRANSIT_DBM_GRID)]
TRANSIT_FEATURES = TRANSIT_BASE_FEATURES + TRANSIT_GRID_COLS


def add_transit_features(d: pd.DataFrame) -> pd.DataFrame:
    """Deterministic transit feature block; identical at train and serve time."""
    d = d.copy()
    v = pd.to_numeric(d["v_lin"], errors="coerce")
    v = v.fillna(v.median() if np.isfinite(v.median()) else 400.0).values
    d["v_diff_ambient"] = v - 420.0
    d["drag_accel_proxy"] = -1e-7 * d["v_diff_ambient"] * np.abs(d["v_diff_ambient"])
    k = 1.0 - np.exp(-1e-7 * AU_KM * 0.4)
    d["v_effective_dbm"] = v * np.exp(-1e-7 * AU_KM * 0.4) + 420.0 * k
    d["transit_hours_dbm_prior"] = AU_KM / (np.clip(d["v_effective_dbm"], 50.0, None) * 3600.0)
    d["is_fast_cme"] = (v >= 1000).astype(int)
    t = pd.to_datetime(d["t_launch"])
    ph = ((t.dt.year - 1996) % 11) / 11.0
    d["sin_cycle"] = np.sin(2 * np.pi * ph)
    d["cos_cycle"] = np.cos(2 * np.pi * ph)
    for i, (w, g) in enumerate(TRANSIT_DBM_GRID):
        tt = dbm_transit_vec(v, np.full(len(v), w), np.full(len(v), g))
        ball = AU_KM / np.clip(v, 50.0, None) / 3600.0
        d[TRANSIT_GRID_COLS[i]] = np.where(np.isfinite(tt), tt, ball)
    return d


def transit_physics(df: pd.DataFrame, physics: dict) -> tuple:
    """Fitted effective DBM: returns (t_phys_h, v1au_km_s, reachable_mask)."""
    v0 = pd.to_numeric(df["v_lin"], errors="coerce")
    v0 = v0.fillna(v0.median() if np.isfinite(v0.median()) else 400.0).values.astype(float)
    w = np.full(len(df), float(physics["w_amb_km_s"]))
    g = np.full(len(df), float(physics["gamma_km_inv"]))
    t = dbm_transit_vec(v0, w, g)
    reach = np.isfinite(t)
    ball = AU_KM / np.clip(v0, 50.0, None) / 3600.0
    t = np.where(reach, t, ball)
    v1 = dbm_speed_vec(v0, w, g)
    v1 = np.where(np.isfinite(v1), v1, v0)
    return t, v1, reach


def attach_transit_physics(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the fitted physics backbone (t_phys, v1au_dbm) to a feature frame."""
    import joblib

    df = df.copy()
    path = ROOT / "models" / "v3" / "transit_v3.joblib"
    if not path.exists():
        df["t_phys"] = np.nan
        df["v1au_dbm"] = np.nan
        df["v_eff_launch"] = effective_launch_speed(df)
        return df
    blob = joblib.load(path)
    phys = blob.get("physics") or {}
    if "w_amb_km_s" not in phys:
        df["t_phys"] = np.nan
        df["v1au_dbm"] = np.nan
        df["v_eff_launch"] = effective_launch_speed(df)
        return df
    t, v1, _ = transit_physics(df, phys)
    df["t_phys"] = t
    df["v1au_dbm"] = v1
    df["v_eff_launch"] = effective_launch_speed(df)
    return df


class MeanRegressor:
    """Pickle-safe average of heterogeneous regressors.

    Defined at module level so joblib serialises it as `v3_common.MeanRegressor`
    and predict.py can load it without importing the training script.
    """

    def __init__(self, models):
        self.models = list(models)

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


class ResidualTransit:
    """Learner on the residual of the *dynamic* DBM, exposed as a transit model.

    The physics backbone (w_amb, gamma) is the one fitted on the pre-2020
    window; the tree only has to correct what the drag model gets wrong, which
    is a much easier target than the absolute transit time.  Defined here so
    joblib pickles it as `v3_common.ResidualTransit`.
    """

    def __init__(self, model, physics: dict):
        self.model = model
        self.physics = dict(physics)

    def predict(self, X):
        v0 = np.asarray(pd.to_numeric(pd.Series(np.asarray(X["v_lin"], dtype=float)),
                                      errors="coerce"), dtype=float)
        t, _, _ = transit_physics(pd.DataFrame({"v_lin": v0}), self.physics)
        if isinstance(self.model, (list, tuple)):
            base = np.mean([m.predict(X) for m in self.model], axis=0)
        else:
            base = np.asarray(self.model.predict(X), dtype=float)
        return base + t


class RankerProba:
    """LambdaRank member exposed as a probability for the stacking ensemble.

    Lives here (not in the training script) so joblib pickles it as
    `v3_common.RankerProba`: a class defined in `__main__` cannot be unpickled
    by predict.py or by the verification suite.  The logistic squash is strictly
    monotone, so the ranking the ranker produces is preserved.
    """

    def __init__(self, ranker, mu: float, sd: float):
        self.ranker = ranker
        self.mu = float(mu)
        self.sd = float(sd) if sd and np.isfinite(sd) and sd > 1e-9 else 1.0

    def predict_proba(self, X):
        z = (self.ranker.predict(np.asarray(X, dtype=float)) - self.mu) / self.sd
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return np.column_stack([1.0 - p, p])


PHYS_INTERACTION_COLS = ["abs_lon", "width", "cos_deprojection", "v_3d_est",
                         "sharp_usflux", "sharp_totpot", "d_totpot_24h",
                         "pa_consistency"]
# `pa_consistency` is a constant column on the shipped catalogue (no PA
# measurement survives the merge).  It is kept in the interaction set because the
# frozen members were fitted with it in that position: renaming or dropping it
# now would create a train/serve skew.  Its coefficient is unidentifiable, which
# is harmless for prediction and is stated here so nobody mistakes it for signal.


class PhysicsInteractionLR:
    """Regularised logistic regression on the key physical interactions.

    A linear member sees the pairwise products the trees have to discover by
    splitting: longitude x width, de-projection x speed, SHARP flux x potential.
    Medians are learned on the training fold (no leakage) and reused at serve.
    """

    def __init__(self, cols=None, C: float = 0.2):
        self.cols = list(cols or PHYS_INTERACTION_COLS)
        self.C = float(C)

    def _design(self, X) -> np.ndarray:
        d = pd.DataFrame(X)
        base = pd.DataFrame(index=d.index)
        for c in self.cols:
            base[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
        base = base.replace([np.inf, -np.inf], np.nan)
        base = base.fillna(pd.Series(self.med_, index=self.cols))
        vals = base.values
        cols = [vals]
        for i in range(len(self.cols)):
            for j in range(i + 1, len(self.cols)):
                cols.append((vals[:, i] * vals[:, j])[:, None])
        return np.hstack(cols)

    def fit(self, X, y):
        d = pd.DataFrame(X)
        med = []
        for c in self.cols:
            v = pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(dtype=float)
            m = float(np.nanmedian(v.values)) if len(v) and np.isfinite(v.values).any() else 0.0
            med.append(m)
        self.med_ = np.asarray(med, float)
        Z = self._design(X)
        self.scaler_ = StandardScaler().fit(Z)
        self.lr_ = LogisticRegression(C=self.C, max_iter=3000).fit(self.scaler_.transform(Z), y)
        return self

    def predict_proba(self, X):
        Z = self.scaler_.transform(self._design(X))
        return self.lr_.predict_proba(Z)
