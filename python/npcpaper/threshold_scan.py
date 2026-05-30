from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def _design(df: pd.DataFrame, time_col: str, event_col: str, covariates: Iterable[str], predictor: str) -> pd.DataFrame:
    cols = [time_col, event_col, predictor, *covariates]
    dat = df.loc[:, cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    dat = pd.get_dummies(dat, columns=[c for c in covariates if dat[c].dtype == "object"], drop_first=True)
    dat[event_col] = dat[event_col].astype(int)
    return dat


def _fit_hr(dat: pd.DataFrame, time_col: str, event_col: str, predictor: str) -> tuple[float, float, int, int]:
    if dat[predictor].nunique() < 2 or dat[event_col].sum() < 2:
        return np.nan, np.nan, dat.shape[0], int(dat[event_col].sum())
    try:
        cph = CoxPHFitter(penalizer=0.001)
        cph.fit(dat, duration_col=time_col, event_col=event_col)
        coef = float(cph.params_.loc[predictor])
        p = float(cph.summary.loc[predictor, "p"])
        return float(np.exp(coef)), p, dat.shape[0], int(dat[event_col].sum())
    except Exception:
        return np.nan, np.nan, dat.shape[0], int(dat[event_col].sum())


def valid_cutoff(
    df: pd.DataFrame,
    cutoff: float,
    ebv_col: str,
    hbsag_col: str,
    event_col: str,
    min_fraction_per_hbsag_arm: float,
    min_events_per_comparison_arm: int,
) -> bool:
    x = df.copy()
    x["ebv_high"] = (x[ebv_col] >= cutoff).astype(int)
    for h in [0, 1]:
        sub = x[x[hbsag_col] == h]
        if sub.empty:
            return False
        n = len(sub)
        for label in [0, 1]:
            arm = sub[sub["ebv_high"] == label]
            if len(arm) < min_fraction_per_hbsag_arm * n:
                return False
            if arm[event_col].sum() < min_events_per_comparison_arm:
                return False
    return True


def scan_thresholds(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    ebv_col: str,
    hbsag_col: str,
    covariates: list[str],
    n_grid: int = 300,
    min_fraction_per_hbsag_arm: float = 0.10,
    min_events_per_comparison_arm: int = 10,
) -> pd.DataFrame:
    ebv = pd.to_numeric(df[ebv_col], errors="coerce")
    if ebv.dropna().empty:
        return pd.DataFrame()
    lo, hi = ebv.quantile([0.02, 0.98]).values
    candidates = np.unique(np.quantile(ebv.dropna(), np.linspace(0.02, 0.98, n_grid)))
    candidates = candidates[(candidates >= lo) & (candidates <= hi)]
    rows = []
    for cutoff in candidates:
        if not valid_cutoff(df, cutoff, ebv_col, hbsag_col, event_col, min_fraction_per_hbsag_arm, min_events_per_comparison_arm):
            continue
        tmp = df.copy()
        tmp["ebv_high"] = (tmp[ebv_col] >= cutoff).astype(int)
        row = {"cutoff": float(cutoff)}
        for h, label in [(0, "hbsag_negative"), (1, "hbsag_positive")]:
            sub = tmp[tmp[hbsag_col] == h]
            dat = _design(sub, time_col, event_col, covariates, predictor="ebv_high")
            hr, p, n, events = _fit_hr(dat, time_col, event_col, predictor="ebv_high")
            row[f"hr_{label}"] = hr
            row[f"p_{label}"] = p
            row[f"n_{label}"] = n
            row[f"events_{label}"] = events
        if np.isfinite(row["hr_hbsag_positive"]) and np.isfinite(row["hr_hbsag_negative"]):
            row["log_hr_difference"] = np.log(row["hr_hbsag_positive"]) - np.log(row["hr_hbsag_negative"])
            row["score"] = row["log_hr_difference"]
        else:
            row["log_hr_difference"] = np.nan
            row["score"] = np.nan
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["cutoff", "score"])
    return pd.DataFrame(rows).sort_values("cutoff")


def choose_locked_threshold(scan: pd.DataFrame) -> dict[str, float]:
    if scan.empty or scan["score"].dropna().empty:
        raise ValueError("No valid candidate threshold under the specified stability constraints.")
    idx = scan["score"].idxmax()
    row = scan.loc[idx]
    return {"locked_cutoff": float(row["cutoff"]), "score": float(row["score"]), "hr_hbsag_positive": float(row["hr_hbsag_positive"]), "hr_hbsag_negative": float(row["hr_hbsag_negative"])}


def bootstrap_thresholds(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    ebv_col: str,
    hbsag_col: str,
    covariates: list[str],
    n_bootstrap: int = 1000,
    seed: int = 1234,
    **scan_kwargs,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    n = df.shape[0]
    for b in range(n_bootstrap):
        sample_idx = rng.choice(np.arange(n), size=n, replace=True)
        boot = df.iloc[sample_idx].reset_index(drop=True)
        try:
            scan = scan_thresholds(boot, time_col, event_col, ebv_col, hbsag_col, covariates, **scan_kwargs)
            choice = choose_locked_threshold(scan)
            choice["bootstrap"] = b + 1
        except Exception:
            choice = {"bootstrap": b + 1, "locked_cutoff": np.nan, "score": np.nan, "hr_hbsag_positive": np.nan, "hr_hbsag_negative": np.nan}
        out.append(choice)
    return pd.DataFrame(out)


def summarize_bootstrap(boot: pd.DataFrame, locked_value: float, tolerance_fraction: float = 0.10) -> dict[str, float]:
    vals = boot["locked_cutoff"].dropna()
    lower = locked_value * (1 - tolerance_fraction)
    upper = locked_value * (1 + tolerance_fraction)
    return {
        "n_success": int(vals.shape[0]),
        "median": float(vals.median()),
        "q1": float(vals.quantile(0.25)),
        "q3": float(vals.quantile(0.75)),
        "within_tolerance_fraction": float(((vals >= lower) & (vals <= upper)).mean()),
        "locked_value": float(locked_value),
        "tolerance_lower": float(lower),
        "tolerance_upper": float(upper),
    }
