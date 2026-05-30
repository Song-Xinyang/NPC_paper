from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


@dataclass
class InteractionResult:
    endpoint: str
    n: int
    events: int
    hr_interaction: float
    ci_lower: float
    ci_upper: float
    p_interaction: float


def add_ebv_terms(df: pd.DataFrame, ebv_col: str, hbsag_col: str, out_col: str = "log10_ebv_plus1") -> pd.DataFrame:
    out = df.copy()
    out[out_col] = np.log10(pd.to_numeric(out[ebv_col], errors="coerce").fillna(0).clip(lower=0) + 1.0)
    out[hbsag_col] = pd.to_numeric(out[hbsag_col], errors="coerce").astype(float)
    out["ebv_hbsag_interaction"] = out[out_col] * out[hbsag_col]
    return out


def _prepare_design(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    covariates: Iterable[str],
    terms: Iterable[str],
) -> pd.DataFrame:
    keep = [time_col, event_col, *terms, *covariates]
    dat = df.loc[:, keep].copy()
    dat = pd.get_dummies(dat, columns=[c for c in covariates if dat[c].dtype == "object"], drop_first=True)
    dat = dat.replace([np.inf, -np.inf], np.nan).dropna()
    dat[event_col] = dat[event_col].astype(int)
    return dat


def fit_interaction_model(
    df: pd.DataFrame,
    endpoint: str,
    time_col: str,
    event_col: str,
    ebv_col: str,
    hbsag_col: str,
    covariates: list[str],
    penalizer: float = 0.0,
) -> tuple[InteractionResult, CoxPHFitter, pd.DataFrame]:
    dat0 = add_ebv_terms(df, ebv_col=ebv_col, hbsag_col=hbsag_col)
    dat = _prepare_design(
        dat0,
        time_col=time_col,
        event_col=event_col,
        covariates=covariates,
        terms=["log10_ebv_plus1", hbsag_col, "ebv_hbsag_interaction"],
    )
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(dat, duration_col=time_col, event_col=event_col)
    row = cph.summary.loc["ebv_hbsag_interaction"]
    ci = cph.confidence_intervals_.loc["ebv_hbsag_interaction"]
    result = InteractionResult(
        endpoint=endpoint,
        n=int(dat.shape[0]),
        events=int(dat[event_col].sum()),
        hr_interaction=float(np.exp(row["coef"])),
        ci_lower=float(np.exp(ci.iloc[0])),
        ci_upper=float(np.exp(ci.iloc[1])),
        p_interaction=float(row["p"]),
    )
    return result, cph, dat


def estimate_hr_trajectories(
    model: CoxPHFitter,
    df: pd.DataFrame,
    ebv_col: str,
    hbsag_col: str,
    covariates: list[str],
    time_col: str,
    event_col: str,
    n_grid: int = 200,
) -> pd.DataFrame:
    """Create HBsAg-stratified relative hazard trajectories over EBV DNA values.

    The baseline row uses median numeric covariates and modal categorical covariates.
    Hazards are normalized to the HBsAg-negative row at median EBV DNA.
    """
    ebv_vals = np.linspace(float(df[ebv_col].min()), float(df[ebv_col].quantile(0.99)), n_grid)
    base = {}
    for c in covariates:
        if pd.api.types.is_numeric_dtype(df[c]):
            base[c] = float(df[c].median())
        else:
            base[c] = df[c].mode(dropna=True).iloc[0]
    rows = []
    for h in [0, 1]:
        for ebv in ebv_vals:
            r = dict(base)
            r[time_col] = 1.0
            r[event_col] = 0
            r["log10_ebv_plus1"] = np.log10(max(ebv, 0) + 1.0)
            r[hbsag_col] = h
            r["ebv_hbsag_interaction"] = r["log10_ebv_plus1"] * h
            rows.append({"ebv_dna": ebv, "hbsag": h, **r})
    pred = pd.DataFrame(rows)
    pred_design = pd.get_dummies(pred.drop(columns=["ebv_dna", "hbsag"]), drop_first=True)
    for col in model.params_.index:
        if col not in pred_design.columns:
            pred_design[col] = 0
    pred_design = pred_design[model.params_.index]
    log_h = np.dot(pred_design.values, model.params_.values)
    ref_idx = np.argmin(np.abs(pred["ebv_dna"].values - np.median(df[ebv_col].values)))
    ref_log_h = log_h[ref_idx]
    pred["hazard_ratio"] = np.exp(log_h - ref_log_h)
    return pred[["ebv_dna", "hbsag", "hazard_ratio"]]


def run_interaction_analysis(
    clinical: pd.DataFrame,
    endpoints: dict[str, dict[str, str]],
    ebv_col: str,
    hbsag_col: str,
    covariates: list[str],
    outdir: str | Path,
) -> pd.DataFrame:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for endpoint, cols in endpoints.items():
        result, model, dat = fit_interaction_model(
            clinical,
            endpoint=endpoint,
            time_col=cols["time"],
            event_col=cols["event"],
            ebv_col=ebv_col,
            hbsag_col=hbsag_col,
            covariates=covariates,
        )
        rows.append(result.__dict__)
        model.summary.to_csv(outdir / f"cox_interaction_{endpoint}.csv")
        traj = estimate_hr_trajectories(
            model,
            clinical,
            ebv_col=ebv_col,
            hbsag_col=hbsag_col,
            covariates=covariates,
            time_col=cols["time"],
            event_col=cols["event"],
        )
        traj.to_csv(outdir / f"hr_trajectory_{endpoint}.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(outdir / "interaction_summary.csv", index=False)
    return res
