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


def add_ebv_terms(df: pd.DataFrame, ebv_col: str, hbsag_col: str, out_col: str = "ln_ebv_1000_plus1") -> pd.DataFrame:
    out = df.copy()
    ebv = pd.to_numeric(out[ebv_col], errors="raise")
    if (ebv < 0).any():
        raise ValueError("EBV DNA must be nonnegative and measured in copies/mL.")
    out[out_col] = np.log1p(ebv / 1000.0)
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
    dat = dat.replace([np.inf, -np.inf], np.nan).dropna()
    categorical = [c for c in covariates if c == "ajcc_stage" or not pd.api.types.is_numeric_dtype(dat[c])]
    dat = pd.get_dummies(dat, columns=categorical, drop_first=True, dtype=float)
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
        terms=["ln_ebv_1000_plus1", hbsag_col, "ebv_hbsag_interaction"],
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
    observed = df[[ebv_col, hbsag_col, time_col, event_col, *covariates]].dropna()
    ebv_vals = np.linspace(observed[ebv_col].min(), observed[ebv_col].quantile(0.99), n_grid)
    # All covariates are held at common values, so their coefficients cancel in the log HR.
    beta = model.params_
    ref = np.log1p(observed[ebv_col].median() / 1000.0)
    rows = []
    for h in [0, 1]:
        for ebv in ebv_vals:
            x = np.log1p(ebv / 1000.0)
            log_hr = beta["ln_ebv_1000_plus1"] * (x - ref) + h * (beta[hbsag_col] + x * beta["ebv_hbsag_interaction"])
            rows.append({"ebv_dna": ebv, "hbsag": h, "hazard_ratio": np.exp(log_hr)})
    return pd.DataFrame(rows)


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
        cohort = clinical
        if endpoint in ("dmfs", "lrrfs"):
            cohort = clinical.copy()
            os_cols = endpoints["os"]
            competing = cohort[cols["event"]].eq(0) & cohort[os_cols["event"]].eq(1)
            cohort.loc[competing, cols["time"]] = cohort.loc[competing, [cols["time"], os_cols["time"]]].min(axis=1, skipna=False)
        result, model, dat = fit_interaction_model(
            cohort,
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
            cohort,
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
