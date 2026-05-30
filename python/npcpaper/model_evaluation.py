from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index


def c_index(df: pd.DataFrame, time_col: str, event_col: str, score_col: str) -> float:
    dat = df[[time_col, event_col, score_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if dat.empty or dat[event_col].nunique() < 2:
        return np.nan
    return float(concordance_index(dat[time_col], -dat[score_col], dat[event_col]))


def bootstrap_c_index(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    score_col: str,
    n_bootstrap: int = 1000,
    seed: int = 1234,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    dat = df[[time_col, event_col, score_col]].dropna().reset_index(drop=True)
    n = dat.shape[0]
    if n == 0:
        return np.nan, np.nan, np.nan
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        b = dat.iloc[idx]
        if b[event_col].nunique() > 1:
            vals.append(c_index(b, time_col, event_col, score_col))
    vals = np.array(vals)
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.nanmean(vals)), float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))


def training_median_cutoff(df_train: pd.DataFrame, score_col: str) -> float:
    return float(df_train[score_col].median())


def assign_risk_group(df: pd.DataFrame, score_col: str, cutoff: float) -> pd.DataFrame:
    out = df.copy()
    out["risk_group"] = np.where(out[score_col] >= cutoff, "High Risk", "Low Risk")
    return out


def km_summary(df: pd.DataFrame, time_col: str, event_col: str, group_col: str = "risk_group") -> dict[str, float | int]:
    groups = list(df[group_col].dropna().unique())
    res: dict[str, float | int] = {"n": int(df.shape[0]), "events": int(df[event_col].sum())}
    if len(groups) == 2:
        a, b = groups
        aa = df[df[group_col] == a]
        bb = df[df[group_col] == b]
        lr = logrank_test(aa[time_col], bb[time_col], aa[event_col], bb[event_col])
        res["logrank_p"] = float(lr.p_value)
    return res


def evaluate_scores(
    df: pd.DataFrame,
    endpoints: dict[str, dict[str, str]],
    score_cols: dict[str, str],
    cohort_col: str,
    train_label: str,
    outdir: str | Path,
    n_bootstrap: int = 1000,
    seed: int = 1234,
) -> pd.DataFrame:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for endpoint, cols in endpoints.items():
        score_col = score_cols[endpoint]
        train = df[df[cohort_col] == train_label]
        cutoff = training_median_cutoff(train, score_col)
        for cohort, sub in df.groupby(cohort_col):
            sub = assign_risk_group(sub, score_col, cutoff)
            mean_ci, lo, hi = bootstrap_c_index(sub, cols["time"], cols["event"], score_col, n_bootstrap=n_bootstrap, seed=seed)
            km = km_summary(sub, cols["time"], cols["event"])
            rows.append({
                "endpoint": endpoint,
                "cohort": cohort,
                "score_col": score_col,
                "training_median_cutoff": cutoff,
                "c_index": mean_ci,
                "c_index_ci_lower": lo,
                "c_index_ci_upper": hi,
                **km,
            })
    res = pd.DataFrame(rows)
    res.to_csv(outdir / "score_evaluation_summary.csv", index=False)
    return res


def decision_curve_binary(y_true: np.ndarray, risk: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """Simple decision-curve net benefit for binary event by a fixed time horizon."""
    y_true = np.asarray(y_true).astype(int)
    risk = np.asarray(risk).astype(float)
    n = len(y_true)
    rows = []
    for pt in thresholds:
        pred = risk >= pt
        tp = np.sum(pred & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        nb = tp / n - fp / n * pt / (1 - pt)
        rows.append({"threshold": float(pt), "net_benefit": float(nb)})
    return pd.DataFrame(rows)
