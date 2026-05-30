from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


def plot_threshold_scan(scan: pd.DataFrame, locked_cutoff: float, out_path: str | Path, title: str = "Locked-threshold derivation") -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(scan["cutoff"] / 1000.0, scan["hr_hbsag_negative"], label="HBsAg-")
    ax.plot(scan["cutoff"] / 1000.0, scan["hr_hbsag_positive"], label="HBsAg+")
    ax.axvline(locked_cutoff / 1000.0, linestyle="--")
    ax.axhline(1.0, linestyle=":")
    ax.set_xlabel("EBV DNA cutoff (x10^3 copies/mL)")
    ax.set_ylabel("Hazard ratio")
    ax.set_title(title)
    ax.legend(frameon=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_km(df: pd.DataFrame, time_col: str, event_col: str, group_col: str, out_path: str | Path, title: str = "Kaplan-Meier") -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    kmf = KaplanMeierFitter()
    for name, sub in df.groupby(group_col):
        kmf.fit(sub[time_col], sub[event_col], label=str(name))
        kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    ax.set_title(title)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_bootstrap_thresholds(boot: pd.DataFrame, locked_cutoff: float, out_path: str | Path) -> None:
    vals = boot["locked_cutoff"].dropna() / 1000.0
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(vals, bins=40, alpha=0.8)
    ax.axvline(locked_cutoff / 1000.0, linestyle="--", label="Locked cutoff")
    ax.axvspan(locked_cutoff * 0.9 / 1000.0, locked_cutoff * 1.1 / 1000.0, alpha=0.15, label="±10% band")
    ax.set_xlabel("Optimal EBV DNA cutoff (x10^3 copies/mL)")
    ax.set_ylabel("Bootstrap count")
    ax.legend(frameon=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
