from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import joblib


@dataclass
class LassoCoxModel:
    endpoint: str
    time_col: str
    event_col: str
    feature_cols: list[str]
    clinical_cols: list[str]
    selected_cols: list[str]
    matrix_cols: list[str]
    scaler: StandardScaler
    cox_model: CoxPHFitter
    penalty: float
    training_median_cutoff: float

    def predict_score(self, df: pd.DataFrame) -> pd.Series:
        X_raw = prepare_raw_matrix(df, self.feature_cols, self.clinical_cols)
        for c in self.matrix_cols:
            if c not in X_raw.columns:
                X_raw[c] = 0.0
        X_raw = X_raw[self.matrix_cols]
        X_scaled = pd.DataFrame(self.scaler.transform(X_raw.astype(float)), columns=self.matrix_cols, index=df.index)
        for c in self.selected_cols:
            if c not in X_scaled.columns:
                X_scaled[c] = 0.0
        X = X_scaled[self.selected_cols]
        return pd.Series(np.dot(X.values, self.cox_model.params_.loc[self.selected_cols].values), index=df.index, name=f"score_{self.endpoint}")


def prepare_raw_matrix(df: pd.DataFrame, feature_cols: list[str], clinical_cols: list[str]) -> pd.DataFrame:
    dat = df[feature_cols + clinical_cols].copy()
    cat_cols = [c for c in clinical_cols if dat[c].dtype == "object" or str(dat[c].dtype).startswith("category")]
    dat = pd.get_dummies(dat, columns=cat_cols, drop_first=True)
    dat = dat.replace([np.inf, -np.inf], np.nan)
    dat = dat.fillna(dat.median(numeric_only=True))
    return dat


def prepare_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    clinical_cols: list[str],
    fit_scaler: bool,
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    dat = prepare_raw_matrix(df, feature_cols, clinical_cols).astype(float)
    cols = list(dat.columns)
    if fit_scaler:
        scaler = StandardScaler()
        dat.loc[:, cols] = scaler.fit_transform(dat[cols].astype(float))
    else:
        if scaler is None:
            raise ValueError("scaler is required when fit_scaler=False")
        dat.loc[:, cols] = scaler.transform(dat[cols].astype(float))
    return dat, scaler


def univariable_screen(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    feature_cols: list[str],
    p_threshold: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        dat = df[[time_col, event_col, col]].replace([np.inf, -np.inf], np.nan).dropna()
        if dat[col].nunique() < 3 or dat[event_col].sum() < 3:
            continue
        try:
            cph = CoxPHFitter(penalizer=0.001)
            cph.fit(dat, time_col, event_col)
            rows.append({"feature": col, "coef": float(cph.params_[col]), "hr": float(np.exp(cph.params_[col])), "p": float(cph.summary.loc[col, "p"])})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["feature", "coef", "hr", "p"])
    res = pd.DataFrame(rows).sort_values("p")
    return res[res["p"] <= p_threshold]


def remove_correlated_features(df: pd.DataFrame, features: list[str], ranking: pd.DataFrame, threshold: float = 0.90) -> list[str]:
    if len(features) <= 1:
        return features
    rank_p = ranking.set_index("feature")["p"].to_dict()
    corr = df[features].corr(method="spearman").abs()
    keep = set(features)
    for i, a in enumerate(features):
        if a not in keep:
            continue
        for b in features[i + 1:]:
            if b not in keep:
                continue
            if corr.loc[a, b] > threshold:
                drop = b if rank_p.get(a, 1.0) <= rank_p.get(b, 1.0) else a
                keep.discard(drop)
                if drop == a:
                    break
    return [f for f in features if f in keep]


def _cv_score_for_penalty(
    X: pd.DataFrame,
    y_time: pd.Series,
    y_event: pd.Series,
    penalty: float,
    clinical_cols_expanded: list[str],
    n_splits: int,
    seed: int,
) -> tuple[float, float]:
    if len(X) < 2:
        return np.nan, np.nan
    n_splits = min(max(2, n_splits), len(X))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, va in kf.split(X):
        train = X.iloc[tr].copy()
        valid = X.iloc[va].copy()
        train["time"] = y_time.iloc[tr].values
        train["event"] = y_event.iloc[tr].values
        penalizer = np.array([0.0 if c in clinical_cols_expanded else penalty for c in X.columns])
        try:
            cph = CoxPHFitter(penalizer=penalizer, l1_ratio=1.0)
            cph.fit(train, "time", "event", show_progress=False)
            pred = np.dot(valid.values, cph.params_.loc[X.columns].values)
            scores.append(concordance_index(y_time.iloc[va], -pred, y_event.iloc[va]))
        except Exception:
            scores.append(np.nan)
    finite = np.isfinite(scores)
    if not finite.any():
        return np.nan, np.nan
    return float(np.nanmean(scores)), float(np.nanstd(scores) / np.sqrt(finite.sum()))


def fit_lasso_cox(
    df_train: pd.DataFrame,
    endpoint: str,
    time_col: str,
    event_col: str,
    feature_cols: list[str],
    clinical_cols: list[str],
    univariable_p_threshold: float = 0.05,
    correlation_threshold: float = 0.90,
    penalty_grid: list[float] | None = None,
    n_folds: int = 10,
    selection_rule: str = "1se",
    seed: int = 1234,
) -> LassoCoxModel:
    if penalty_grid is None:
        penalty_grid = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    screen = univariable_screen(df_train, time_col, event_col, feature_cols, univariable_p_threshold)
    screened = list(screen["feature"])
    if not screened:
        screened = feature_cols[: min(50, len(feature_cols))]
    reduced = remove_correlated_features(df_train, screened, screen if not screen.empty else pd.DataFrame({"feature": screened, "p": 1.0}), correlation_threshold)
    X, scaler = prepare_matrix(df_train, reduced, clinical_cols, fit_scaler=True)
    y_time = df_train[time_col].reset_index(drop=True)
    y_event = df_train[event_col].reset_index(drop=True)
    X = X.reset_index(drop=True)
    expanded_clinical = [c for c in X.columns if not c in reduced]
    cv_rows = []
    for pen in penalty_grid:
        mean, se = _cv_score_for_penalty(X, y_time, y_event, pen, expanded_clinical, n_folds, seed)
        cv_rows.append({"penalty": pen, "mean_cindex": mean, "se": se})
    cv = pd.DataFrame(cv_rows).sort_values("penalty")
    if cv["mean_cindex"].dropna().empty:
        cv["mean_cindex"] = cv["mean_cindex"].fillna(0.5)
        cv["se"] = cv["se"].fillna(0.0)
    best_idx = cv["mean_cindex"].idxmax()
    if selection_rule.lower() == "1se":
        best = cv.loc[best_idx]
        eligible = cv[cv["mean_cindex"] >= best["mean_cindex"] - best["se"]]
        penalty = float(eligible["penalty"].max())
    else:
        penalty = float(cv.loc[best_idx, "penalty"])
    train_fit = X.copy()
    train_fit["time"] = y_time.values
    train_fit["event"] = y_event.values
    penalizer = np.array([0.0 if c in expanded_clinical else penalty for c in X.columns])
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=1.0)
    cph.fit(train_fit, "time", "event", show_progress=False)
    coefs = cph.params_.loc[X.columns]
    selected = [c for c, v in coefs.items() if abs(float(v)) > 1e-8]
    if not selected:
        selected = list(coefs.abs().sort_values(ascending=False).head(5).index)
    train_sel = X[selected].copy()
    train_sel["time"] = y_time.values
    train_sel["event"] = y_event.values
    cph2 = CoxPHFitter(penalizer=0.001)
    cph2.fit(train_sel, "time", "event")
    scores = np.dot(X[selected].values, cph2.params_.loc[selected].values)
    model = LassoCoxModel(
        endpoint=endpoint,
        time_col=time_col,
        event_col=event_col,
        feature_cols=reduced,
        clinical_cols=clinical_cols,
        selected_cols=selected,
        matrix_cols=list(X.columns),
        scaler=scaler,
        cox_model=cph2,
        penalty=penalty,
        training_median_cutoff=float(np.median(scores)),
    )
    model.cv_table = cv
    model.univariable_screen = screen
    return model


def save_model(model: LassoCoxModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> LassoCoxModel:
    return joblib.load(path)
