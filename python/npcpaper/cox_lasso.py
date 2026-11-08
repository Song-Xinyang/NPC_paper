from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from sklearn.model_selection import StratifiedKFold
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
    category_levels: dict | None = None

    def predict_score(self, df: pd.DataFrame) -> pd.Series:
        if getattr(self, "category_levels", None) is not None:
            valid = df[self.clinical_cols + self.feature_cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
            valid &= supported_categories(df, self.category_levels)
            scores = pd.Series(np.nan, index=df.index, name=f"score_{self.endpoint}")
            if valid.any():
                raw = encode_matrix(df.loc[valid], self.feature_cols, self.clinical_cols, self.category_levels)
                scaled = self.scaler.transform(raw[self.matrix_cols])
                design = pd.DataFrame(scaled, index=raw.index, columns=self.matrix_cols)
                scores.loc[valid] = design[self.selected_cols] @ self.cox_model.params_.loc[self.selected_cols]
            return scores
        # Keep the input representation of previously serialized models.
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
    return res[res["p"] < p_threshold]


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


def supported_categories(df, levels):
    """A coefficient cannot be estimated for a category absent from training."""
    supported = pd.Series(True, index=df.index)
    for col, categories in levels.items():
        supported &= df[col].isin(categories)
    return supported


def encode_matrix(df, features, clinical, levels):
    """Apply the training fold's category levels and reference categories."""
    data = df[features + clinical].copy()
    for col, categories in levels.items():
        data[col] = pd.Categorical(data[col], categories=categories)
    return pd.get_dummies(data, columns=list(levels), drop_first=True, dtype=float).astype(float)


def _fit_design(train, time_col, event_col, features, clinical, categorical, p_threshold, corr_threshold):
    screen = univariable_screen(train, time_col, event_col, features, p_threshold)
    reduced = remove_correlated_features(train, list(screen["feature"]), screen, corr_threshold)
    levels = {col: pd.Categorical(train[col]).categories.tolist() for col in clinical
              if col in categorical or not pd.api.types.is_numeric_dtype(train[col])}
    raw = encode_matrix(train, reduced, clinical, levels)
    raw = raw.loc[:, raw.nunique() > 1]
    if raw.empty:
        raise ValueError("No estimable clinical predictors or screened imaging features remain.")
    scaler = StandardScaler().fit(raw)
    design = pd.DataFrame(scaler.transform(raw), index=train.index, columns=raw.columns)
    design[time_col] = train[time_col]
    design[event_col] = train[event_col]
    return design, reduced, levels, scaler, screen


def _fit_penalized(frame, time_col, event_col, features, penalty):
    # Prespecified clinical terms have zero L1 penalty.
    cols = frame.columns.drop([time_col, event_col])
    weights = np.where(cols.isin(features), penalty, 0.0)
    return CoxPHFitter(penalizer=weights, l1_ratio=1.0).fit(frame, time_col, event_col)


def _select_penalty(train, time_col, event_col, features, clinical, categorical,
                    p_threshold, corr_threshold, penalties, n_folds, rule, seed):
    event = train[event_col]
    n_splits = min(n_folds, int(event.sum()), len(train))
    if n_splits < 2:
        raise ValueError("Cross-validation requires at least two folds containing events.")
    split = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    losses = {penalty: [] for penalty in penalties}
    unsupported_rows = []
    for train_idx, valid_idx in split.split(train, event):
        training, held_out = train.iloc[train_idx], train.iloc[valid_idx]
        design, reduced, levels, scaler, _ = _fit_design(
            training, time_col, event_col, features, clinical, categorical, p_threshold, corr_threshold
        )
        # Never encode a previously unseen disease stage as the training reference.
        supported = supported_categories(held_out, levels)
        unsupported_rows.append(int((~supported).sum()))
        held_out = held_out.loc[supported]
        if not held_out[event_col].any():
            raise ValueError("A validation fold has no events with estimable clinical categories. "
                             "Review rare levels and the fold count.")
        # Apply the fold's feature selection, coding and scale before computing held-out loss.
        cols = list(scaler.feature_names_in_)
        raw = encode_matrix(held_out, reduced, clinical, levels)
        valid = pd.DataFrame(scaler.transform(raw[cols]), index=held_out.index, columns=cols)
        valid[[time_col, event_col]] = held_out[[time_col, event_col]]
        joined = pd.concat([design, valid], ignore_index=True)
        for penalty, fold_losses in losses.items():
            try:
                cph = _fit_penalized(design, time_col, event_col, reduced, penalty)
                # Grouped CV uses risk sets from the full cohort: subtract the training
                # partial log-likelihood evaluated at the very same fold coefficients.
                full_ll = len(joined) * cph.score(joined, scoring_method="log_likelihood")
                train_ll = len(design) * cph.score(design, scoring_method="log_likelihood")
                fold_losses.append(-2.0 * (full_ll - train_ll) / valid[event_col].sum())
            except (ConvergenceError, ValueError, np.linalg.LinAlgError, FloatingPointError):
                fold_losses.append(np.nan)
    rows = []
    for penalty, fold_losses in losses.items():
        valid_folds = int(np.isfinite(fold_losses).sum())
        complete = valid_folds == n_splits
        rows.append({"penalty": penalty, "mean_deviance": np.mean(fold_losses) if complete else np.nan,
                     "se": np.std(fold_losses, ddof=1) / np.sqrt(n_splits) if complete else np.nan,
                     "valid_folds": valid_folds, "total_folds": n_splits,
                     "cv_rows_with_unseen_categories": sum(unsupported_rows),
                     **{f"fold_{i}_deviance": loss for i, loss in enumerate(fold_losses, 1)}})
    cv = pd.DataFrame(rows).sort_values("penalty")
    if not cv["mean_deviance"].notna().any():
        raise ValueError("No penalty produced finite partial-likelihood deviance in every fold.")
    best = cv.loc[cv["mean_deviance"].idxmin()]
    if rule == "1se":
        eligible = cv["mean_deviance"] <= best["mean_deviance"] + best["se"]
        penalty = cv.loc[eligible, "penalty"].max()
    else:
        penalty = best["penalty"]
    cv["selected"] = cv["penalty"] == penalty
    return float(penalty), cv


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
    categorical_clinical_cols: list[str] | None = None,
    final_ridge_penalty: float = 0.001,
) -> LassoCoxModel:
    penalties = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0] if penalty_grid is None else penalty_grid
    if not len(penalties) or not np.isfinite(penalties).all() or min(penalties) < 0:
        raise ValueError("The Cox penalty grid must contain finite, nonnegative values.")
    if selection_rule.lower() not in ("min", "1se"):
        raise ValueError("selection_rule must be 'min' or '1se'")
    categorical = set(categorical_clinical_cols or ())
    if not categorical.issubset(clinical_cols):
        raise ValueError("Categorical terms must also be included in the clinical covariates.")
    required = [time_col, event_col, *clinical_cols]
    train = df_train.replace([np.inf, -np.inf], np.nan).dropna(subset=required).reset_index(drop=True)
    if not train[event_col].isin([0, 1]).all() or (train[time_col] < 0).any():
        raise ValueError("Cox inputs require binary event indicators and nonnegative event times.")
    if train[feature_cols].isna().any().any():
        raise ValueError("The imaging cohort requires complete candidate feature measurements.")
    penalty, cv = _select_penalty(
        train, time_col, event_col, feature_cols, clinical_cols, categorical, univariable_p_threshold,
        correlation_threshold, penalties, n_folds, selection_rule.lower(), seed
    )
    design, reduced, levels, scaler, screen = _fit_design(
        train, time_col, event_col, feature_cols, clinical_cols, categorical,
        univariable_p_threshold, correlation_threshold
    )
    selection_fit = _fit_penalized(design, time_col, event_col, reduced, penalty)
    # lifelines smooths the L1 penalty. Treat numerical residue as zero on the standardized scale.
    selected = [col for col, beta in selection_fit.params_.items() if col not in reduced or abs(beta) > 1e-6]
    if not selected:
        raise ValueError("The selected penalty retained no estimable coefficients.")
    post_selection = CoxPHFitter(penalizer=final_ridge_penalty).fit(design[selected + [time_col, event_col]], time_col, event_col)
    scores = design[selected] @ post_selection.params_.loc[selected]
    model = LassoCoxModel(
        endpoint=endpoint, time_col=time_col, event_col=event_col,
        feature_cols=reduced, clinical_cols=clinical_cols, selected_cols=selected,
        matrix_cols=list(scaler.feature_names_in_), scaler=scaler, cox_model=post_selection,
        penalty=penalty, training_median_cutoff=float(np.median(scores)), category_levels=levels,
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
