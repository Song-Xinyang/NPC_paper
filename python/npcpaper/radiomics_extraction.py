from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_extractor(bin_width: int = 25, filters: dict | None = None):
    from radiomics import featureextractor
    settings = {
        "binWidth": bin_width,
        "interpolator": "sitkBSpline",
        "resampledPixelSpacing": None,
        "normalize": False,
        "removeOutliers": None,
        "geometryTolerance": 1e-4,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    if filters is None:
        filters = {}
    if filters.get("exponential", True):
        extractor.enableImageTypeByName("Exponential")
    if filters.get("gradient", True):
        extractor.enableImageTypeByName("Gradient")
    if filters.get("logarithm", True):
        extractor.enableImageTypeByName("Logarithm")
    if filters.get("square", True):
        extractor.enableImageTypeByName("Square")
    if filters.get("squareroot", True):
        extractor.enableImageTypeByName("SquareRoot")
    if filters.get("wavelet", True):
        extractor.enableImageTypeByName("Wavelet")
    log_cfg = filters.get("log", {"sigma": [1.0, 3.0, 5.0]})
    if log_cfg:
        extractor.enableImageTypeByName("LoG", customArgs={"sigma": log_cfg.get("sigma", [1.0, 3.0, 5.0])})
    extractor.enableAllFeatures()
    return extractor


def extract_for_patient(row: pd.Series, sequences: list[str], mask_col: str, extractor, compute_shape_per_sequence: bool = True) -> dict[str, float | str]:
    features: dict[str, float | str] = {"patient_id": str(row["patient_id"])}
    shape_done = False
    for seq in sequences:
        img_col = f"{seq}_preprocessed_path" if f"{seq}_preprocessed_path" in row else f"{seq}_path"
        image_path = row[img_col]
        mask_path = row[mask_col]
        result = extractor.execute(str(image_path), str(mask_path))
        for key, value in result.items():
            if key.startswith("diagnostics"):
                continue
            clean_key = key.replace("original_", "original_")
            if (not compute_shape_per_sequence) and "shape" in clean_key.lower():
                if shape_done:
                    continue
                prefix = "shape"
            else:
                prefix = seq
            try:
                features[f"{prefix}_{clean_key}"] = float(value)
            except Exception:
                pass
        shape_done = True
    return features


def extract_manifest(manifest: pd.DataFrame, sequences: list[str], mask_col: str, out_csv: str | Path, bin_width: int = 25, filters: dict | None = None, compute_shape_per_sequence: bool = True) -> pd.DataFrame:
    extractor = make_extractor(bin_width=bin_width, filters=filters)
    rows = []
    for _, row in manifest.iterrows():
        rows.append(extract_for_patient(row, sequences, mask_col, extractor, compute_shape_per_sequence))
    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def compute_icc_filter(feature_a: pd.DataFrame, feature_b: pd.DataFrame, id_col: str = "patient_id", threshold: float = 0.80) -> pd.DataFrame:
    """Compute a two-way consistency-style ICC approximation for two segmentations.

    This is intended as a transparent robustness screen. For regulatory analyses,
    confirm with a dedicated ICC package such as pingouin or irr.
    """
    merged = feature_a.merge(feature_b, on=id_col, suffixes=("_a", "_b"))
    rows = []
    for base in sorted({c[:-2] for c in merged.columns if c.endswith("_a")}):
        a = merged[f"{base}_a"].astype(float).values
        b = merged[f"{base}_b"].astype(float).values
        x = pd.DataFrame({"a": a, "b": b}).dropna()
        if x.shape[0] < 3:
            continue
        n = x.shape[0]
        k = 2
        grand = x.values.mean()
        mean_subject = x.mean(axis=1).values
        mean_rater = x.mean(axis=0).values
        ss_subject = k * np.sum((mean_subject - grand) ** 2)
        ss_error = np.sum((x.values - mean_subject[:, None] - mean_rater[None, :] + grand) ** 2)
        ms_subject = ss_subject / (n - 1)
        ms_error = ss_error / ((n - 1) * (k - 1))
        icc = (ms_subject - ms_error) / (ms_subject + (k - 1) * ms_error) if (ms_subject + (k - 1) * ms_error) != 0 else np.nan
        rows.append({"feature": base, "icc": float(icc), "keep": bool(icc >= threshold)})
    return pd.DataFrame(rows)
