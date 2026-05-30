from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial.distance import cdist


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return mask
    struct = generate_binary_structure(mask.ndim, 1)
    eroded = binary_erosion(mask, structure=struct, border_value=0)
    return mask ^ eroded


def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(a, b).sum() / denom)


def surface_distances(a: np.ndarray, b: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    sa = np.argwhere(_surface(a))
    sb = np.argwhere(_surface(b))
    if sa.size == 0 or sb.size == 0:
        return np.array([np.inf])
    sp = np.array([spacing[2], spacing[1], spacing[0]])
    sa = sa * sp
    sb = sb * sp
    d_ab = cdist(sa, sb).min(axis=1)
    d_ba = cdist(sb, sa).min(axis=1)
    return np.concatenate([d_ab, d_ba])


def hd95(a: np.ndarray, b: np.ndarray, spacing: tuple[float, float, float]) -> float:
    d = surface_distances(a, b, spacing)
    return float(np.percentile(d[np.isfinite(d)], 95)) if np.any(np.isfinite(d)) else np.inf


def asd(a: np.ndarray, b: np.ndarray, spacing: tuple[float, float, float]) -> float:
    d = surface_distances(a, b, spacing)
    return float(np.mean(d[np.isfinite(d)])) if np.any(np.isfinite(d)) else np.inf


def evaluate_pair(reference_path: str | Path, prediction_path: str | Path) -> dict[str, float]:
    ref_img = sitk.ReadImage(str(reference_path))
    pred_img = sitk.ReadImage(str(prediction_path))
    pred_img = sitk.Resample(pred_img, ref_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, pred_img.GetPixelID())
    ref = sitk.GetArrayFromImage(ref_img) > 0
    pred = sitk.GetArrayFromImage(pred_img) > 0
    spacing = ref_img.GetSpacing()
    return {"dice": dice_score(ref, pred), "hd95_mm": hd95(ref, pred, spacing), "asd_mm": asd(ref, pred, spacing)}


def evaluate_manifest(manifest: pd.DataFrame, reference_col: str, prediction_col: str) -> pd.DataFrame:
    rows = []
    for _, row in manifest.iterrows():
        metrics = evaluate_pair(row[reference_col], row[prediction_col])
        rows.append({"patient_id": row["patient_id"], **metrics})
    return pd.DataFrame(rows)
