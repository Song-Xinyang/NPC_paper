from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset


def _read_array(path: str | Path) -> tuple[np.ndarray, sitk.Image]:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return arr, img


def crop_and_resize_modalities(row: pd.Series, sequences: list[str], mask_col: str, input_size: tuple[int, int, int] = (128, 128, 128), margin: int = 8) -> np.ndarray:
    mask_arr, _ = _read_array(row[mask_col])
    coords = np.argwhere(mask_arr > 0)
    if coords.size == 0:
        raise ValueError("Empty mask for one patient.")
    z0, y0, x0 = np.maximum(coords.min(axis=0) - margin, 0)
    z1, y1, x1 = np.minimum(coords.max(axis=0) + margin + 1, mask_arr.shape)
    channels = []
    for seq in sequences:
        col = f"{seq}_preprocessed_path" if f"{seq}_preprocessed_path" in row else f"{seq}_path"
        arr, _ = _read_array(row[col])
        crop = arr[z0:z1, y0:y1, x0:x1]
        factors = [input_size[i] / crop.shape[i] for i in range(3)]
        resized = zoom(crop, factors, order=1)
        mean = float(np.mean(resized))
        std = float(np.std(resized)) or 1.0
        channels.append((resized - mean) / std)
    return np.stack(channels, axis=0).astype(np.float32)


class MRISurvivalDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, endpoints: dict[str, dict[str, str]], sequences: list[str], mask_col: str, input_size=(128, 128, 128)):
        self.manifest = manifest.reset_index(drop=True)
        self.endpoints = endpoints
        self.sequences = sequences
        self.mask_col = mask_col
        self.input_size = tuple(input_size)

    def __len__(self) -> int:
        return self.manifest.shape[0]

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        x = crop_and_resize_modalities(row, self.sequences, self.mask_col, self.input_size)
        y = {}
        for endpoint, cols in self.endpoints.items():
            y[f"{endpoint}_time"] = np.float32(row[cols["time"]])
            y[f"{endpoint}_event"] = np.float32(row[cols["event"]])
        return torch.from_numpy(x), y, str(row["patient_id"])


def choose_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def extract_feature_table(model, loader, device: torch.device, endpoints: list[str] | None = None) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for x, y, pids in loader:
            x = x.to(device)
            feats = model.extract_features(x).cpu().numpy()
            for pid, f in zip(pids, feats):
                row = {"patient_id": pid}
                row.update({f"DL{i+1}": float(v) for i, v in enumerate(f)})
                rows.append(row)
    return pd.DataFrame(rows)
