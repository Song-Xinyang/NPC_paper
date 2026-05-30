from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=index)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(path, index=index)
    else:
        df.to_csv(path, index=index)


def write_json(obj: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_feature_tables(clinical: pd.DataFrame, feature_paths: Iterable[str | Path], id_col: str) -> pd.DataFrame:
    out = clinical.copy()
    for p in feature_paths:
        if p is None:
            continue
        p = Path(p)
        if not p.exists():
            continue
        feats = read_table(p)
        if id_col not in feats.columns:
            raise ValueError(f"{id_col!r} not present in {p}")
        out = out.merge(feats, on=id_col, how="left")
    return out


def infer_feature_columns(df: pd.DataFrame, prefixes: tuple[str, ...] = ("Rad", "DL")) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if any(c.startswith(prefix) for prefix in prefixes):
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
    return cols
