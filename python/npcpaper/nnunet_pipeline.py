from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import shutil


def make_nnunet_dataset(
    manifest: pd.DataFrame,
    dataset_id: int,
    dataset_name: str,
    nnunet_raw: str | Path,
    sequences: list[str],
    label_col: str = "mask_preprocessed_path",
    image_suffix: str = "_0000.nii.gz",
) -> Path:
    """Create an nnU-Net v2 raw dataset folder from a manifest.

    Sequence order is encoded as channel 0000, 0001, ... according to `sequences`.
    """
    nnunet_raw = Path(nnunet_raw)
    ds_name = f"Dataset{int(dataset_id):03d}_{dataset_name}"
    ds_dir = nnunet_raw / ds_name
    images_tr = ds_dir / "imagesTr"
    labels_tr = ds_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    for _, row in manifest.iterrows():
        pid = str(row["patient_id"])
        for i, seq in enumerate(sequences):
            src_col = f"{seq}_preprocessed_path" if f"{seq}_preprocessed_path" in row else f"{seq}_path"
            src = Path(row[src_col])
            dst = images_tr / f"{pid}_{i:04d}.nii.gz"
            if src.exists() and not dst.exists():
                shutil.copyfile(src, dst)
        lab = Path(row[label_col]) if label_col in row else None
        if lab is not None and lab.exists():
            dst_lab = labels_tr / f"{pid}.nii.gz"
            if not dst_lab.exists():
                shutil.copyfile(lab, dst_lab)
    channel_names = {str(i): seq for i, seq in enumerate(sequences)}
    dataset_json = {
        "channel_names": channel_names,
        "labels": {"background": 0, "primary_tumor": 1},
        "numTraining": int(manifest.shape[0]),
        "file_ending": ".nii.gz",
    }
    with (ds_dir / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2)
    return ds_dir


def run_command(cmd: list[str], env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def nnunet_env(raw: str | Path, preprocessed: str | Path, results: str | Path) -> dict[str, str]:
    env = os.environ.copy()
    env["nnUNet_raw"] = str(raw)
    env["nnUNet_preprocessed"] = str(preprocessed)
    env["nnUNet_results"] = str(results)
    return env


def plan_and_preprocess(dataset_id: int, env: dict[str, str], dry_run: bool = False) -> None:
    run_command(["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "--verify_dataset_integrity"], env=env, dry_run=dry_run)


def train(dataset_id: int, configuration: str, fold: int, trainer: str = "nnUNetTrainer", env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    run_command(["nnUNetv2_train", str(dataset_id), configuration, str(fold), "-tr", trainer], env=env, dry_run=dry_run)


def predict(input_dir: str | Path, output_dir: str | Path, dataset_id: int, configuration: str, folds: list[int], env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    fold_args = [str(f) for f in folds]
    run_command(["nnUNetv2_predict", "-i", str(input_dir), "-o", str(output_dir), "-d", str(dataset_id), "-c", configuration, "-f", *fold_args], env=env, dry_run=dry_run)
