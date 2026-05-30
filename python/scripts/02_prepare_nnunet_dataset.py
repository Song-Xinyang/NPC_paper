from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))

import argparse
from pathlib import Path
import pandas as pd

from npcpaper.config import load_config
from npcpaper.nnunet_pipeline import make_nnunet_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    mri_dir = Path(cfg['mri']['preprocessed_dir'])
    manifest_path = mri_dir / 'preprocessed_manifest.csv'
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
    else:
        manifest = pd.read_csv(cfg['inputs']['imaging_manifest_csv'])
    ds = make_nnunet_dataset(
        manifest,
        dataset_id=cfg['nnunet']['dataset_id'],
        dataset_name=cfg['nnunet']['dataset_name'],
        nnunet_raw=cfg['nnunet']['nnunet_raw'],
        sequences=cfg['mri']['sequences'],
        label_col='mask_preprocessed_path' if 'mask_preprocessed_path' in manifest.columns else cfg['inputs'].get('manual_mask_column', 'mask_path'),
    )
    print(ds)

if __name__ == '__main__':
    main()
