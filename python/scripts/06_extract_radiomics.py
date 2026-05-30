from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))

import argparse
from pathlib import Path
import pandas as pd

from npcpaper.config import load_config, ensure_dir
from npcpaper.radiomics_extraction import extract_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    prep_manifest = Path(cfg['mri']['preprocessed_dir']) / 'preprocessed_manifest.csv'
    manifest = pd.read_csv(prep_manifest if prep_manifest.exists() else cfg['inputs']['imaging_manifest_csv'])
    outdir = ensure_dir(cfg['radiomics']['output_dir'])
    mask_col = 'mask_preprocessed_path' if 'mask_preprocessed_path' in manifest.columns else cfg['inputs'].get('manual_mask_column', 'mask_path')
    out_csv = outdir / 'radiomics_features.csv'
    extract_manifest(
        manifest,
        sequences=cfg['mri']['sequences'],
        mask_col=mask_col,
        out_csv=out_csv,
        bin_width=cfg['radiomics'].get('bin_width', 25),
        filters=cfg['radiomics'].get('filters', {}),
        compute_shape_per_sequence=cfg['radiomics'].get('compute_shape_per_sequence', True),
    )
    print(out_csv)

if __name__ == '__main__':
    main()
