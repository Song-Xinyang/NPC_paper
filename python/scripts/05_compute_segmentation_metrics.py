from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))

import argparse
from pathlib import Path
import pandas as pd

from npcpaper.config import load_config, ensure_dir
from npcpaper.segmentation_metrics import evaluate_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    manifest = pd.read_csv(cfg['inputs']['imaging_manifest_csv'])
    ref_col = cfg['inputs'].get('manual_mask_column', 'mask_path')
    pred_col = cfg['inputs'].get('predicted_mask_column', 'pred_mask_path')
    if pred_col not in manifest.columns:
        raise ValueError(f'{pred_col} not found in manifest; add predicted mask paths first.')
    metrics = evaluate_manifest(manifest, reference_col=ref_col, prediction_col=pred_col)
    outdir = ensure_dir(Path(cfg['project']['output_dir']) / 'segmentation')
    metrics.to_csv(outdir / 'segmentation_metrics.csv', index=False)
    print(metrics.describe())

if __name__ == '__main__':
    main()
