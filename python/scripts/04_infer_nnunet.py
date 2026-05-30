from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))

import argparse
from pathlib import Path

from npcpaper.config import load_config, ensure_dir
from npcpaper.nnunet_pipeline import nnunet_env, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    cfg = load_config(args.config)
    outdir = args.output_dir or str(Path(cfg['project']['output_dir']) / 'nnunet_predictions')
    ensure_dir(outdir)
    env = nnunet_env(cfg['nnunet']['nnunet_raw'], cfg['nnunet']['nnunet_preprocessed'], cfg['nnunet']['nnunet_results'])
    predict(args.input_dir, outdir, cfg['nnunet']['dataset_id'], cfg['nnunet']['configuration'], cfg['nnunet']['folds'], env=env, dry_run=args.dry_run)

if __name__ == '__main__':
    main()
