from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))

import argparse

from npcpaper.config import load_config
from npcpaper.nnunet_pipeline import nnunet_env, plan_and_preprocess, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    cfg = load_config(args.config)
    env = nnunet_env(cfg['nnunet']['nnunet_raw'], cfg['nnunet']['nnunet_preprocessed'], cfg['nnunet']['nnunet_results'])
    plan_and_preprocess(cfg['nnunet']['dataset_id'], env, dry_run=args.dry_run)
    for fold in cfg['nnunet']['folds']:
        train(cfg['nnunet']['dataset_id'], cfg['nnunet']['configuration'], fold, cfg['nnunet'].get('trainer', 'nnUNetTrainer'), env=env, dry_run=args.dry_run)

if __name__ == '__main__':
    main()
