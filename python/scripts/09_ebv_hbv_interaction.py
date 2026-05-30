from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))

import argparse
from pathlib import Path
import pandas as pd

from npcpaper.config import load_config, ensure_dir
from npcpaper.survival_interaction import run_interaction_analysis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    clinical = pd.read_csv(cfg['inputs']['clinical_csv'])
    c = cfg['clinical']
    if c.get('cohort_col') in clinical.columns:
        clinical = clinical[clinical[c['cohort_col']].isin([c.get('train_label', 'train'), c.get('internal_label', 'internal')])].copy()
    outdir = ensure_dir(Path(cfg['project']['output_dir']) / 'interaction')
    res = run_interaction_analysis(
        clinical=clinical,
        endpoints=c['endpoints'],
        ebv_col=c['ebv_col'],
        hbsag_col=c['hbsag_col'],
        covariates=c['covariates'],
        outdir=outdir,
    )
    print(res)

if __name__ == '__main__':
    main()
