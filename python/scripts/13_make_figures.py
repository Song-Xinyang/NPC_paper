from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))

import argparse
from pathlib import Path
import pandas as pd

from npcpaper.config import load_config, ensure_dir
from npcpaper.figures import plot_threshold_scan, plot_bootstrap_thresholds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    outdir = ensure_dir(Path(cfg['project']['output_dir']) / 'figures')
    threshold_dir = Path(cfg['project']['output_dir']) / 'threshold'
    ep = cfg['clinical'].get('threshold_reference_endpoint', 'dmfs')
    scan_path = threshold_dir / f'threshold_scan_{ep}.csv'
    boot_path = threshold_dir / f'bootstrap_thresholds_{ep}.csv'
    locked = cfg['clinical'].get('locked_ebv_cutoff', 6020.0)
    if scan_path.exists():
        plot_threshold_scan(pd.read_csv(scan_path), locked, outdir / f'Figure_threshold_{ep}.png')
    if boot_path.exists():
        plot_bootstrap_thresholds(pd.read_csv(boot_path), locked, outdir / f'Figure_bootstrap_threshold_{ep}.png')
    print(outdir)

if __name__ == '__main__':
    main()
