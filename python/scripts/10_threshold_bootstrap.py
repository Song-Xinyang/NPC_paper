from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))

import argparse
from pathlib import Path
import pandas as pd

from npcpaper.config import load_config, ensure_dir
from npcpaper.threshold_scan import scan_thresholds, choose_locked_threshold, bootstrap_thresholds, summarize_bootstrap
from npcpaper.figures import plot_threshold_scan, plot_bootstrap_thresholds
from npcpaper.utils_io import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    clinical = pd.read_csv(cfg['inputs']['clinical_csv'])
    c = cfg['clinical']
    cohort_col = c.get('cohort_col')
    if cohort_col in clinical.columns:
        clinical = clinical[clinical[cohort_col].isin([c.get('train_label', 'train'), c.get('internal_label', 'internal')])].copy()
    ep = c.get('threshold_reference_endpoint', 'dmfs')
    cols = c['endpoints'][ep]
    scfg = c['threshold_scan']
    outdir = ensure_dir(Path(cfg['project']['output_dir']) / 'threshold')
    scan = scan_thresholds(
        clinical, cols['time'], cols['event'], c['ebv_col'], c['hbsag_col'], c['covariates'],
        n_grid=scfg.get('n_grid', 300),
        min_fraction_per_hbsag_arm=scfg.get('min_fraction_per_hbsag_arm', 0.10),
        min_events_per_comparison_arm=scfg.get('min_events_per_comparison_arm', 10),
    )
    scan.to_csv(outdir / f'threshold_scan_{ep}.csv', index=False)
    chosen = choose_locked_threshold(scan)
    chosen['paper_locked_cutoff'] = c.get('locked_ebv_cutoff', 6020.0)
    write_json(chosen, outdir / 'chosen_threshold.json')
    plot_threshold_scan(scan, c.get('locked_ebv_cutoff', chosen['locked_cutoff']), outdir / f'threshold_scan_{ep}.png', title=f'{ep.upper()} threshold scan')
    boot = bootstrap_thresholds(
        clinical, cols['time'], cols['event'], c['ebv_col'], c['hbsag_col'], c['covariates'],
        n_bootstrap=scfg.get('bootstrap_iterations', 1000),
        seed=cfg['project'].get('seed', 1234),
        n_grid=scfg.get('n_grid', 300),
        min_fraction_per_hbsag_arm=scfg.get('min_fraction_per_hbsag_arm', 0.10),
        min_events_per_comparison_arm=scfg.get('min_events_per_comparison_arm', 10),
    )
    boot.to_csv(outdir / f'bootstrap_thresholds_{ep}.csv', index=False)
    summary = summarize_bootstrap(boot, c.get('locked_ebv_cutoff', chosen['locked_cutoff']), scfg.get('tolerance_fraction', 0.10))
    write_json(summary, outdir / 'bootstrap_threshold_summary.json')
    plot_bootstrap_thresholds(boot, c.get('locked_ebv_cutoff', chosen['locked_cutoff']), outdir / f'bootstrap_thresholds_{ep}.png')
    print(chosen)
    print(summary)

if __name__ == '__main__':
    main()
