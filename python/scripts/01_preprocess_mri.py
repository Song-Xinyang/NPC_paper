from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))

import argparse
import pandas as pd

from npcpaper.config import load_config, ensure_dir
from npcpaper.preprocessing import preprocess_patient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    manifest = pd.read_csv(cfg['inputs']['imaging_manifest_csv'])
    outdir = ensure_dir(cfg['mri']['preprocessed_dir'])
    rows = []
    for _, row in manifest.iterrows():
        rows.append(preprocess_patient(
            row,
            sequences=cfg['mri']['sequences'],
            reference_sequence=cfg['mri']['reference_sequence'],
            outdir=outdir,
            spacing=cfg['mri']['spacing'],
            do_n4=cfg['mri'].get('n4_bias_correction', True),
        ))
    prep = pd.DataFrame(rows)
    merged = manifest.merge(prep, on='patient_id', how='left')
    out_csv = outdir / 'preprocessed_manifest.csv'
    merged.to_csv(out_csv, index=False)
    print(out_csv)

if __name__ == '__main__':
    main()
