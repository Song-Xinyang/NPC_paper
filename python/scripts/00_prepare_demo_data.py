from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))

import argparse
import numpy as np
import pandas as pd

from npcpaper.utils_io import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='data/demo')
    ap.add_argument('--n', type=int, default=90)
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    n = args.n
    patient_id = [f'DEMO_{i:04d}' for i in range(n)]
    cohort = np.array(['train'] * int(n*0.55) + ['internal'] * int(n*0.25) + ['external'] * (n - int(n*0.55) - int(n*0.25)))
    rng.shuffle(cohort)
    hbsag = rng.binomial(1, 0.18, n)
    ebv_dna = np.exp(rng.normal(7.7, 1.2, n))
    age = rng.normal(46, 11, n)
    sex = rng.binomial(1, 0.55, n)
    ajcc_stage = rng.choice([2,3,4], n, p=[0.15,0.55,0.30])
    chemotherapy = rng.binomial(1, 0.65, n)
    induction = rng.binomial(1, 0.45, n)
    logebv = np.log10(ebv_dna + 1)
    base_lp = 0.22*(age-46)/10 + 0.35*(ajcc_stage-2) + 0.25*sex + 0.35*logebv + 0.22*hbsag*logebv
    def sim_endpoint(scale, extra=0.0):
        rate = np.exp(base_lp + extra) / scale
        t = rng.exponential(1/rate)
        censor = rng.uniform(24, 96, n)
        return np.minimum(t, censor), (t <= censor).astype(int)
    dmfs_t, dmfs_e = sim_endpoint(220)
    pfs_t, pfs_e = sim_endpoint(160)
    os_t, os_e = sim_endpoint(260)
    lrrfs_t, lrrfs_e = sim_endpoint(300, extra=-0.3)
    clinical = pd.DataFrame({
        'patient_id': patient_id, 'cohort': cohort, 'age': age, 'sex': sex,
        'ajcc_stage': ajcc_stage, 'chemotherapy': chemotherapy, 'induction_chemotherapy': induction,
        'hbsag': hbsag, 'ebv_dna': ebv_dna,
        'dmfs_time': dmfs_t, 'dmfs_event': dmfs_e,
        'pfs_time': pfs_t, 'pfs_event': pfs_e,
        'os_time': os_t, 'os_event': os_e,
        'lrrfs_time': lrrfs_t, 'lrrfs_event': lrrfs_e,
    })
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    clinical.to_csv(out / 'clinical_demo.csv', index=False)
    rad = pd.DataFrame({'patient_id': patient_id})
    deep = pd.DataFrame({'patient_id': patient_id})
    for i in range(1, 5):
        rad[f'Rad{i}'] = 0.4*base_lp + rng.normal(0, 1, n)
    for i in range(1, 9):
        deep[f'DL{i}'] = 0.55*base_lp + rng.normal(0, 1, n)
    rad.to_csv(out / 'radiomics_demo.csv', index=False)
    deep.to_csv(out / 'deep_features_demo.csv', index=False)
    print(f'Wrote demo data to {out}')

if __name__ == '__main__':
    main()
