from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))

import argparse
from pathlib import Path
import pandas as pd

from npcpaper.config import load_config, ensure_dir
from npcpaper.model_evaluation import evaluate_scores, assign_risk_group
from npcpaper.figures import plot_km


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    c = cfg['clinical']
    m = cfg['modeling']
    outdir = ensure_dir(m['output_dir'])
    scores = pd.read_csv(outdir / 'endpoint_scores.csv')
    score_cols = {ep: f'score_{ep}' for ep in m.get('endpoint_order', ['dmfs', 'os', 'pfs'])}
    res = evaluate_scores(
        scores,
        endpoints={ep: c['endpoints'][ep] for ep in score_cols},
        score_cols=score_cols,
        cohort_col=c['cohort_col'],
        train_label=c['train_label'],
        outdir=outdir,
        n_bootstrap=m.get('bootstrap_iterations', 1000),
        seed=cfg['project'].get('seed', 1234),
    )
    print('Model evaluation complete.')
    for ep, score_col in score_cols.items():
        train = scores[scores[c['cohort_col']] == c['train_label']]
        cutoff = train[score_col].median()
        tmp = assign_risk_group(scores, score_col, cutoff)
        for cohort, sub in tmp.groupby(c['cohort_col']):
            plot_km(sub, c['endpoints'][ep]['time'], c['endpoints'][ep]['event'], 'risk_group', outdir / f'KM_{ep}_{cohort}.png', title=f'{ep.upper()} {cohort}')

if __name__ == '__main__':
    main()
