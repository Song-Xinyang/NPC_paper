from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))

import argparse
from pathlib import Path
import pandas as pd

from npcpaper.config import load_config, ensure_dir
from npcpaper.utils_io import load_feature_tables, infer_feature_columns
from npcpaper.cox_lasso import fit_lasso_cox, save_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    c = cfg['clinical']
    m = cfg['modeling']
    clinical = pd.read_csv(cfg['inputs']['clinical_csv'])
    feature_paths = [cfg['inputs'].get('radiomics_csv'), cfg['inputs'].get('deep_features_csv')]
    df = load_feature_tables(clinical, feature_paths, id_col=c.get('patient_id_col', 'patient_id'))
    feature_cols = infer_feature_columns(df, prefixes=('Rad', 'DL', 't1', 't2', 't1c'))
    if not feature_cols:
        raise ValueError('No imaging feature columns found. Expected prefixes Rad, DL, t1, t2 or t1c.')
    train = df[df[c['cohort_col']] == c['train_label']].copy()
    outdir = ensure_dir(m['output_dir'])
    predictions = df[[c.get('patient_id_col', 'patient_id'), c['cohort_col']]].copy()
    for ep in m.get('endpoint_order', ['dmfs', 'os', 'pfs']):
        cols = c['endpoints'][ep]
        model = fit_lasso_cox(
            train,
            endpoint=ep,
            time_col=cols['time'],
            event_col=cols['event'],
            feature_cols=feature_cols,
            clinical_cols=c['covariates'],
            univariable_p_threshold=m.get('univariable_p_threshold', 0.05),
            correlation_threshold=m.get('correlation_threshold', 0.90),
            penalty_grid=m.get('lasso_penalty_grid'),
            n_folds=m.get('lasso_cv_folds', 10),
            selection_rule=m.get('lasso_selection_rule', '1se'),
            seed=cfg['project'].get('seed', 1234),
        )
        save_model(model, outdir / f'{ep}_lasso_cox.joblib')
        pd.DataFrame({'selected_feature': model.selected_cols, 'coef': model.cox_model.params_.loc[model.selected_cols].values}).to_csv(outdir / f'{ep}_selected_features.csv', index=False)
        model.cv_table.to_csv(outdir / f'{ep}_lasso_cv.csv', index=False)
        scores = model.predict_score(df)
        predictions[f'score_{ep}'] = scores.values
    for ep in m.get('endpoint_order', ['dmfs', 'os', 'pfs']):
        predictions[c['endpoints'][ep]['time']] = df[c['endpoints'][ep]['time']].values
        predictions[c['endpoints'][ep]['event']] = df[c['endpoints'][ep]['event']].values
    predictions.to_csv(outdir / 'endpoint_scores.csv', index=False)
    print(outdir / 'endpoint_scores.csv')

if __name__ == '__main__':
    main()
