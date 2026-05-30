from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))

import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from npcpaper.config import load_config, ensure_dir
from npcpaper.deep_features import MRISurvivalDataset, choose_device, extract_feature_table
from npcpaper.deep_resnet3d import MultiEndpointCoxResNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    clinical = pd.read_csv(cfg['inputs']['clinical_csv'])
    prep_manifest = Path(cfg['mri']['preprocessed_dir']) / 'preprocessed_manifest.csv'
    manifest = pd.read_csv(prep_manifest if prep_manifest.exists() else cfg['inputs']['imaging_manifest_csv'])
    dat = manifest.merge(clinical, on='patient_id', how='inner')
    mask_col = 'mask_preprocessed_path' if 'mask_preprocessed_path' in dat.columns else cfg['inputs'].get('manual_mask_column', 'mask_path')
    endpoints = {k: cfg['clinical']['endpoints'][k] for k in cfg['deep_learning']['endpoints']}
    ds = MRISurvivalDataset(dat, endpoints=endpoints, sequences=cfg['mri']['sequences'], mask_col=mask_col, input_size=cfg['deep_learning']['input_size'])
    loader = DataLoader(ds, batch_size=cfg['deep_learning']['batch_size'], shuffle=False, num_workers=2)
    device = choose_device(cfg['deep_learning'].get('device', 'auto'))
    model = MultiEndpointCoxResNet(endpoints=list(endpoints.keys()), in_channels=len(cfg['mri']['sequences'])).to(device)
    ckpt = args.checkpoint or str(Path(cfg['deep_learning']['output_dir']) / 'resnet3d_multicox_best.pt')
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state.get('model_state_dict', state))
    feats = extract_feature_table(model, loader, device)
    outdir = ensure_dir(cfg['deep_learning']['output_dir'])
    out_csv = outdir / 'deep_features.csv'
    feats.to_csv(out_csv, index=False)
    print('Deep feature extraction complete.')

if __name__ == '__main__':
    main()
