from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))

import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from npcpaper.config import load_config, ensure_dir
from npcpaper.deep_features import MRISurvivalDataset, choose_device
from npcpaper.deep_resnet3d import MultiEndpointCoxResNet, cox_partial_likelihood_loss, load_medicalnet_weights
from npcpaper.utils_io import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg['project'].get('seed', 1234))
    clinical = pd.read_csv(cfg['inputs']['clinical_csv'])
    prep_manifest = Path(cfg['mri']['preprocessed_dir']) / 'preprocessed_manifest.csv'
    manifest = pd.read_csv(prep_manifest if prep_manifest.exists() else cfg['inputs']['imaging_manifest_csv'])
    dat = manifest.merge(clinical, on='patient_id', how='inner')
    dat = dat[dat[cfg['clinical']['cohort_col']] == cfg['clinical']['train_label']].reset_index(drop=True)
    mask_col = 'mask_preprocessed_path' if 'mask_preprocessed_path' in dat.columns else cfg['inputs'].get('manual_mask_column', 'mask_path')
    endpoints = {k: cfg['clinical']['endpoints'][k] for k in cfg['deep_learning']['endpoints']}
    ds = MRISurvivalDataset(dat, endpoints=endpoints, sequences=cfg['mri']['sequences'], mask_col=mask_col, input_size=cfg['deep_learning']['input_size'])
    n_val = max(1, int(len(ds) * 0.15))
    n_tr = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(cfg['project'].get('seed', 1234)))
    train_loader = DataLoader(train_ds, batch_size=cfg['deep_learning']['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg['deep_learning']['batch_size'], shuffle=False, num_workers=2)
    device = choose_device(cfg['deep_learning'].get('device', 'auto'))
    model = MultiEndpointCoxResNet(endpoints=list(endpoints.keys()), in_channels=len(cfg['mri']['sequences'])).to(device)
    ckpt_path = cfg['deep_learning'].get('pretrained_medicalnet_path')
    if ckpt_path and Path(ckpt_path).exists():
        load_medicalnet_weights(model, ckpt_path, strict=False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['deep_learning']['learning_rate'], weight_decay=cfg['deep_learning'].get('weight_decay', 1e-5))
    best_val = float('inf')
    patience = cfg['deep_learning'].get('patience', 20)
    stale = 0
    outdir = ensure_dir(cfg['deep_learning']['output_dir'])
    for epoch in range(1, cfg['deep_learning']['max_epochs'] + 1):
        model.train()
        losses = []
        for x, y, pids in train_loader:
            x = x.to(device)
            pred = model(x)
            loss = 0
            for ep in endpoints:
                time = y[f'{ep}_time'].to(device)
                event = y[f'{ep}_event'].to(device)
                loss = loss + cox_partial_likelihood_loss(pred[ep], time, event)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, pids in val_loader:
                x = x.to(device)
                pred = model(x)
                loss = 0
                for ep in endpoints:
                    loss = loss + cox_partial_likelihood_loss(pred[ep], y[f'{ep}_time'].to(device), y[f'{ep}_event'].to(device))
                val_losses.append(float(loss.cpu()))
        val_loss = sum(val_losses) / max(1, len(val_losses))
        if epoch == 1 or epoch == cfg['deep_learning']['max_epochs'] or epoch % 10 == 0:
            print(f'epoch={epoch} status=running')
        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            torch.save({'model_state_dict': model.state_dict()}, outdir / 'resnet3d_multicox_best.pt')
        else:
            stale += 1
        if stale >= patience:
            break
    print('Training complete.')

if __name__ == '__main__':
    main()
