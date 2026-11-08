from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))

import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, random_split

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
    dat = manifest.merge(clinical, on='patient_id', how='inner', validate='one_to_one')
    dat = dat[dat[cfg['clinical']['cohort_col']] == cfg['clinical']['train_label']].reset_index(drop=True)
    mask_col = 'mask_preprocessed_path' if 'mask_preprocessed_path' in dat.columns else cfg['inputs'].get('manual_mask_column', 'mask_path')
    endpoints = {k: cfg['clinical']['endpoints'][k] for k in cfg['deep_learning']['endpoints']}
    ds = MRISurvivalDataset(dat, endpoints=endpoints, sequences=cfg['mri']['sequences'], mask_col=mask_col, input_size=cfg['deep_learning']['input_size'])
    split_column = cfg['deep_learning'].get('inner_split_column')
    if split_column and split_column in dat:
        split = dat[split_column]
        if not split.isin(['gradient', 'validation']).all() or split.nunique() != 2:
            raise ValueError(f"Every development row in {split_column} must belong to gradient or validation; both subsets are required.")
        train_ds = Subset(ds, dat.index[split.eq('gradient')].tolist())
        val_ds = Subset(ds, dat.index[split.eq('validation')].tolist())
    else:
        fraction = cfg['deep_learning'].get('inner_validation_fraction', 1 / 7)
        n_val = max(1, round(len(ds) * fraction))
        if not 0 < fraction < 1 or n_val >= len(ds):
            raise ValueError("The inner split needs at least one development patient in each subset.")
        print('Creating a new seeded inner split; historical patient assignments were not supplied.')
        train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(cfg['project'].get('seed', 1234)))
    train_loader = DataLoader(train_ds, batch_size=cfg['deep_learning']['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg['deep_learning']['batch_size'], shuffle=False, num_workers=2)
    device = choose_device(cfg['deep_learning'].get('device', 'auto'))
    model = MultiEndpointCoxResNet(endpoints=list(endpoints.keys()), in_channels=len(cfg['mri']['sequences'])).to(device)
    ckpt_path = cfg['deep_learning'].get('pretrained_medicalnet_path')
    if ckpt_path:
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"Configured MedicalNet weights are missing: {ckpt_path}")
        load_medicalnet_weights(model, ckpt_path, strict=False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['deep_learning']['learning_rate'], weight_decay=cfg['deep_learning'].get('weight_decay', 1e-5))
    best_val = float('inf')
    patience = cfg['deep_learning'].get('patience', 10)
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
        validation = {ep: {'risk': [], 'time': [], 'event': []} for ep in endpoints}
        with torch.no_grad():
            for x, y, pids in val_loader:
                x = x.to(device)
                pred = model(x)
                for ep in endpoints:
                    validation[ep]['risk'].append(pred[ep].cpu())
                    validation[ep]['time'].append(y[f'{ep}_time'].cpu())
                    validation[ep]['event'].append(y[f'{ep}_event'].cpu())
            val_loss = sum(float(cox_partial_likelihood_loss(
                torch.cat(part['risk']), torch.cat(part['time']), torch.cat(part['event'])
            )) for part in validation.values())
        if epoch == 1 or epoch == cfg['deep_learning']['max_epochs'] or epoch % 10 == 0:
            print(f'epoch={epoch} status=running')
        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'validation_loss': val_loss},
                       outdir / 'resnet3d_multicox_best.pt')
        else:
            stale += 1
        if stale >= patience:
            break
    print('Training complete.')

if __name__ == '__main__':
    main()
