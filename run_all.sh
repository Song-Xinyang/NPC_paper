#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-demo}"
PY_CONFIG="${2:-config/config.yaml}"
R_CONFIG="${3:-config/scRNA_config.yaml}"

mkdir -p results logs

if [[ "$MODE" == "demo" ]]; then
  echo "[demo] Generating synthetic clinical and feature data..."
  python python/scripts/00_prepare_demo_data.py --outdir data/demo
  echo "[demo] Running EBV-HBV interaction and threshold scan..."
  python python/scripts/09_ebv_hbv_interaction.py --config config/demo_config.yaml
  python python/scripts/10_threshold_bootstrap.py --config config/demo_config.yaml
  echo "[demo] Running LASSO Cox prognostic models..."
  python python/scripts/11_lasso_cox_models.py --config config/demo_config.yaml
  python python/scripts/12_evaluate_models.py --config config/demo_config.yaml
  python python/scripts/14_verify_demo.py
  echo "[demo] Done."
elif [[ "$MODE" == "full" ]]; then
  echo "[full] Running MRI preprocessing..."
  python python/scripts/01_preprocess_mri.py --config "$PY_CONFIG"
  echo "[full] Preparing nnU-Net dataset..."
  python python/scripts/02_prepare_nnunet_dataset.py --config "$PY_CONFIG"
  echo "[full] nnU-Net training/inference are intentionally explicit; set run_nnunet=true in config or run scripts 03/04 manually."
  python python/scripts/05_compute_segmentation_metrics.py --config "$PY_CONFIG" || true
  python python/scripts/06_extract_radiomics.py --config "$PY_CONFIG"
  python python/scripts/07_train_3dresnet_cox.py --config "$PY_CONFIG"
  python python/scripts/08_extract_deep_features.py --config "$PY_CONFIG"
  python python/scripts/09_ebv_hbv_interaction.py --config "$PY_CONFIG"
  python python/scripts/10_threshold_bootstrap.py --config "$PY_CONFIG"
  python python/scripts/11_lasso_cox_models.py --config "$PY_CONFIG"
  python python/scripts/12_evaluate_models.py --config "$PY_CONFIG"
  python python/scripts/13_make_figures.py --config "$PY_CONFIG"
  Rscript R/run_scRNA_pipeline.R "$R_CONFIG"
  echo "[full] Done."
else
  echo "Usage: bash run_all.sh {demo|full} [python_config] [scRNA_config]" >&2
  exit 1
fi
