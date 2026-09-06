# HBV modifies EBV-associated systemic risk refined by multimodal MRI in nasopharyngeal carcinoma code

This repository contains the analysis code and trained model weights for the study:

**HBV modifies EBV-associated systemic risk refined by multimodal MRI in nasopharyngeal carcinoma**

The repository is intended for paper code availability. It includes source code, configuration templates, environment files and the trained weights in `models/`. Protected patient-level clinical, imaging, survival and single-cell data are not included, and generated results/logs are intentionally omitted.

## Repository map

```text
NPC_paper_code/
|-- R/
|   |-- run_scRNA_pipeline.R
|   |-- functions/
|   `-- scripts/
|-- python/
|   |-- npcpaper/
|   `-- scripts/
|-- config/
|-- models/
|-- run_all.sh
|-- Makefile
|-- environment.yml
`-- requirements.txt
```

## Lightweight demo and automatic checks

The synthetic demo needs only Python 3.12 and the frozen analysis dependencies;
it does not need Conda, R, MRI software, model weights or patient data.
From the repository root (Bash; on Windows use Git Bash):

```bash
python -m venv .venv-demo
source .venv-demo/bin/activate  # Windows Git Bash: source .venv-demo/Scripts/activate
python -m pip install -r requirements/demo/requirements.txt
python -m pip check
bash run_all.sh demo
```

The demo generates synthetic inputs, runs interaction/threshold analyses and
LASSO Cox modeling, and checks the resulting patient scores and cohort summaries.
GitHub Actions runs this same command for pull requests and changes to `master`.
It does not reproduce manuscript values. See [maintenance and environment status](MAINTENANCE.md)
for snapshot provenance, dependency updates and the limits of CI coverage.

## Full research workflow

`environment.yml`, root `requirements.txt` and the R package installer describe
the research environment but are not a validated lock of the published analysis.
The demo dependency snapshot covers only the demo. Capture and validate the full
environment on the research machine before relying on it for reproduction.

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate npc-paper
Rscript R/scripts/00_install_packages.R
```

### 2. Run the workflow on de-identified data

```bash
bash run_all.sh full config/config.yaml config/scRNA_config.yaml
```

## Released weights

The trained weights are stored with Git LFS under `models/`:

- `models/nnNet_best.pth`
- `models/Resnet-3D_best.pth`

## Data expected by the pipelines

### Clinical survival CSV

Required columns include:

- `patient_id`
- `cohort`: one of `train`, `internal`, `external`
- `age`, `sex`, `ajcc_stage`, `chemotherapy`, `induction_chemotherapy`
- `hbsag`: 0 for negative, 1 for positive
- `ebv_dna`: pretreatment plasma EBV DNA in copies/mL
- endpoint time/event pairs such as `dmfs_time`, `dmfs_event`, `os_time`, `os_event`, `pfs_time`, `pfs_event`, `lrrfs_time`, `lrrfs_event`

### Imaging manifest CSV

Each row is one patient with paths to preprocessed or raw NIfTI files:

- `patient_id`
- `t1_path`, `t2_path`, `t1c_path`
- `mask_path` for manual or nnU-Net mask
- `cohort`

### scRNA metadata CSV

Required columns include:

- `sample_id`, `patient_id`
- `tenx_path`
- `hbsag`: 0 or 1
- `ebv_dna`
- optional `batch`, `site`, `cluster_map_path`

## Main parameters encoded in config

- Reproducibility seed: `1234`
- EBV DNA transformation: `log10(EBV DNA + 1)`
- Locked EBV DNA threshold: `6.02e3 copies/mL`
- Threshold stability constraints: at least 10% of patients per EBV-high/low arm within each HBsAg stratum and at least 10 endpoint events per arm
- Bootstrap iterations: 1,000
- Clinical covariates: age, sex, AJCC stage, chemotherapy, induction chemotherapy
- MRI preprocessing: N4 bias correction, rigid registration to CE-T1WI, isotropic resampling, per-patient z-score normalization
- Radiomics: PyRadiomics, fixed bin width 25, original + exponential + gradient + logarithm + square + square-root + LoG + 3D wavelet filters
- Radiomic robustness filter: inter-observer ICC >= 0.80
- Deep learning: 3D ResNet-50/MedicalNet-style encoder, 128 x 128 x 128 crop, Adam lr 1e-4, batch size 4, max epochs 200, multi-endpoint Cox losses for OS/PFS/DMFS
- scRNA: Seurat, SCTransform, Harmony, UMAP, marker-based annotation, donor-aware pseudobulk differential expression, clusterProfiler GO enrichment, Spearman image-transcriptomic correlation with BH-FDR control

## Important notes

1. The repository will not reproduce published values unless the original de-identified data, imaging files, segmentation masks and metadata are placed in the expected locations.
2. `nnU-Net v2` commands are wrapped rather than reimplemented. Install nnU-Net separately and set the required environment variables before running segmentation training or inference.
3. MedicalNet pretrained weights are not redistributed here. Put the permitted checkpoint under `models/pretrained/` and set the path in `config/config.yaml`.
4. Do not commit raw NIfTI files, raw 10x matrices, patient identifiers, additional generated checkpoints or institutional documents unless release permissions explicitly allow it.
