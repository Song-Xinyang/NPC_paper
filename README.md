# NPC dual-virus MRI paper code

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

## One-click use

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

### 3. Run a lightweight demo with synthetic data

```bash
bash run_all.sh demo
```

The demo verifies the repository structure and the survival/modeling code. It does not reproduce manuscript values because the protected patient-level data, MRI volumes, segmentation masks and metadata are not included.

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
