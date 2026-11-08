# HBsAg status and multimodal MRI refine metastatic risk associated with EBV DNA in nasopharyngeal carcinoma

This repository contains the analysis code and trained model weights for the study:

**HBsAg status and multimodal MRI refine metastatic risk associated with EBV DNA in nasopharyngeal carcinoma**

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

The revision implementation documents the methods reported by the authors. The
earlier public snapshot (`c6c4b5c`) screened features before cross-validation,
tuned the Cox penalty by C-index, used an inner-validation fraction of 15% and
patience of 20, and used edgeR for pseudobulk analysis. These are distinct
implementations. Updating the code does not establish which executable produced
an archived table, selected checkpoint, patient-level MRI score or figure.

The clinical interaction script estimates the adjusted log-linear association
in Group I (development and internal-test patients together). Its optional
interpretation via relative hazards is log-linear; the archived Figure 3 spline
display uses a separate log-base-10 model. The threshold script reconstructs
the strict-greater-than historical DMFS discovery scan in Group I. Use the
historical recorded DMFS times when checking that archived scan. For the
revised cause-specific clinical analyses, supply adjudicated endpoint times
in months from first treatment and status indicators. All downstream viral
groups use the fixed inclusive threshold of 6,020 copies/mL. The historical
1,000 resample-level cutoffs and the original executable for that bootstrap are
unavailable. The full configuration reconstructs the grid only; the synthetic
demo illustrates a separate resampling workflow.

The historical retrospective MRI models included treatments received. The
revision additionally evaluated a supplied endpoint-specific imaging score
alongside pretreatment clinical covariates and the four viral groups. The
added-score comparison depends on the particular patient-level score export;
training this example pipeline creates a new fit, with its own selected
features and checkpoint. The internal test cohort was held out from downstream
MRI fitting but contributed to the discovery of the clinical threshold.

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
- Adjusted continuous interaction: `ln(1 + EBV DNA in copies/mL / 1000)`; documented undetectable results are zero and missing values remain missing.
- Locked EBV DNA threshold: `6.02e3 copies/mL`
- Historical DMFS grid: 10 through 100,000 copies/mL in steps of 10; maximize the HBsAg-positive stratum HR from a single binary predictor (L2 penalty 0.01). The first tied maximizer is reported. Results depend on the supplied discovery data.
- Clinical covariates: age, sex, AJCC stage, chemotherapy, induction chemotherapy
- MRI preprocessing: N4 bias correction, rigid registration to CE-T1WI, isotropic resampling, per-patient z-score normalization
- Radiomics: PyRadiomics, fixed bin width 25, original + exponential + gradient + logarithm + square + square-root + LoG + 3D wavelet filters
- Radiomic robustness filter: inter-observer ICC >= 0.80
- Deep learning: ResNet-50, Adam lr 1e-4, batch size 4, up to 200 epochs; use the checkpoint minimizing inner-validation Cox loss with patience 10. Supply `resnet_subset` in the clinical CSV for the actual gradient and validation assignments. Otherwise a new seeded split allocates 128 of 899 development patients to validation. Epoch 163 is the reported study checkpoint; a new fit selects its own epoch.
- Cox tuning: fit preprocessing and univariable screening on the training portion of each development fold; use ten-fold partial-likelihood deviance and the largest penalty within one SE of the minimum. Evaluate concordance on separate test cohorts.
- Single-cell: aggregate raw UMI counts by donor and cell type, then compare HBsAg-positive with HBsAg-negative donors using DESeq2. Report baseMean, log2 fold change, unadjusted P and within-cell-type BH-adjusted P. Figure 6d stars use the adjusted values. The optional signature-level correlation plot is separate from the archived seven-feature, 14-gene heat map in Supplementary Figure 11.

## Important notes

1. The repository will not reproduce published values unless the original de-identified data, imaging files, segmentation masks and metadata are placed in the expected locations.
2. `nnU-Net v2` commands are wrapped rather than reimplemented. Install nnU-Net separately and set the required environment variables before running segmentation training or inference.
3. MedicalNet pretrained weights are not redistributed here. Put the permitted checkpoint under `models/pretrained/` and set the path in `config/config.yaml`.
4. Do not commit raw NIfTI files, raw 10x matrices, patient identifiers, additional generated checkpoints or institutional documents unless release permissions explicitly allow it.
