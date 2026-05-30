#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
config_path <- ifelse(length(args) >= 1, args[[1]], "config/scRNA_config.yaml")

source("R/functions/utils.R")
source("R/scripts/01_load_qc.R")
source("R/scripts/02_sct_harmony_cluster.R")
source("R/scripts/03_annotate_cells.R")
source("R/scripts/04_subset_proportions.R")
source("R/scripts/05_pseudobulk_DE.R")
source("R/scripts/06_go_enrichment.R")
source("R/scripts/07_exhaustion_scores_correlation.R")
source("R/scripts/08_plot_figure6.R")

cfg <- read_cfg(config_path)
set.seed(as.integer(get_cfg(cfg, c("project", "seed"), 1234)))
outdir <- get_cfg(cfg, c("project", "output_dir"), "results/full/scRNA")
make_dir(outdir)

message("[1/8] Load 10x matrices and perform QC")
obj <- load_and_qc_scRNA(cfg)

message("[2/8] SCTransform, Harmony, clustering, UMAP")
obj <- run_sct_harmony_cluster(obj, cfg)

message("[3/8] Cell-type annotation")
obj <- annotate_cell_types(obj, cfg)

message("[4/8] Lymphocyte subset proportions")
prop_tests <- calculate_lymphocyte_proportions(obj, cfg)

message("[5/8] Donor-aware pseudobulk differential expression")
de_all <- run_pseudobulk_de(obj, cfg)

message("[6/8] GO enrichment")
go <- run_go_enrichment(de_all, cfg)

message("[7/8] Exhaustion/cytotoxic scores and image-transcriptomic correlation")
corr_result <- run_exhaustion_score_and_correlation(obj, cfg)
obj <- corr_result$obj

message("[8/8] Plot figure panels")
plot_scRNA_outputs(obj, de_all, go, corr_result, cfg)

safe_save_rds(obj, file.path(outdir, "final_scRNA_object.rds"))
message("Done.")
