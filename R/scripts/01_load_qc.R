load_and_qc_scRNA <- function(cfg) {
  source("R/functions/utils.R")
  metadata_path <- get_cfg(cfg, c("input", "metadata_csv"))
  meta <- readr::read_csv(metadata_path, show_col_types = FALSE)
  sample_col <- get_cfg(cfg, c("columns", "sample_id"), "sample_id")
  patient_col <- get_cfg(cfg, c("columns", "patient_id"), "patient_id")
  tenx_col <- get_cfg(cfg, c("columns", "tenx_path"), "tenx_path")
  hbsag_col <- get_cfg(cfg, c("columns", "hbsag"), "hbsag")
  ebv_col <- get_cfg(cfg, c("columns", "ebv_dna"), "ebv_dna")
  if (isTRUE(get_cfg(cfg, c("filtering", "keep_ebv_high_only"), TRUE))) {
    cutoff <- as.numeric(get_cfg(cfg, c("filtering", "locked_ebv_cutoff"), 6020))
    meta <- meta %>% filter(.data[[ebv_col]] >= cutoff)
  }
  meta <- add_hbsag_group(meta, hbsag_col)
  objects <- list()
  for (i in seq_len(nrow(meta))) {
    sample_id <- as.character(meta[[sample_col]][i])
    message("Reading ", sample_id)
    counts <- read_10x_any(as.character(meta[[tenx_col]][i]))
    if (is.list(counts)) counts <- counts[[1]]
    obj <- CreateSeuratObject(
      counts = counts,
      project = sample_id,
      min.cells = as.integer(get_cfg(cfg, c("filtering", "min_cells_per_gene"), 3)),
      min.features = as.integer(get_cfg(cfg, c("filtering", "min_features"), 200))
    )
    obj$sample_id <- sample_id
    obj$patient_id <- as.character(meta[[patient_col]][i])
    obj$hbsag <- as.integer(meta[[hbsag_col]][i])
    obj$hbsag_group <- ifelse(obj$hbsag == 1, "HBV", "non_HBV")
    obj$ebv_dna <- as.numeric(meta[[ebv_col]][i])
    batch_col <- get_cfg(cfg, c("columns", "batch"), "batch")
    obj$batch <- if (batch_col %in% colnames(meta)) as.character(meta[[batch_col]][i]) else sample_id
    mt_pat <- percent_mito_pattern(rownames(obj))
    obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = mt_pat)
    objects[[sample_id]] <- obj
  }
  merged <- Reduce(function(x, y) merge(x, y), objects)
  min_features <- as.integer(get_cfg(cfg, c("filtering", "min_features"), 200))
  max_features <- as.integer(get_cfg(cfg, c("filtering", "max_features"), 7500))
  max_percent_mt <- as.numeric(get_cfg(cfg, c("filtering", "max_percent_mt"), 20))
  merged <- subset(merged, subset = nFeature_RNA >= min_features & nFeature_RNA <= max_features & percent.mt <= max_percent_mt)
  safe_save_rds(merged, file.path(get_cfg(cfg, c("project", "output_dir")), "01_qc_merged.rds"))
  merged
}
