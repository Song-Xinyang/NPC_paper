run_exhaustion_score_and_correlation <- function(obj, cfg) {
  source("R/functions/utils.R")
  source("R/functions/markers.R")
  outdir <- get_cfg(cfg, c("project", "output_dir"))
  DefaultAssay(obj) <- "SCT"
  ex_genes <- intersect(unlist(get_cfg(cfg, c("signatures", "exhaustion_genes"), as.list(exhaustion_genes_default))), rownames(obj))
  cyto_genes <- intersect(unlist(get_cfg(cfg, c("signatures", "cytotoxic_genes"), as.list(cytotoxic_genes_default))), rownames(obj))
  if (length(ex_genes) >= 2) obj <- AddModuleScore(obj, features = list(ex_genes), name = "Exhaustion", search = FALSE)
  if (length(cyto_genes) >= 2) obj <- AddModuleScore(obj, features = list(cyto_genes), name = "Cytotoxic", search = FALSE)
  donor_scores <- obj@meta.data %>%
    group_by(patient_id, hbsag_group, lymphocyte_subset) %>%
    summarise(
      n_cells = n(),
      exhaustion_score = mean(Exhaustion1, na.rm = TRUE),
      cytotoxic_score = mean(Cytotoxic1, na.rm = TRUE),
      .groups = "drop"
    )
  safe_write_csv(donor_scores, file.path(outdir, "07_donor_level_signature_scores.csv"))
  image_path <- get_cfg(cfg, c("input", "imaging_features_csv"), NULL)
  if (is.null(image_path) || !file.exists(image_path)) {
    warning("No imaging feature CSV found; skipping image-transcriptomic correlation.")
    safe_save_rds(obj, file.path(outdir, "07_signature_scored.rds"))
    return(list(obj = obj, donor_scores = donor_scores, correlation = tibble::tibble()))
  }
  img <- readr::read_csv(image_path, show_col_types = FALSE)
  features <- unlist(get_cfg(cfg, c("correlation", "imaging_feature_columns"), list()))
  features <- intersect(features, colnames(img))
  dat <- donor_scores %>% filter(lymphocyte_subset %in% c("CD8+ T", "CD4+ T", "Proliferating T", "NK cell")) %>%
    left_join(img, by = "patient_id")
  rows <- list()
  for (ct in unique(dat$lymphocyte_subset)) {
    sub <- dat %>% filter(lymphocyte_subset == ct)
    for (feat in features) {
      for (score_col in c("exhaustion_score", "cytotoxic_score")) {
        ok <- is.finite(sub[[feat]]) & is.finite(sub[[score_col]])
        if (sum(ok) >= 4) {
          test <- suppressWarnings(cor.test(sub[[feat]][ok], sub[[score_col]][ok], method = "spearman", exact = FALSE))
          rows[[length(rows) + 1]] <- tibble::tibble(
            lymphocyte_subset = ct,
            imaging_feature = feat,
            signature = score_col,
            rho = unname(test$estimate),
            p_value = test$p.value,
            n = sum(ok)
          )
        }
      }
    }
  }
  corr <- bind_rows(rows) %>% mutate(p_adj = p.adjust(p_value, method = "BH"))
  safe_write_csv(corr, file.path(outdir, "07_image_transcriptomic_correlations.csv"))
  safe_save_rds(obj, file.path(outdir, "07_signature_scored.rds"))
  list(obj = obj, donor_scores = donor_scores, correlation = corr)
}
