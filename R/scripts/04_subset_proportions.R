calculate_lymphocyte_proportions <- function(obj, cfg) {
  source("R/functions/utils.R")
  outdir <- get_cfg(cfg, c("project", "output_dir"))
  lymph_types <- unlist(get_cfg(cfg, c("cell_types", "lymphocyte_cell_types"), list("CD4+ T", "CD8+ T", "Proliferating T", "NK cell")))
  negative_label <- get_cfg(cfg, c("pseudobulk", "negative_label"), "HBsAg-negative")
  positive_label <- get_cfg(cfg, c("pseudobulk", "positive_label"), "HBsAg-positive")
  meta <- obj@meta.data %>%
    mutate(lymphocyte_subset = as.character(lymphocyte_subset)) %>%
    filter(lymphocyte_subset %in% lymph_types)
  counts <- meta %>%
    count(patient_id, hbsag_group, lymphocyte_subset, name = "n_cells") %>%
    tidyr::complete(tidyr::nesting(patient_id, hbsag_group), lymphocyte_subset = lymph_types, fill = list(n_cells = 0L)) %>%
    group_by(patient_id) %>%
    mutate(prop = n_cells / sum(n_cells)) %>%
    ungroup()
  wide <- counts %>% select(patient_id, hbsag_group, lymphocyte_subset, prop) %>% tidyr::pivot_wider(names_from = lymphocyte_subset, values_from = prop, values_fill = 0)
  tests <- counts %>%
    group_by(lymphocyte_subset) %>%
    summarise(
      HBsAg_negative_mean = mean(prop[hbsag_group == negative_label]),
      HBsAg_positive_mean = mean(prop[hbsag_group == positive_label]),
      test = list(if (min(sum(hbsag_group == negative_label), sum(hbsag_group == positive_label)) >= 2L) {
        tryCatch(t.test(prop[hbsag_group == negative_label], prop[hbsag_group == positive_label],
                        var.equal = TRUE), error = function(e) NULL)
      }),
      .groups = "drop"
    ) %>% mutate(t_value = vapply(test, function(x) if (is.null(x)) NA_real_ else unname(x$statistic), numeric(1)),
                 p_value = vapply(test, function(x) if (is.null(x)) NA_real_ else x$p.value, numeric(1)),
                 p_adj = p.adjust(p_value, method = "BH")) %>% select(-test)
  safe_write_csv(counts, file.path(outdir, "04_lymphocyte_subset_proportions_long.csv"))
  safe_write_csv(wide, file.path(outdir, "04_lymphocyte_subset_proportions_wide.csv"))
  safe_write_csv(tests, file.path(outdir, "04_lymphocyte_subset_proportion_tests.csv"))
  tests
}
