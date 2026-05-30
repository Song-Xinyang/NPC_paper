calculate_lymphocyte_proportions <- function(obj, cfg) {
  source("R/functions/utils.R")
  outdir <- get_cfg(cfg, c("project", "output_dir"))
  lymph_types <- unlist(get_cfg(cfg, c("cell_types", "lymphocyte_cell_types"), list("CD4+ T", "CD8+ T", "Proliferating T", "NK cell")))
  meta <- obj@meta.data %>%
    mutate(lymphocyte_subset = as.character(lymphocyte_subset)) %>%
    filter(lymphocyte_subset %in% lymph_types)
  counts <- meta %>%
    count(patient_id, hbsag_group, lymphocyte_subset, name = "n_cells") %>%
    group_by(patient_id) %>%
    mutate(prop = n_cells / sum(n_cells)) %>%
    ungroup()
  wide <- counts %>% select(patient_id, hbsag_group, lymphocyte_subset, prop) %>% tidyr::pivot_wider(names_from = lymphocyte_subset, values_from = prop, values_fill = 0)
  tests <- counts %>%
    group_by(lymphocyte_subset) %>%
    summarise(
      non_HBV_mean = mean(prop[hbsag_group == "non_HBV"]),
      HBV_mean = mean(prop[hbsag_group == "HBV"]),
      t_value = ifelse(length(unique(hbsag_group)) == 2, unname(t.test(prop ~ hbsag_group)$statistic), NA_real_),
      p_value = ifelse(length(unique(hbsag_group)) == 2, t.test(prop ~ hbsag_group)$p.value, NA_real_),
      .groups = "drop"
    ) %>% mutate(p_adj = p.adjust(p_value, method = "BH"))
  safe_write_csv(counts, file.path(outdir, "04_lymphocyte_subset_proportions_long.csv"))
  safe_write_csv(wide, file.path(outdir, "04_lymphocyte_subset_proportions_wide.csv"))
  safe_write_csv(tests, file.path(outdir, "04_lymphocyte_subset_proportion_tests.csv"))
  tests
}
