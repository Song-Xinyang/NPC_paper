run_pseudobulk_de <- function(obj, cfg) {
  source("R/functions/utils.R")
  source("R/functions/markers.R")
  suppressPackageStartupMessages({ library(edgeR) })
  outdir <- get_cfg(cfg, c("project", "output_dir"))
  group_col <- get_cfg(cfg, c("pseudobulk", "group_col"), "hbsag_group")
  min_cells <- as.integer(get_cfg(cfg, c("filtering", "min_cells_per_donor_celltype"), 20))
  genes_interest <- unlist(get_cfg(cfg, c("pseudobulk", "genes_of_interest"), as.list(immune_genes_of_interest_default)))
  DefaultAssay(obj) <- "RNA"
  counts <- GetAssayData(obj, assay = "RNA", slot = "counts")
  meta <- obj@meta.data
  cell_types <- sort(unique(meta$lymphocyte_subset))
  all_res <- list()
  focused <- list()
  for (ct in cell_types) {
    idx <- which(meta$lymphocyte_subset == ct)
    if (length(idx) < min_cells * 2) next
    sub_meta <- meta[idx, , drop = FALSE]
    donor_counts <- sub_meta %>% count(patient_id, .data[[group_col]], name = "n_cells")
    keep_donors <- donor_counts %>% filter(n_cells >= min_cells) %>% pull(patient_id)
    idx <- idx[sub_meta$patient_id %in% keep_donors]
    sub_meta <- meta[idx, , drop = FALSE]
    if (length(unique(sub_meta[[group_col]])) < 2 || length(unique(sub_meta$patient_id)) < 4) next
    pb <- aggregate_counts_by_group(counts[, idx, drop = FALSE], sub_meta$patient_id)
    donor_meta <- sub_meta %>% distinct(patient_id, .data[[group_col]]) %>% arrange(match(patient_id, colnames(pb)))
    donor_meta <- donor_meta[match(colnames(pb), donor_meta$patient_id), , drop = FALSE]
    y <- DGEList(counts = pb, samples = donor_meta)
    keep <- filterByExpr(y, group = donor_meta[[group_col]])
    y <- y[keep, , keep.lib.sizes = FALSE]
    y <- calcNormFactors(y)
    design <- model.matrix(~ donor_meta[[group_col]])
    colnames(design) <- c("Intercept", "HBV_vs_non_HBV")
    y <- estimateDisp(y, design)
    fit <- glmQLFit(y, design)
    qlf <- glmQLFTest(fit, coef = "HBV_vs_non_HBV")
    tab <- topTags(qlf, n = Inf)$table %>%
      tibble::rownames_to_column("gene") %>%
      mutate(cell_type = ct, padj = FDR, baseMean = logCPM, log2FoldChange = logFC) %>%
      select(cell_type, gene, log2FoldChange, baseMean, PValue, padj, FDR)
    all_res[[ct]] <- tab
    focused[[ct]] <- tab %>% filter(gene %in% genes_interest)
  }
  de_all <- bind_rows(all_res)
  de_focus <- bind_rows(focused)
  safe_write_csv(de_all, file.path(outdir, "05_pseudobulk_DE_all_genes.csv"))
  safe_write_csv(de_focus, file.path(outdir, "05_pseudobulk_DE_exhaustion_cytotoxic_genes.csv"))
  de_all
}
