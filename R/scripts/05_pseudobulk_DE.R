run_pseudobulk_de <- function(obj, cfg) {
  source("R/functions/utils.R")
  source("R/functions/markers.R")
  outdir <- get_cfg(cfg, c("project", "output_dir"))
  group_col <- get_cfg(cfg, c("pseudobulk", "group_col"), "hbsag_group")
  negative_label <- get_cfg(cfg, c("pseudobulk", "negative_label"), "HBsAg-negative")
  positive_label <- get_cfg(cfg, c("pseudobulk", "positive_label"), "HBsAg-positive")
  if (get_cfg(cfg, c("pseudobulk", "method"), "DESeq2") != "DESeq2") {
    stop("Use pseudobulk.method: DESeq2 for this workflow.")
  }
  min_cells <- as.integer(get_cfg(cfg, c("filtering", "min_cells_per_donor_celltype"), 20))
  genes_interest <- unlist(get_cfg(cfg, c("pseudobulk", "genes_of_interest"), as.list(immune_genes_of_interest_default)))
  if (inherits(obj[["RNA"]], "Assay5")) {
    rna <- SeuratObject::JoinLayers(obj[["RNA"]], layers = "counts", new = "counts")
    counts <- SeuratObject::LayerData(rna, layer = "counts")
  } else {
    counts <- GetAssayData(obj, assay = "RNA", slot = "counts")
  }
  meta <- obj[[]][colnames(counts), , drop = FALSE]
  if (!all(c("patient_id", group_col, "lymphocyte_subset") %in% names(meta))) {
    stop("Missing donor ID, HBsAg group or cell-type annotation in RNA count metadata.")
  }
  group_labels <- as.character(meta[[group_col]])
  group_labels[group_labels %in% c("HBV", "HBsAg-positive")] <- positive_label
  group_labels[group_labels %in% c("non_HBV", "HBsAg-negative")] <- negative_label
  meta[[group_col]] <- group_labels
  if (anyNA(meta$patient_id) || any(!nzchar(as.character(meta$patient_id))) ||
      any(!meta[[group_col]] %in% c(negative_label, positive_label)) ||
      nrow(unique(meta[c("patient_id", group_col)])) != length(unique(meta$patient_id))) {
    stop("Every donor must have one known HBsAg group across all analysed cells.")
  }
  cell_types <- unlist(get_cfg(cfg, c("cell_types", "lymphocyte_cell_types"),
                             list("CD4+ T", "CD8+ T", "NK cell", "Proliferating T")))
  all_res <- list()
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
    donor_rows <- match(colnames(pb), as.character(sub_meta$patient_id))
    group <- factor(sub_meta[[group_col]][donor_rows], levels = c(negative_label, positive_label))
    donor_meta <- data.frame(group = group, row.names = colnames(pb))
    if (any(pb < 0) || any(pb != floor(pb)) || any(colSums(pb) == 0)) {
      stop("Expected non-negative raw UMI counts and non-empty donor libraries for ", ct)
    }
    dds <- DESeq2::DESeqDataSetFromMatrix(pb, donor_meta, design = ~ group)
    # Sparse pseudobulks can lack a gene with positive counts in every donor.
    size_factor_type <- if (any(rowSums(pb == 0) == 0)) "ratio" else "poscounts"
    fit <- DESeq2::DESeq(dds, sfType = size_factor_type, quiet = TRUE)
    res <- DESeq2::results(fit, contrast = c("group", positive_label, negative_label),
                           independentFiltering = FALSE, pAdjustMethod = "BH")
    tab <- as.data.frame(res) %>%
      tibble::rownames_to_column("gene") %>%
      transmute(cell_type = ct, gene, log2FoldChange, baseMean, PValue = pvalue, padj, FDR = padj)
    all_res[[ct]] <- tab
  }
  if (!length(all_res)) stop("No cell type has enough eligible donors in both HBsAg groups.")
  de_all <- bind_rows(all_res)
  de_focus <- de_all %>% filter(gene %in% genes_interest)
  safe_write_csv(de_all, file.path(outdir, "05_pseudobulk_DE_all_genes.csv"))
  safe_write_csv(de_focus, file.path(outdir, "05_pseudobulk_DE_exhaustion_cytotoxic_genes.csv"))
  de_all
}
