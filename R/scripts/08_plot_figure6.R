plot_scRNA_outputs <- function(obj, de_all, go, corr_result, cfg) {
  source("R/functions/utils.R")
  source("R/functions/markers.R")
  suppressPackageStartupMessages({ library(patchwork); library(pheatmap) })
  outdir <- get_cfg(cfg, c("project", "output_dir"))
  make_dir(file.path(outdir, "figures"))
  p1 <- DimPlot(obj, reduction = "umap", group.by = "sample_id", raster = TRUE) + ggtitle("Samples")
  p2 <- DimPlot(obj, reduction = "umap", group.by = "cell_type", label = TRUE, repel = TRUE, raster = TRUE) + ggtitle("Cell types")
  ggsave(file.path(outdir, "figures", "Fig6b_UMAP_sample_celltype.png"), p1 + p2, width = 12, height = 5, dpi = 300)

  genes <- unique(unlist(marker_sets))
  genes <- intersect(genes, rownames(obj))
  if (length(genes) >= 5) {
    p3 <- DoHeatmap(obj, features = genes, group.by = "cell_type", size = 3) + ggtitle("Canonical markers")
    ggsave(file.path(outdir, "figures", "Fig6c_marker_heatmap.png"), p3, width = 10, height = 8, dpi = 300)
  }

  focus_path <- file.path(outdir, "05_pseudobulk_DE_exhaustion_cytotoxic_genes.csv")
  if (file.exists(focus_path)) {
    focus <- readr::read_csv(focus_path, show_col_types = FALSE)
    focus$gene_class <- ifelse(focus$gene %in% exhaustion_genes_default, "Checkpoint/Dysfunction", "Cytotoxicity")
    p4 <- focus %>%
      filter(cell_type %in% c("CD4+ T", "CD8+ T", "NK cell", "Proliferating T")) %>%
      ggplot(aes(x = log2FoldChange, y = gene, fill = gene_class)) +
      geom_col() +
      facet_wrap(~cell_type, scales = "free_y") +
      theme_bw() + xlab("log2 fold change (HBV vs non-HBV)") + ylab(NULL)
    ggsave(file.path(outdir, "figures", "Fig6d_exhaustion_cytotoxic_barplot.png"), p4, width = 12, height = 6, dpi = 300)
  }

  if (!is.null(go) && nrow(go) > 0) {
    top_go <- go %>% arrange(p.adjust) %>% group_by(cell_type) %>% slice_head(n = 8) %>% ungroup()
    p5 <- top_go %>%
      mutate(term = stringr::str_trunc(Description, 45), neglog10 = -log10(p.adjust)) %>%
      ggplot(aes(x = neglog10, y = reorder(term, neglog10), fill = cell_type)) +
      geom_col(show.legend = FALSE) + facet_wrap(~cell_type, scales = "free_y") + theme_bw() +
      xlab("-log10 adjusted P") + ylab(NULL)
    ggsave(file.path(outdir, "figures", "Fig6e_GO_terms.png"), p5, width = 12, height = 8, dpi = 300)
  }

  corr <- corr_result$correlation
  if (!is.null(corr) && nrow(corr) > 0) {
    mat <- corr %>% filter(signature == "exhaustion_score") %>% select(lymphocyte_subset, imaging_feature, rho) %>%
      tidyr::pivot_wider(names_from = imaging_feature, values_from = rho) %>% tibble::column_to_rownames("lymphocyte_subset") %>% as.matrix()
    png(file.path(outdir, "figures", "Fig6f_image_exhaustion_correlation_heatmap.png"), width = 1600, height = 1000, res = 200)
    pheatmap::pheatmap(mat, cluster_rows = FALSE, cluster_cols = FALSE, main = "Spearman rho: imaging features vs exhaustion score")
    dev.off()
  }
  invisible(TRUE)
}
