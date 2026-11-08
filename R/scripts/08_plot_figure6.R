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

  shown_genes <- unlist(get_cfg(cfg, c("pseudobulk", "genes_of_interest"), as.list(immune_genes_of_interest_default)))
  focus <- de_all %>% filter(gene %in% shown_genes, is.finite(log2FoldChange))
  if (nrow(focus) > 0) {
    focus$gene_class <- ifelse(focus$gene %in% exhaustion_genes_default, "Checkpoint/Dysfunction", "Cytotoxicity")
    focus$stars <- case_when(is.na(focus$padj) ~ "", focus$padj < 0.01 ~ "**",
                             focus$padj < 0.05 ~ "*", TRUE ~ "")
    p4 <- focus %>%
      filter(cell_type %in% c("CD4+ T", "CD8+ T", "NK cell", "Proliferating T")) %>%
      ggplot(aes(x = log2FoldChange, y = gene, fill = gene_class)) +
      geom_col() +
      geom_text(aes(label = stars, hjust = ifelse(log2FoldChange >= 0, -0.15, 1.15))) +
      scale_x_continuous(expand = expansion(mult = c(0.14, 0.14))) +
      facet_wrap(~cell_type, scales = "free_y") +
      theme_bw() + theme(axis.text.y = element_text(face = "italic")) +
      labs(x = "log2 fold change (HBsAg-positive vs HBsAg-negative)", y = NULL, fill = "Gene class",
                        caption = "* BH-adjusted P < 0.05; ** BH-adjusted P < 0.01")
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
    # This signature-level plot is an exploratory output distinct from the archived gene-level heatmap.
    png(file.path(outdir, "figures", "exploratory_signature_correlations.png"), width = 1600, height = 1000, res = 200)
    pheatmap::pheatmap(mat, cluster_rows = FALSE, cluster_cols = FALSE, main = "Spearman rho: imaging features vs exhaustion score")
    dev.off()
  }
  invisible(TRUE)
}
