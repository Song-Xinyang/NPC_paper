run_go_enrichment <- function(de_all, cfg) {
  source("R/functions/utils.R")
  outdir <- get_cfg(cfg, c("project", "output_dir"))
  if (!requireNamespace("clusterProfiler", quietly = TRUE) || !requireNamespace("org.Hs.eg.db", quietly = TRUE)) {
    warning("clusterProfiler/org.Hs.eg.db not installed; skipping GO enrichment.")
    return(tibble::tibble())
  }
  suppressPackageStartupMessages({ library(clusterProfiler); library(org.Hs.eg.db) })
  fdr <- as.numeric(get_cfg(cfg, c("pseudobulk", "fdr_threshold"), 0.05))
  enrich_list <- list()
  for (ct in unique(de_all$cell_type)) {
    genes <- de_all %>% filter(cell_type == ct, padj < fdr) %>% arrange(padj) %>% pull(gene) %>% unique()
    if (length(genes) < 5) next
    eg <- bitr(genes, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db, drop = TRUE)
    if (nrow(eg) < 5) next
    ego <- enrichGO(
      gene = eg$ENTREZID,
      OrgDb = org.Hs.eg.db,
      keyType = "ENTREZID",
      ont = "BP",
      pAdjustMethod = "BH",
      qvalueCutoff = 0.25,
      readable = TRUE
    )
    if (!is.null(ego) && nrow(as.data.frame(ego)) > 0) {
      enrich_list[[ct]] <- as.data.frame(ego) %>% mutate(cell_type = ct)
    }
  }
  go <- bind_rows(enrich_list)
  safe_write_csv(go, file.path(outdir, "06_GO_enrichment_BP.csv"))
  go
}
