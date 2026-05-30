annotate_cell_types <- function(obj, cfg) {
  source("R/functions/utils.R")
  source("R/functions/markers.R")
  outdir <- get_cfg(cfg, c("project", "output_dir"))
  cluster_map_path <- get_cfg(cfg, c("input", "cluster_map_csv"), NULL)
  if (!is.null(cluster_map_path) && file.exists(cluster_map_path)) {
    cluster_map <- readr::read_csv(cluster_map_path, show_col_types = FALSE)
    stopifnot(all(c("seurat_cluster", "cell_type") %in% colnames(cluster_map)))
    mapping <- setNames(cluster_map$cell_type, as.character(cluster_map$seurat_cluster))
    obj$cell_type <- unname(mapping[as.character(obj$seurat_clusters)])
    obj$cell_type[is.na(obj$cell_type)] <- paste0("Cluster_", obj$seurat_clusters[is.na(obj$cell_type)])
  } else {
    DefaultAssay(obj) <- "SCT"
    present_sets <- lapply(marker_sets, function(g) intersect(g, rownames(obj)))
    present_sets <- present_sets[lengths(present_sets) >= 2]
    score_names <- character(0)
    for (nm in names(present_sets)) {
      obj <- AddModuleScore(obj, features = list(present_sets[[nm]]), name = paste0("MS_", make.names(nm)), search = FALSE)
      score_names <- c(score_names, paste0("MS_", make.names(nm), "1"))
    }
    score_df <- obj@meta.data[, score_names, drop = FALSE]
    labels <- names(present_sets)[max.col(score_df, ties.method = "first")]
    obj$cell_type <- labels
  }
  obj$lymphocyte_subset <- obj$cell_type
  if ("CD8A" %in% rownames(obj) && "CD4" %in% rownames(obj)) {
    data <- FetchData(obj, vars = c("CD8A", "CD4", "NKG7", "MKI67"), slot = "data")
    t_like <- grepl("T cell|CD4|CD8|Proliferating", obj$cell_type)
    obj$lymphocyte_subset[t_like & data$MKI67 > quantile(data$MKI67, 0.90, na.rm = TRUE)] <- "Proliferating T"
    obj$lymphocyte_subset[t_like & data$CD8A >= data$CD4 & obj$lymphocyte_subset != "Proliferating T"] <- "CD8+ T"
    obj$lymphocyte_subset[t_like & data$CD4 > data$CD8A & obj$lymphocyte_subset != "Proliferating T"] <- "CD4+ T"
    nk_like <- data$NKG7 > quantile(data$NKG7, 0.75, na.rm = TRUE) & !t_like
    obj$lymphocyte_subset[nk_like] <- "NK cell"
  }
  markers <- FindAllMarkers(obj, only.pos = TRUE, test.use = "wilcox", logfc.threshold = 0.25, min.pct = 0.10)
  safe_write_csv(markers, file.path(outdir, "03_cluster_markers.csv"))
  safe_save_rds(obj, file.path(outdir, "03_annotated.rds"))
  obj
}
