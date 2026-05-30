run_sct_harmony_cluster <- function(obj, cfg) {
  source("R/functions/utils.R")
  suppressPackageStartupMessages({ library(harmony) })
  vars <- unlist(get_cfg(cfg, c("normalization", "vars_to_regress"), list("percent.mt")))
  npcs <- as.integer(get_cfg(cfg, c("clustering", "n_pcs"), 30))
  resolution <- as.numeric(get_cfg(cfg, c("clustering", "resolution"), 0.6))
  harmony_group <- get_cfg(cfg, c("normalization", "harmony_group"), "sample_id")
  obj <- SCTransform(obj, vars.to.regress = vars, verbose = FALSE)
  obj <- RunPCA(obj, npcs = npcs, verbose = FALSE)
  obj <- RunHarmony(obj, group.by.vars = harmony_group, reduction = "pca", dims.use = seq_len(npcs), verbose = FALSE)
  obj <- FindNeighbors(obj, reduction = "harmony", dims = seq_len(npcs), verbose = FALSE)
  obj <- FindClusters(obj, resolution = resolution, verbose = FALSE)
  obj <- RunUMAP(
    obj,
    reduction = "harmony",
    dims = seq_len(npcs),
    min.dist = as.numeric(get_cfg(cfg, c("clustering", "umap_min_dist"), 0.3)),
    n.neighbors = as.integer(get_cfg(cfg, c("clustering", "umap_n_neighbors"), 30)),
    verbose = FALSE
  )
  safe_save_rds(obj, file.path(get_cfg(cfg, c("project", "output_dir")), "02_sct_harmony_clustered.rds"))
  obj
}
