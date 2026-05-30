suppressPackageStartupMessages({
  library(Seurat)
  library(dplyr)
  library(tidyr)
  library(Matrix)
  library(readr)
  library(ggplot2)
  library(yaml)
})

`%||%` <- function(x, y) if (is.null(x)) y else x

make_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE, showWarnings = FALSE)
  invisible(path)
}

read_cfg <- function(path) {
  yaml::read_yaml(path)
}

get_cfg <- function(cfg, keys, default = NULL) {
  cur <- cfg
  for (k in keys) {
    if (is.null(cur[[k]])) return(default)
    cur <- cur[[k]]
  }
  cur
}

safe_save_rds <- function(obj, path) {
  make_dir(dirname(path))
  saveRDS(obj, path)
  invisible(path)
}

safe_write_csv <- function(x, path) {
  make_dir(dirname(path))
  readr::write_csv(x, path)
  invisible(path)
}

add_hbsag_group <- function(meta, hbsag_col = "hbsag") {
  meta %>%
    mutate(
      hbsag_numeric = as.integer(.data[[hbsag_col]]),
      hbsag_group = ifelse(hbsag_numeric == 1, "HBV", "non_HBV")
    )
}

read_10x_any <- function(path) {
  if (grepl("\\.h5$", path, ignore.case = TRUE)) {
    return(Read10X_h5(path))
  }
  Read10X(data.dir = path)
}

percent_mito_pattern <- function(features) {
  if (any(grepl("^MT-", features))) return("^MT-")
  if (any(grepl("^mt-", features))) return("^mt-")
  "^MT-"
}

aggregate_counts_by_group <- function(counts, groups) {
  groups <- as.factor(groups)
  levs <- levels(groups)
  mats <- lapply(levs, function(g) {
    idx <- which(groups == g)
    if (length(idx) == 1) {
      counts[, idx, drop = FALSE]
    } else {
      Matrix::rowSums(counts[, idx, drop = FALSE])
    }
  })
  mat <- do.call(cbind, mats)
  if (is.null(dim(mat))) mat <- matrix(mat, ncol = length(levs))
  colnames(mat) <- levs
  rownames(mat) <- rownames(counts)
  as.matrix(mat)
}

bh_adjust <- function(p) p.adjust(p, method = "BH")
