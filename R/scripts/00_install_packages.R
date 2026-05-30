cran_packages <- c(
  "Seurat",
  "dplyr",
  "tidyr",
  "Matrix",
  "readr",
  "ggplot2",
  "yaml",
  "harmony",
  "patchwork",
  "pheatmap",
  "stringr",
  "tibble",
  "BiocManager"
)

missing_cran <- cran_packages[!vapply(cran_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_cran) > 0) {
  install.packages(missing_cran, repos = "https://cloud.r-project.org")
}

bioc_packages <- c("edgeR", "clusterProfiler", "org.Hs.eg.db")
missing_bioc <- bioc_packages[!vapply(bioc_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_bioc) > 0) {
  BiocManager::install(missing_bioc, ask = FALSE, update = FALSE)
}
