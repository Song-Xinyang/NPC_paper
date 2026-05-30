marker_sets <- list(
  `Epithelial cell` = c("EPCAM", "KRT8", "KRT18", "KRT19"),
  `T cell` = c("CD3D", "CD3E", "TRAC"),
  `CD4+ T` = c("CD3D", "CD3E", "CD4", "IL7R"),
  `CD8+ T` = c("CD3D", "CD3E", "CD8A", "CD8B", "NKG7"),
  `Proliferating T` = c("CD3D", "MKI67", "TOP2A", "STMN1"),
  `NK cell` = c("NKG7", "GNLY", "KLRD1", "FCGR3A"),
  `B cell` = c("MS4A1", "CD79A", "CD79B"),
  `Plasma cell` = c("MZB1", "JCHAIN", "IGHG1"),
  Macrophage = c("LYZ", "CD68", "C1QA", "C1QB"),
  `Dendritic cell` = c("FCER1A", "CLEC10A", "LILRA4", "ITGAX"),
  `Mast cell` = c("TPSAB1", "TPSB2", "KIT"),
  `Endothelial cell` = c("PECAM1", "VWF", "KDR"),
  Fibroblast = c("COL1A1", "COL1A2", "DCN", "LUM")
)

exhaustion_genes_default <- c("PDCD1", "CTLA4", "HAVCR2", "LAG3", "TIGIT", "BTLA", "KLRG1", "TOX", "EOMES", "CXCL13")
cytotoxic_genes_default <- c("IFNG", "GZMB", "PRF1", "NKG7")
immune_genes_of_interest_default <- c(exhaustion_genes_default, cytotoxic_genes_default)
