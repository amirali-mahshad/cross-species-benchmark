suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(optparse)
})

option_list <- list(
  make_option("--counts_mtx", type = "character"),
  make_option("--obs_csv", type = "character"),
  make_option("--genes_tsv", type = "character"),
  make_option("--output_csv", type = "character"),

  make_option("--method", type = "character", default = "cca"),
  make_option("--correction_mode", type = "character", default = "species"),

  make_option("--species_key", type = "character", default = "species"),
  make_option("--technical_key", type = "character", default = "sample"),

  make_option("--n_pcs", type = "integer", default = 20),
  make_option("--seed", type = "integer", default = 0)
)

opt <- parse_args(OptionParser(option_list = option_list))

set.seed(opt$seed)

message("======================================")
message("Running Seurat integration")
message("Method: ", opt$method)
message("Correction mode: ", opt$correction_mode)
message("n_pcs: ", opt$n_pcs)
message("seed: ", opt$seed)
message("======================================")

# ------------------------------------------------------------
# Read input files
# ------------------------------------------------------------

counts <- readMM(opt$counts_mtx)
genes <- readLines(opt$genes_tsv)
obs <- read.csv(opt$obs_csv, check.names = FALSE, stringsAsFactors = FALSE)

if (!"cell_id" %in% colnames(obs)) {
  stop("obs_csv must contain a column named cell_id.")
}

rownames(counts) <- genes
colnames(counts) <- obs$cell_id
rownames(obs) <- obs$cell_id

# ------------------------------------------------------------
# Create Seurat object
# ------------------------------------------------------------

obj <- CreateSeuratObject(
  counts = counts,
  meta.data = obs,
  project = "cross_species_benchmark"
)

# ------------------------------------------------------------
# Decide integration key
# ------------------------------------------------------------

if (opt$correction_mode == "species") {

  integration_key <- opt$species_key

} else if (opt$correction_mode == "technical") {

  integration_key <- opt$technical_key

} else if (opt$correction_mode == "both") {

  # Seurat CCA/RPCA does not use multiple covariates in the same way as scVI.
  # This creates a composite batch key.
  obj$seurat_composite_key <- paste(
    obj[[opt$technical_key]][, 1],
    obj[[opt$species_key]][, 1],
    sep = "__"
  )
  integration_key <- "seurat_composite_key"

} else {
  stop("correction_mode must be one of: species, technical, both.")
}

if (!integration_key %in% colnames(obj@meta.data)) {
  stop(paste0("Missing integration key in metadata: ", integration_key))
}

message("Integration key used by Seurat: ", integration_key)

# ------------------------------------------------------------
# Split object
# ------------------------------------------------------------

obj_list <- SplitObject(obj, split.by = integration_key)

if (length(obj_list) < 2) {
  stop("Seurat integration needs at least two groups after splitting.")
}

features <- rownames(obj)

message("Number of split objects: ", length(obj_list))
message("Number of integration features: ", length(features))

# ------------------------------------------------------------
# Normalize each split object
# ------------------------------------------------------------

for (i in seq_along(obj_list)) {
  obj_list[[i]] <- NormalizeData(
    obj_list[[i]],
    verbose = FALSE
  )
}

# ------------------------------------------------------------
# For RPCA, each object needs PCA before anchor finding
# ------------------------------------------------------------

if (opt$method == "rpca") {

  for (i in seq_along(obj_list)) {
    obj_list[[i]] <- ScaleData(
      obj_list[[i]],
      features = features,
      verbose = FALSE
    )

    obj_list[[i]] <- RunPCA(
      obj_list[[i]],
      features = features,
      npcs = opt$n_pcs,
      verbose = FALSE
    )
  }

  reduction_method <- "rpca"

} else if (opt$method == "cca") {

  reduction_method <- "cca"

} else {
  stop("method must be one of: cca, rpca.")
}

# ------------------------------------------------------------
# Find anchors and integrate
# ------------------------------------------------------------

anchors <- FindIntegrationAnchors(
  object.list = obj_list,
  anchor.features = features,
  reduction = reduction_method,
  dims = 1:opt$n_pcs,
  verbose = FALSE
)

integrated <- IntegrateData(
  anchorset = anchors,
  dims = 1:opt$n_pcs,
  verbose = FALSE
)

# ------------------------------------------------------------
# PCA on integrated assay
# ------------------------------------------------------------

DefaultAssay(integrated) <- "integrated"

integrated <- ScaleData(
  integrated,
  verbose = FALSE
)

integrated <- RunPCA(
  integrated,
  npcs = opt$n_pcs,
  verbose = FALSE
)

emb <- Embeddings(integrated, reduction = "pca")

emb_df <- as.data.frame(emb)
emb_df$cell_id <- rownames(emb_df)

# Put cell_id first
emb_df <- emb_df[, c("cell_id", setdiff(colnames(emb_df), "cell_id"))]

write.csv(
  emb_df,
  file = opt$output_csv,
  row.names = FALSE,
  quote = FALSE
)

message("Saved embedding to: ", opt$output_csv)
