#!/usr/bin/env Rscript
# DR-SC Clustering Tool
# Environment: R

suppressPackageStartupMessages({
  library(DR.SC)
  library(Seurat)
  library(optparse)
})

# Parse arguments
option_list <- list(
  make_option(c("--data_path"), type="character", help="Path to Visium data directory"),
  make_option(c("--sample_id"), type="character", help="Sample identifier"),
  make_option(c("--output_dir"), type="character", help="Output directory"),
  make_option(c("--n_clusters"), type="integer", default=7, help="Number of clusters [default %default]"),
  make_option(c("--q"), type="integer", default=15, help="Number of low-dimensional embeddings [default %default]"),
  make_option(c("--random_seed"), type="integer", default=2023, help="Random seed [default %default]")
)

parser <- OptionParser(option_list=option_list)
args <- parse_args(parser)

cat("[DR-SC] Starting clustering for", args$sample_id, "...\n")

# Set random seed
set.seed(args$random_seed)

# Load data
cat("[DR-SC] Loading data from", args$data_path, "...\n")
seu <- Load10X_Spatial(args$data_path)

# Preprocessing
cat("[DR-SC] Preprocessing...\n")
seu <- NormalizeData(seu)
seu <- FindVariableFeatures(seu, nfeatures = 2000)
seu <- ScaleData(seu)

# Run DR-SC
cat("[DR-SC] Running DR-SC (K=", args$n_clusters, ", q=", args$q, ")...\n", sep="")
seu <- DR.SC(seu, K = args$n_clusters, q = args$q)

# Extract results
labels <- seu@meta.data$spatial.drsc.cluster

# Save results
cat("[DR-SC] Saving results...\n")
dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)

res <- data.frame(
  spot_id = colnames(seu),
  `DR-SC_domain` = as.integer(labels),
  check.names = FALSE
)

output_file <- file.path(args$output_dir, paste0("DR-SC_", args$sample_id, "_domain.csv"))
write.csv(res, output_file, row.names = FALSE)

cat("[DR-SC] Results saved to", output_file, "\n")
cat("[DR-SC] Completed successfully!\n")


