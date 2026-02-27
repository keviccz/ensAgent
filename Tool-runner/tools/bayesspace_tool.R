#!/usr/bin/env Rscript
# BayesSpace Clustering Tool
# Environment: R

suppressPackageStartupMessages({
  library(BayesSpace)
  library(SingleCellExperiment)
  library(optparse)
})

# Parse arguments
option_list <- list(
  make_option(c("--data_path"), type="character", help="Path to Visium data directory"),
  make_option(c("--sample_id"), type="character", help="Sample identifier"),
  make_option(c("--output_dir"), type="character", help="Output directory"),
  make_option(c("--n_clusters"), type="integer", default=7, help="Number of clusters [default %default]"),
  make_option(c("--nrep"), type="integer", default=50000, help="Number of MCMC iterations [default %default]"),
  make_option(c("--random_seed"), type="integer", default=2023, help="Random seed [default %default]")
)

parser <- OptionParser(option_list=option_list)
args <- parse_args(parser)

cat("[BayesSpace] Starting clustering for", args$sample_id, "...\n")

# Set random seed
set.seed(args$random_seed)

# Load data
cat("[BayesSpace] Loading data from", args$data_path, "...\n")
sce <- readVisium(args$data_path)

# Preprocessing
cat("[BayesSpace] Preprocessing...\n")
sce <- spatialPreprocess(sce, platform="Visium", n.PCs=50, n.HVGs=2000, log.normalize=TRUE)

# Run BayesSpace
cat("[BayesSpace] Running BayesSpace (q=", args$n_clusters, ", nrep=", args$nrep, ")...\n", sep="")
sce <- spatialCluster(sce, q=args$n_clusters, platform="Visium", nrep=args$nrep, burn.in=10000)

# Extract results
labels <- sce$spatial.cluster

# Save results
cat("[BayesSpace] Saving results...\n")
dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)

res <- data.frame(
  spot_id = colnames(sce),
  BayesSpace_domain = as.integer(labels),
  check.names = FALSE
)

output_file <- file.path(args$output_dir, paste0("BayesSpace_", args$sample_id, "_domain.csv"))
write.csv(res, output_file, row.names = FALSE)

cat("[BayesSpace] Results saved to", output_file, "\n")
cat("[BayesSpace] Completed successfully!\n")


