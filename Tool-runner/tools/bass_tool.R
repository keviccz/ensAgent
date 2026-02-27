#!/usr/bin/env Rscript
# BASS Clustering Tool
# Environment: R

suppressPackageStartupMessages({
  library(BASS)
  library(Matrix)
  library(optparse)
})

# Parse arguments
option_list <- list(
  make_option(c("--data_path"), type="character", help="Path to data directory containing RData folder"),
  make_option(c("--sample_id"), type="character", help="Sample identifier"),
  make_option(c("--output_dir"), type="character", help="Output directory"),
  make_option(c("--n_clusters"), type="integer", default=7, help="Number of spatial domains [default %default]"),
  make_option(c("--C"), type="integer", default=20, help="Number of cell types [default %default]"),
  make_option(c("--random_seed"), type="integer", default=2023, help="Random seed [default %default]")
)

parser <- OptionParser(option_list=option_list)
args <- parse_args(parser)

cat("[BASS] Starting clustering for", args$sample_id, "...\n")

# Set random seed
set.seed(args$random_seed)

# Construct path to RData file
rdata_dir <- file.path(args$data_path, "RData")
input_data_path <- file.path(rdata_dir, "spatialLIBD_p1.RData")

# Check if file exists
if (!file.exists(input_data_path)) {
  stop(paste("[BASS] Error: Input data file not found:", input_data_path))
}

# Load data
cat("[BASS] Loading data from", input_data_path, "...\n")
load(input_data_path)

# Extract data for the specific sample
cat("[BASS] Extracting data for sample", args$sample_id, "...\n")
# Assuming data format: cntm, xym, infom (lists with sample IDs as names)

# Create BASS object
cat("[BASS] Creating BASS object (C=", args$C, ", R=", args$n_clusters, ")...\n", sep="")
BASS_obj <- createBASSObject(
  cntm, xym,
  C = args$C,
  R = args$n_clusters,
  beta_method = "SW",
  init_method = "mclust",
  nsample = 10000
)

# Preprocess
cat("[BASS] Preprocessing data...\n")
BASS_obj <- BASS.preprocess(
  BASS_obj,
  doLogNormalize = TRUE,
  geneSelect = "sparkx",
  nSE = 3000,
  doPCA = TRUE,
  scaleFeature = FALSE,
  nPC = 30
)

# Run BASS
cat("[BASS] Running BASS algorithm...\n")
BASS_obj <- BASS.run(BASS_obj)

# Post-process
cat("[BASS] Post-processing results...\n")
BASS_obj <- BASS.postprocess(BASS_obj)

# Extract labels
zlabels <- BASS_obj@results$z

# Save results for the specific sample
cat("[BASS] Saving results...\n")
dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)

# Find the index of the sample
sample_idx <- which(names(cntm) == args$sample_id)
if (length(sample_idx) == 0) {
  sample_idx <- 1  # Use first sample if not found
  cat("[BASS] Warning: Sample", args$sample_id, "not found, using first sample\n")
}

labs <- zlabels[[sample_idx]]
labs_chr <- as.character(labs)
labs_num <- sub("^C", "", labs_chr)

spot_id <- colnames(cntm[[sample_idx]])
res <- data.frame(
  spot_id = spot_id,
  BASS_domain = labs_num,
  check.names = FALSE
)

output_file <- file.path(args$output_dir, paste0("BASS_", args$sample_id, "_domain.csv"))
write.csv(res, output_file, row.names = FALSE)

cat("[BASS] Results saved to", output_file, "\n")
cat("[BASS] Completed successfully!\n")


