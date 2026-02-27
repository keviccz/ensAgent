#!/usr/bin/env Rscript
# IRIS Clustering Tool
# Environment: R

suppressPackageStartupMessages({
  library(IRIS)
  library(Matrix)
  library(optparse)
})

# Parse arguments
option_list <- list(
  make_option(c("--data_path"), type="character", help="Path to data directory containing RData folder"),
  make_option(c("--sample_id"), type="character", help="Sample identifier"),
  make_option(c("--output_dir"), type="character", help="Output directory"),
  make_option(c("--n_clusters"), type="integer", default=7, help="Number of clusters [default %default]"),
  make_option(c("--random_seed"), type="integer", default=2023, help="Random seed [default %default]")
)

parser <- OptionParser(option_list=option_list)
args <- parse_args(parser)

cat("[IRIS] Starting clustering for", args$sample_id, "...\n")

# Set random seed
set.seed(args$random_seed)

# Construct paths to RDS files
rdata_dir <- file.path(args$data_path, "RData")
spatial_data_path <- file.path(rdata_dir, "countList_spatial_LIBD.RDS")
sc_data_path <- file.path(rdata_dir, "scRef_input_mainExample.RDS")

# Check if files exist
if (!file.exists(spatial_data_path)) {
  stop(paste("[IRIS] Error: Spatial data file not found:", spatial_data_path))
}
if (!file.exists(sc_data_path)) {
  stop(paste("[IRIS] Error: scRNA-seq reference file not found:", sc_data_path))
}

# Load spatial data
cat("[IRIS] Loading spatial data from", spatial_data_path, "...\n")
sp_in <- readRDS(spatial_data_path)
spatial_countMat_list <- sp_in$spatial_countMat_list
spatial_location_list <- sp_in$spatial_location_list

# Load scRNA-seq reference
cat("[IRIS] Loading scRNA-seq reference from", sc_data_path, "...\n")
sc_in <- readRDS(sc_data_path)
sc_count <- sc_in$sc_count
sc_meta <- sc_in$sc_meta
ct.varname <- sc_in$ct.varname
sample.varname <- sc_in$sample.varname

# Create IRIS object
cat("[IRIS] Creating IRIS object...\n")
IRIS_object <- createIRISObject(
  spatial_countMat_list = spatial_countMat_list,
  spatial_location_list = spatial_location_list,
  sc_count = sc_count,
  sc_meta = sc_meta,
  ct.varname = ct.varname,
  sample.varname = sample.varname,
  minCountGene = 100,
  minCountSpot = 5
)

# Run IRIS spatial domain detection
cat("[IRIS] Running spatial domain detection (k=", args$n_clusters, ")...\n", sep="")
IRIS_object <- IRIS_spatial(IRIS_object, numCluster = args$n_clusters)

# Extract and save results
cat("[IRIS] Saving results...\n")
dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)

dom_df <- IRIS_object@spatialDomain
dom_out <- data.frame(
  spot_id = rownames(dom_df),
  IRIS_domain = as.integer(as.character(dom_df[, "IRIS_domain"])) + 1,
  row.names = NULL,
  check.names = FALSE
)

output_file <- file.path(args$output_dir, paste0("IRIS_", args$sample_id, "_domain.csv"))
write.csv(dom_out, output_file, row.names = FALSE)

cat("[IRIS] Results saved to", output_file, "\n")
cat("[IRIS] Completed successfully!\n")


