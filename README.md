# EnsAgent Tool-Runner: Ensemble Spatial Clustering Framework

A tool orchestration system for spatial transcriptomics analysis that integrates eight state-of-the-art clustering methods to generate robust spatial domain partitions.

## Overview

The Tool-Runner Agent is the first module of the EnsAgent framework, designed to execute multiple spatial clustering algorithms and standardize their outputs for ensemble-based analysis. This system addresses the challenge of inconsistent clustering results across different methods by providing:

- **Automated Tool Execution**: Parallel execution of 8 spatial clustering methods
- **Label Alignment**: IoU-based Hungarian matching algorithm for domain label standardization
- **Downstream Analysis**: Automated generation of differential expression, pathway enrichment, and visualizations

## Supported Methods

| Method | Environment | Reference | Input Format |
|--------|-------------|-----------|--------------|
| **IRIS** | R | [Nat Commun 2024](https://doi.org/10.1038/s41467-024-46638-4) | RDS files |
| **BASS** | R | [Nat Biotechnol 2022](https://doi.org/10.1038/s41587-022-01536-2) | RData |
| **DR-SC** | R | [Nat Commun 2023](https://doi.org/10.1038/s41467-023-35947-w) | Visium |
| **BayesSpace** | R | [Nat Biotechnol 2021](https://doi.org/10.1038/s41587-021-00935-2) | Visium |
| **SEDR** | Python | [Nat Commun 2022](https://doi.org/10.1038/s41467-022-29439-6) | Visium |
| **GraphST** | Python | [Nat Commun 2023](https://doi.org/10.1038/s41467-023-36796-3) | Visium |
| **STAGATE** | Python | [Nat Commun 2022](https://doi.org/10.1038/s41467-022-29439-6) | Visium |
| **stLearn** | Python | [bioRxiv 2020](https://doi.org/10.1101/2020.05.31.125658) | Visium |

## System Requirements

### Hardware
- Minimum 16GB RAM (32GB recommended)
- Multi-core CPU (8+ cores recommended)
- GPU optional (accelerates deep learning methods)

### Software
- Miniconda or Anaconda
- Windows 10+ / Linux / macOS
- Python 3.9+
- R 4.2+

### Environment Configuration

Due to package dependency conflicts, three separate conda environments are required:

- **R environment**: IRIS, BASS, DR-SC, BayesSpace
- **PY environment**: SEDR, GraphST, STAGATE
- **PY2 environment**: stLearn

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/EnsAgent-ToolRunner.git
cd EnsAgent-ToolRunner
```

### 2. Setup Conda Environments

```bash
# Create R environment
conda env create -f envs/R_environment.yml

# Create PY environment
conda env create -f envs/PY_environment.yml

# Create PY2 environment
conda env create -f envs/PY2_environment.yml
```

**Note**: Environment file creation may take 30-60 minutes depending on network speed.

### 3. Verify Installation

```bash
# Test R environment
conda activate R
Rscript --version
R --version
conda deactivate

# Test PY environment
conda activate PY
python --version
conda deactivate

# Test PY2 environment
conda activate PY2
python --version
conda deactivate
```

## Quick Start

### Prepare Input Data

Organize your 10X Visium data in the following structure:

```
data_directory/
├── filtered_feature_bc_matrix.h5    # Required
├── metadata.tsv                      # Optional (for reference annotations)
├── spatial/                          # Required
│   ├── tissue_hires_image.png
│   ├── tissue_lowres_image.png
│   ├── scalefactors_json.json
│   └── tissue_positions_list.csv
└── RData/                            # Required for IRIS/BASS
    ├── countList_spatial_LIBD.RDS
    ├── scRef_input_mainExample.RDS
    └── spatialLIBD_p1.RData
```

### Configure Parameters

Edit `configs/DLPFC_151507.yaml`:

```yaml
sample_id: "DLPFC_151507"
data_path: "/path/to/your/data"  # Update this path
output_dir: "./output/DLPFC_151507"
n_clusters: 7
random_seed: 2023
timeout: 7200  # Seconds per method
```

### Execute Pipeline

```bash
# Using configuration file
python orchestrator.py --config configs/DLPFC_151507.yaml

# Using command-line arguments
python orchestrator.py \
  --data_path /path/to/data \
  --sample_id DLPFC_151507 \
  --output_dir ./output/DLPFC_151507 \
  --n_clusters 7
```

### Output Structure

```
output/DLPFC_151507/
├── domains/                      # Raw clustering results (8 CSV files)
│   ├── IRIS_DLPFC_151507_domain.csv
│   ├── BASS_DLPFC_151507_domain.csv
│   └── ...
├── DLPFC_151507_aligned.h5ad     # Aligned AnnData object
├── spot/                         # Spot-level data (8 CSV files)
├── PICTURES/                     # Spatial visualizations (8 PNG files)
├── DEGs/                         # Differential expression results (8 CSV files)
├── PATHWAY/                      # Pathway enrichment results (8 CSV files)
└── tool_runner_report.json       # Execution summary
```

## Advanced Usage

### Selective Method Execution

```bash
python orchestrator.py \
  --config configs/DLPFC_151507.yaml \
  --methods SEDR GraphST STAGATE
```

### Custom Parameters

```yaml
# In config file
methods:
  - IRIS
  - SEDR
  - GraphST
  
min_success: 2  # Minimum successful methods to proceed
timeout: 7200   # 2 hours timeout per method

alignment:
  enable_flip_check: true
  flip_corr_threshold: 0.55
  enable_mean_order_fallback: true
  low_corr_threshold: 0.30

downstream:
  min_gene_pct: 0.1
  gene_sets: "KEGG_2021_Human"
```

## Methodology

### Phase 1: Tool Execution

Each clustering method is executed independently with standardized preprocessing:
- Gene filtering (min_cells=50, min_counts=10)
- Normalization and log-transformation
- Highly variable gene selection (n=2000-3000)
- PCA dimensionality reduction (n_components=200)

### Phase 2: Label Alignment

The alignment algorithm addresses label inconsistency through:

1. **Mode-based Mapping**: Each domain is assigned to the reference layer with maximum overlap
2. **Low-Correlation Fallback**: For Spearman ρ < 0.3, uses mean layer position ordering
3. **Direction Auto-Correction**: Automatically flips inverted domain orders (ρ < -0.55)
4. **Validation**: Ensures label continuity and domain count consistency

### Phase 3: Downstream Analysis

For each aligned clustering result:
- **Differential Expression**: Wilcoxon rank-sum test (adjusted p < 0.05, |log2FC| > 1)
- **Pathway Enrichment**: GSEA with configurable gene set databases
- **Spatial Visualization**: High-resolution PNG (300 DPI)
- **Structured Output**: CSV format with spot coordinates and domain labels

## File Formats

### Domain CSV

```csv
spot_id,METHOD_domain
AAACAAGTATCTCCCA-1,1
AAACACCAATAACTGC-1,3
...
```

### Spot CSV

```csv
spot_id,x,y,in_tissue,n_genes,spatial_domain
AAACAAGTATCTCCCA-1,100.5,200.3,True,1500,1
...
```

### DEG CSV

```csv
domain,names,logfoldchanges,pvals_adj
1,SNAP25,2.34,1.2e-45
...
```

### Pathway CSV

```csv
Domain,Term,NES,NOM p-val,Lead_genes
1,Neuroactive ligand-receptor,2.45,0.001,GRIA1;GRIN2A
...
```

## Troubleshooting

### Method Fails

Check `tool_runner_report.json` for detailed error information. The pipeline continues if at least `min_success` methods complete successfully.

### Memory Errors

- Close memory-intensive applications
- Reduce `n_top_genes` in method configurations
- Process fewer methods simultaneously

### Environment Issues

Verify environment activation:
```bash
conda activate R
which Rscript  # Should point to conda environment
```

### Path Issues

Ensure paths do not contain special characters or spaces. Use absolute paths when possible.

## Performance Optimization

- **GPU Acceleration**: Install CUDA-enabled PyTorch for SEDR, GraphST, and STAGATE
- **Parallel Execution**: Methods run sequentially by default; modify orchestrator for parallel execution
- **Memory Management**: Adjust `n_top_genes` based on available RAM

## Contributing

Contributions are welcome. Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## Acknowledgments

We thank the authors of IRIS, BASS, DR-SC, BayesSpace, SEDR, GraphST, STAGATE, and stLearn for making their methods publicly available. This tool integrates and standardizes these excellent approaches for ensemble-based spatial analysis.


