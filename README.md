# Tool-Runner Agent: Spatial Transcriptomics Clustering Ensemble

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.2%2B-276DC3.svg)](https://www.r-project.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A multi-environment tool orchestration system for robust spatial domain identification in spatial transcriptomics data.**

Part of the **ensAgent** framework for ensemble-based annotation in spatial transcriptomics.

---

## 📖 Overview

This repository contains the **Tool-Runner Agent**, which orchestrates 8 state-of-the-art spatial clustering methods, aligns their outputs, and generates comprehensive downstream analyses. This is the first stage of the ensAgent pipeline described in our paper:

> **ensAgent: a tool-ensemble multiple Agent system for robust annotation in spatial transcriptomics**

### Pipeline Stages

1. **Tool Execution**: Runs 8 clustering methods in parallel
2. **Label Alignment**: Aligns domain labels using IoU-based matching
3. **Downstream Analysis**: Generates DEGs, pathway enrichment, visualizations, and spot files

### Supported Methods

| Method | Environment | Description |
|--------|-------------|-------------|
| **IRIS** | R | Integrative Robust Inference of Spatial domains |
| **BASS** | R | Bayesian Analytics for Spatial Structures |
| **DR-SC** | R | Dimension Reduction with Spatial Clustering |
| **BayesSpace** | R | Bayesian spatial clustering |
| **SEDR** | PY | Spatial Embedding with Dual Regularization |
| **GraphST** | PY | Graph-based Spatial Transcriptomics |
| **STAGATE** | PY | Spatial Transcriptomics Analysis with Graph Attention |
| **stLearn** | PY2 | Spatial Learning framework |

---

## 🚀 Quick Start

### Prerequisites

- **Miniconda or Anaconda** installed
- **Windows 10+** (tested environment)
- **At least 16GB RAM** recommended
- **GPU** optional (speeds up GraphST, SEDR, STAGATE)

### Installation

#### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/ensAgent-ToolRunner.git
cd ensAgent-ToolRunner
```

#### Step 2: Create conda environments

Due to package conflicts, **3 separate environments** are required:

```bash
# R environment (for IRIS, BASS, DR-SC, BayesSpace)
conda env create -f envs/R_environment.yml

# PY environment (for SEDR, GraphST, STAGATE)
conda env create -f envs/PY_environment.yml

# PY2 environment (for stLearn)
conda env create -f envs/PY2_environment.yml
```

**Note**: Environment files are located in the parent directory (`../R_environment.yml`, `../PY_environment.yml`, `../PY2_environment.yml`). Please manually copy them to the `envs/` folder if not present:

```bash
# On Windows PowerShell
Copy-Item ..\R_environment.yml envs\
Copy-Item ..\PY_environment.yml envs\
Copy-Item ..\PY2_environment.yml envs\
```

#### Step 3: Verify installation

```bash
conda activate R
Rscript --version
conda deactivate

conda activate PY
python --version
conda deactivate

conda activate PY2
python --version
conda deactivate
```

---

## 📊 Usage

### Demo: DLPFC_151507 Sample

We provide a complete example for the DLPFC (Dorsolateral Prefrontal Cortex) dataset.

#### Step 1: Prepare your data

Organize your Visium data as follows:

```
D:/zuo ye/YAN0/zhangmingshen/DATA/DLPFC/151507/
├── filtered_feature_bc_matrix.h5
├── metadata.tsv (optional, for reference annotations)
└── spatial/
    ├── tissue_hires_image.png
    ├── tissue_lowres_image.png
    ├── scalefactors_json.json
    └── tissue_positions_list.csv
```

#### Step 2: Configure the run

Edit `configs/DLPFC_151507.yaml`:

```yaml
sample_id: "DLPFC_151507"
data_path: "D:/path/to/your/data/151507"  # UPDATE THIS
output_dir: "./output/DLPFC_151507"
n_clusters: 7
```

#### Step 3: Run the pipeline

```bash
# Using config file (recommended)
python orchestrator.py --config configs/DLPFC_151507.yaml

# Or using command-line arguments
python orchestrator.py \
  --data_path D:/path/to/data/151507 \
  --sample_id DLPFC_151507 \
  --output_dir ./output/DLPFC_151507 \
  --n_clusters 7
```

#### Step 4: Check results

```
output/DLPFC_151507/
├── domains/                      # Raw clustering results
│   ├── IRIS_DLPFC_151507_domain.csv
│   ├── BASS_DLPFC_151507_domain.csv
│   └── ... (8 files total)
├── DLPFC_151507_aligned.h5ad     # Aligned AnnData object
├── spot/                         # Spot-level data
│   ├── IRIS_domain_DLPFC_151507_spot.csv
│   └── ... (8 files)
├── PICTURES/                     # Spatial visualizations
│   ├── IRIS_domain_DLPFC_151507.png
│   └── ... (8 files)
├── DEGs/                         # Differentially expressed genes
│   ├── IRIS_domain_DLPFC_151507_DEGs.csv
│   └── ... (8 files)
├── PATHWAY/                      # Pathway enrichment results
│   ├── IRIS_domain_DLPFC_151507_PATHWAY.csv
│   └── ... (8 files)
└── tool_runner_report.json       # Execution summary
```

---

## 🔧 Advanced Configuration

### Custom Method Selection

Run only specific methods:

```bash
python orchestrator.py \
  --config configs/DLPFC_151507.yaml \
  --methods IRIS STAGATE GraphST SEDR
```

### Alignment Parameters

Fine-tune alignment behavior in your config:

```yaml
alignment:
  enable_flip_check: true           # Auto-flip inverted domain orders
  flip_corr_threshold: 0.55         # Correlation threshold for flipping
  enable_mean_order_fallback: true  # Use mean layer ordering for low correlation
  low_corr_threshold: 0.30          # Threshold for low correlation
```

### Downstream Analysis Parameters

```yaml
downstream:
  min_gene_pct: 0.1                 # Min % of cells for gene filtering
  gene_sets: "KEGG_2021_Human"      # Gene set database for enrichment
```

---

## 📁 Project Structure

```
ensAgent-ToolRunner/
├── orchestrator.py           # Main controller
├── configs/                  # Configuration files
│   └── DLPFC_151507.yaml
├── tools/                    # Individual clustering tools
│   ├── iris_tool.R
│   ├── bass_tool.R
│   ├── drsc_tool.R
│   ├── bayesspace_tool.R
│   ├── sedr_tool.py
│   ├── graphst_tool.py
│   ├── stagate_tool.py
│   └── stlearn_tool.py
├── postprocess/              # Downstream analysis scripts
│   ├── align_labels.py
│   ├── generate_degs.py
│   ├── generate_spots.py
│   ├── generate_pathways.py
│   └── generate_pictures.py
├── envs/                     # Conda environment files
│   ├── R_environment.yml
│   ├── PY_environment.yml
│   └── PY2_environment.yml
├── examples/                 # Example data and notebooks
└── README.md
```

---

## 🔬 Methodology

### Phase 1: Tool Execution

Each method independently clusters the spatial transcriptomics data. Tools are executed with:
- Standardized preprocessing (normalization, HVG selection, PCA)
- Consistent random seeds for reproducibility
- Automatic error handling and retry logic

### Phase 2: Label Alignment

Domain labels from different methods are aligned using an IoU-based Hungarian matching algorithm:

1. **Mode-based mapping**: Each domain is mapped to the most common reference layer
2. **Low-correlation fallback**: For poor alignments (Spearman ρ < 0.3), use mean layer ordering
3. **Auto-flip detection**: Automatically correct inverted domain orders (ρ < -0.55)
4. **Validation**: Ensure label continuity and domain count consistency

### Phase 3: Downstream Analysis

For each aligned method:
- **DEGs**: Wilcoxon rank-sum test (adj. p < 0.05, |log2FC| > 1)
- **Pathway Enrichment**: GSEA with KEGG/GO databases
- **Visualizations**: High-resolution spatial domain plots
- **Spot Files**: Structured CSV with coordinates and domain labels

---

## 📊 Output Formats

### Domain Files (CSV)

```csv
spot_id,{METHOD}_domain
AAACAAGTATCTCCCA-1,1
AAACACCAATAACTGC-1,3
...
```

### Spot Files (CSV)

```csv
spot_id,x,y,in_tissue,n_genes,spatial_domain
AAACAAGTATCTCCCA-1,100.5,200.3,True,1500,1
...
```

### DEG Files (CSV)

```csv
domain,names,logfoldchanges,pvals_adj
1,SNAP25,2.34,1.2e-45
1,SYT1,2.01,3.4e-38
...
```

### Pathway Files (CSV)

```csv
Domain,Term,NES,NOM p-val,Lead_genes
1,Neuroactive ligand-receptor,2.45,0.001,GRIA1;GRIN2A;...
...
```

---

## ❓ Troubleshooting

### Common Issues

**1. Tool fails with "module not found"**
- Ensure you're in the correct conda environment
- Check `conda list` for missing packages
- Try re-creating the environment

**2. Alignment fails with "domain count mismatch"**
- Some methods may produce fewer/more domains than expected
- Check `tool_runner_report.json` for failed methods
- Lower `min_success` in config if acceptable

**3. R scripts fail on Windows**
- Ensure Rscript is in your PATH
- Check R_HOME environment variable
- Try running R tools manually first

**4. Memory errors**
- Reduce `n_top_genes` in preprocessing
- Process fewer methods simultaneously
- Use a machine with more RAM

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/yourusername/ensAgent-ToolRunner/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ensAgent-ToolRunner/discussions)
- **Email**: your.email@example.com

---

## 📄 Citation

If you use this tool in your research, please cite:

```bibtex
@article{ensagent2026,
  title={ensAgent: a tool-ensemble multiple Agent system for robust annotation in spatial transcriptomics},
  author={Your Name et al.},
  journal={TBD},
  year={2026}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This tool integrates the following excellent methods:
- [IRIS](https://github.com/LilithYe/IRIS)
- [BASS](https://github.com/zhengli09/BASS)
- [DR-SC](https://github.com/feiyoung/DR.SC)
- [BayesSpace](https://github.com/edward130603/BayesSpace)
- [SEDR](https://github.com/JinmiaoChenLab/SEDR)
- [GraphST](https://github.com/JinmiaoChenLab/GraphST)
- [STAGATE](https://github.com/zhanglabtools/STAGATE)
- [stLearn](https://github.com/BiomedicalMachineLearning/stLearn)

We thank the authors for making their code publicly available.

---

**For the complete ensAgent pipeline (including annotation stages), see the main repository.**

