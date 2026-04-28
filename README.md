# EnsAgent

<a id="english"></a>

<div align="center">
  <h3>Ensemble Multi-Agent Framework for Spatial Transcriptomics</h3>
  <p>
    <b>English</b> | <a href="#中文说明">中文</a>
  </p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/R-4.2%2B-276DC3?logo=r&logoColor=white" alt="R 4.2+">
    <img src="https://img.shields.io/badge/TypeScript-5%2B-3178C6?logo=typescript&logoColor=white" alt="TypeScript 5+">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
    <img src="https://img.shields.io/badge/npm-required-CB3837?logo=npm&logoColor=white" alt="npm required">
    <img src="https://img.shields.io/badge/Next.js-14.2.5-000000?logo=nextdotjs&logoColor=white" alt="Next.js 14.2.5">
    <img src="https://img.shields.io/badge/FastAPI-supported-009688?logo=fastapi&logoColor=white" alt="FastAPI supported">
  </p>
  <p>
    <a href="#quick-start">Quick Start</a> ·
    <a href="#method-code-and-data-availability">Method & Data</a> ·
    <a href="#core-commands">Commands</a> ·
    <a href="#demo-video">Demo Video</a>
  </p>
  <img src="Readme_example/Diagram.png" alt="EnsAgent pipeline diagram" width="900">
  <br>
  <br>
  <a id="demo-video"></a>
  <video src="Readme_example/Frontend-use.mp4" controls muted width="720"></video>
</div>


https://github.com/user-attachments/assets/4b26e417-85c9-4d6c-8dc2-84326afd8a17


EnsAgent is an ensemble multi-agent framework for spatial transcriptomics analysis on 10x Visium data. It runs eight spatial clustering methods, scores domain-level evidence with LLM/VLM agents, builds a consensus BEST result, and annotates spatial domains through a multi-agent workflow.

## Highlights

- Eight wrapped spatial clustering methods: IRIS, BASS, DR-SC, BayesSpace, SEDR, GraphST, STAGATE, and stLearn.
- Supported local app stack: FastAPI backend plus Next.js frontend, launched with `python start.py`.
- Reproducible outputs for clustering labels, spatial plots, DEGs, pathway summaries, score matrices, BEST labels, and final annotations.
- Example data under `example_data/` is included for repository demos and frontend inspection.

## Quick Start

### 1. Create Environments

```bash
mamba env create -f environment.yml
mamba activate ensagent
python -m pip install -r requirements.txt

mamba env create -f envs/R_environment.yml
mamba env create -f envs/PY_environment.yml
mamba env create -f envs/PY2_environment.yml
```

### 2. Configure

```bash
cp pipeline_config.example.yaml pipeline_config.yaml
```

Edit `pipeline_config.yaml` and set at least:

```yaml
data_path: "Tool-runner/151507"
sample_id: "DLPFC_151507"
n_clusters: 7
```

Recommended provider environment variables:

```bash
export ENSAGENT_API_PROVIDER="openai"
export ENSAGENT_API_KEY="sk-..."
export ENSAGENT_API_MODEL="gpt-4o"
```

Azure-compatible aliases are also supported: `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, and `AZURE_OPENAI_API_VERSION`.

### 3. Run

```bash
python start.py
```

The launcher starts:

- FastAPI backend: `http://localhost:8000`
- Next.js frontend: `http://localhost:3000`

Run the full pipeline from the command line:

```bash
python endtoend.py
python endtoend.py --data_path "<VISIUM_DIR>" --sample_id "DLPFC_151507" --n_clusters 7
```

## Method Code And Data Availability

EnsAgent wraps public method implementations. The table below lists the upstream implementation and the input data expected by the current wrapper.

| Method | Upstream availability | Data expected by EnsAgent |
| ------ | --------------------- | ------------------------- |
| IRIS | [GitHub: YingMa0107/IRIS](https://github.com/YingMa0107/IRIS) | `RData/countList_spatial_LIBD.RDS` plus `RData/scRef_input_mainExample.RDS` for spatial count/location data and matched scRNA-seq reference |
| BASS | [GitHub: zhengli09/BASS](https://github.com/zhengli09/BASS) | `RData/spatialLIBD_p1.RData` containing BASS-ready count and coordinate lists |
| DR-SC | [GitHub: feiyoung/DR.SC](https://github.com/feiyoung/DR.SC) | 10x Visium directory loadable with Seurat `Load10X_Spatial` |
| BayesSpace | [Bioconductor](https://bioconductor.org/packages/release/bioc/html/BayesSpace.html), [GitHub: edward130603/BayesSpace](https://github.com/edward130603/BayesSpace) | 10x Visium directory loadable with BayesSpace `readVisium` |
| SEDR | [GitHub: JinmiaoChenLab/SEDR](https://github.com/JinmiaoChenLab/SEDR) | 10x Visium directory loadable with Scanpy `read_visium` |
| GraphST | [GitHub: JinmiaoChenLab/GraphST](https://github.com/JinmiaoChenLab/GraphST) | 10x Visium directory loadable with Scanpy `read_visium` |
| STAGATE | [GitHub: QIFEIDKN/STAGATE_pyG](https://github.com/QIFEIDKN/STAGATE_pyG) | 10x Visium directory loadable with Scanpy `read_visium` |
| stLearn | [GitHub: BiomedicalMachineLearning/stLearn](https://github.com/BiomedicalMachineLearning/stLearn) | 10x Visium directory with `spatial/` image files, loadable with stLearn/Scanpy Visium readers |

## Input Data

```text
data_directory/
├── filtered_feature_bc_matrix.h5
├── metadata.tsv
├── spatial/
│   ├── tissue_hires_image.png
│   ├── tissue_lowres_image.png
│   ├── scalefactors_json.json
│   └── tissue_positions_list.csv
└── RData/
    ├── countList_spatial_LIBD.RDS
    ├── scRef_input_mainExample.RDS
    └── spatialLIBD_p1.RData
```

`RData/` is required only for wrappers that need original method-specific R objects, especially IRIS and BASS. Other methods read the Visium directory directly.

## Outputs

```text
output/
├── tool_runner/<sample_id>/
├── best/<sample_id>/
└── scoring/output/<sample_id>/
```

Key output files include `BEST_<sample_id>_spot.csv`, `BEST_<sample_id>_DEGs.csv`, `BEST_<sample_id>_PATHWAY.csv`, `<sample_id>_result.png`, and `annotation_output/domain_annotations.json`.

## Core Commands

```bash
python Tool-runner/orchestrator.py --config Tool-runner/configs/DLPFC_151507.yaml
cd scoring && python scoring.py
python tools/health_check.py
python -m unittest discover -s tests -v
cd frontend && npm run build
```

## Requirements

- Python 3.10+
- R 4.2+
- Miniforge/Mamba or Conda
- 16 GB RAM minimum; 32 GB recommended
- GPU optional for SEDR, GraphST, and STAGATE

---

## 中文说明

<p align="center">
  <a href="#english">English</a> | <b>中文</b>
</p>

EnsAgent 是一个面向 10x Visium 空间转录组数据的集成式多智能体分析框架。它统一运行 8 种空间聚类方法，使用 LLM/VLM 智能体评估 domain 证据，构建 BEST 共识结果，并通过多智能体流程完成空间 domain 注释。

### 主要特点

- 集成 8 种空间聚类方法：IRIS、BASS、DR-SC、BayesSpace、SEDR、GraphST、STAGATE、stLearn。
- 当前支持的本地应用栈为 FastAPI 后端 + Next.js 前端，通过 `python start.py` 启动。
- 输出包括聚类标签、空间图、DEG、pathway、评分矩阵、BEST 标签和最终注释。
- `example_data/` 中包含可用于仓库展示和前端查看的示例数据。

### 快速运行

```bash
mamba env create -f environment.yml
mamba activate ensagent
python -m pip install -r requirements.txt

mamba env create -f envs/R_environment.yml
mamba env create -f envs/PY_environment.yml
mamba env create -f envs/PY2_environment.yml

cp pipeline_config.example.yaml pipeline_config.yaml
python start.py
```

启动后访问：

- FastAPI 后端：`http://localhost:8000`
- Next.js 前端：`http://localhost:3000`

### 数据与方法

方法实现与数据要求见上方英文表格。IRIS 和 BASS 需要额外的 `RData/` 输入对象，其余方法主要读取标准 10x Visium 目录。
