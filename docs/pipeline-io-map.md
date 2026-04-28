# Pipeline I/O Map

Updated: 2026-04-10

## Overview

The recommended EnsAgent data flow is:

1. Stage A runs clustering methods through Tool-Runner.
2. Stage A outputs are staged into scoring inputs.
3. Stage B scores method/domain evidence and builds consensus matrices.
4. Stage C builds BEST artifacts from scores and labels.
5. Stage D annotates BEST domains.
6. API and frontend routes read sample-scoped outputs.

## Stage A: Tool-Runner

Entrypoint:

- `Tool-runner/orchestrator.py`
- Wrapper: `ensagent_tools/tool_runner.py`

Inputs:

- `data_path`
- `sample_id`
- `methods`
- `n_clusters`
- resolved Conda/Mamba executable and method environment names

Outputs:

- `output/tool_runner/<sample_id>/domains/`
- `output/tool_runner/<sample_id>/spot/`
- `output/tool_runner/<sample_id>/DEGs/`
- `output/tool_runner/<sample_id>/PATHWAY/`
- `output/tool_runner/<sample_id>/PICTURES/`
- `output/tool_runner/<sample_id>/tool_runner_report.json`

Downstream consumers:

- Stage B staging and scoring
- API/frontend visualizations

## Stage A to Stage B Staging

Entrypoint:

- `ensagent_tools/scoring.py`

Purpose:

- Copy and normalize Stage A `spot/`, `DEGs/`, and `PATHWAY/` files into `scoring/input/`.

Inputs:

- `tool_output_dir`
- `csv_path`
- `sample_id`

Output:

- `scoring/input/*_<sample_id>_spot.csv`
- `scoring/input/*_<sample_id>_DEGs.csv`
- `scoring/input/*_<sample_id>_PATHWAY.csv`

## Stage B: Scoring

Entrypoint:

- `scoring/scoring.py`
- Wrapper: `ensagent_tools/scoring.py`

Inputs:

- `scoring/input/`
- provider/API configuration
- optional visual scoring cache and image analysis inputs

Outputs:

- `scoring/output/<sample_id>/consensus/scores_matrix.csv`
- `scoring/output/<sample_id>/consensus/labels_matrix.csv`
- method/domain scoring logs and intermediate scoring artifacts

Downstream consumers:

- Stage C BEST builder
- API `/api/data/scores`

## Stage C: BEST Builder

Entrypoint:

- `ensemble/build_best.py`
- Wrapper: `ensagent_tools/best_builder.py`

Inputs:

- `scores_matrix.csv`
- `labels_matrix.csv`
- spot template CSV
- `visium_dir`
- optional truth file

Outputs:

- `output/best/<sample_id>/BEST_<sample_id>_spot.csv`
- `output/best/<sample_id>/BEST_<sample_id>_DEGs.csv`
- `output/best/<sample_id>/BEST_<sample_id>_PATHWAY.csv`
- `output/best/<sample_id>/<sample_id>_result.png`
- optional `output/best/<sample_id>/ari.json`

Downstream consumers:

- Stage D annotation
- API `/api/data/spatial`

## Stage D: Annotation

Entrypoint:

- `annotation/run_annotation_main.py`
- Wrapper: `ensagent_tools/annotation.py`

Inputs:

- `BEST_<sample_id>_spot.csv`
- `BEST_<sample_id>_DEGs.csv`
- `BEST_<sample_id>_PATHWAY.csv`
- provider/API configuration

Outputs:

- `output/best/<sample_id>/annotation_output/domain_annotations.json`

Downstream consumers:

- API `/api/annotation/{sample_id}/{cluster_id}`
- frontend annotation panel

## API Read Layer

`/api/data/spatial`

- Reads `output/best/<sample_id>/BEST_<sample_id>_spot.csv`
- Returns spot coordinates and domain labels

`/api/data/scores`

- Reads `scoring/output/<sample_id>/consensus/scores_matrix.csv`
- Reads `scoring/output/<sample_id>/consensus/labels_matrix.csv`
- Falls back to shared legacy consensus paths only for old result compatibility

`/api/annotation/{sample_id}/{cluster_id}`

- Preferentially reads `output/best/<sample_id>/annotation_output/domain_annotations.json`
- Falls back to the old shared annotation path only for old result compatibility

## Current Constraints

- API and frontend code must not treat shared output directories as the default source of truth.
- New sample runs must isolate outputs by `sample_id`.
- `pipeline_config.yaml` describes runtime parameters; it should not be used to infer the newest result path.
