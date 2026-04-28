# Pipeline I/O Map

日期：2026-04-10

## 总览

EnsAgent 当前推荐的数据流是：

1. Stage A `Tool-runner`
2. Stage A -> B staging
3. Stage B `Scoring`
4. Stage C `BEST Builder`
5. Stage D `Annotation`
6. API / Frontend 只读取样本级输出

## Stage A: Tool-runner

入口：

- `Tool-runner/orchestrator.py`
- `ensagent_tools/tool_runner.py`
- `ensagent_tools/pipeline.py`

主要输入：

- `data_path`
- `sample_id`
- `methods`
- `n_clusters`
- `random_seed`
- 环境解析结果（`conda_exe`、env 名）

规范输出：

- `output/tool_runner/<sample_id>/domains/`
- `output/tool_runner/<sample_id>/spot/`
- `output/tool_runner/<sample_id>/DEGs/`
- `output/tool_runner/<sample_id>/PATHWAY/`
- `output/tool_runner/<sample_id>/PICTURES/`
- `output/tool_runner/<sample_id>/tool_runner_report.json`

下游消费者：

- Stage A -> B staging

## Stage A -> B Staging

入口：

- `ensagent_tools/pipeline.py::stage_toolrunner_outputs`

作用：

- 把 Stage A 的 `spot/`、`DEGs/`、`PATHWAY/` 复制并标准化到 `scoring/input/`

主要输入：

- `output/tool_runner/<sample_id>/spot/`
- `output/tool_runner/<sample_id>/DEGs/`
- `output/tool_runner/<sample_id>/PATHWAY/`

规范输出：

- `scoring/input/*_<sample_id>_spot.csv`
- `scoring/input/*_<sample_id>_DEGs.csv`
- `scoring/input/*_<sample_id>_PATHWAY.csv`

## Stage B: Scoring

入口：

- `scoring/scoring.py`
- `ensagent_tools/scoring.py`

主要输入：

- `scoring/input/`
- API/provider 配置
- `sample_id`
- 视觉评分缓存与图片分析输入

规范输出：

- `scoring/output/<sample_id>/`
- `scoring/output/<sample_id>/consensus/scores_matrix.csv`
- `scoring/output/<sample_id>/consensus/labels_matrix.csv`

下游消费者：

- Stage C `BEST Builder`
- `/api/data/scores`

## Stage C: BEST Builder

入口：

- `ensemble/build_best.py`
- `ensagent_tools/best_builder.py`

主要输入：

- `scoring/output/<sample_id>/consensus/scores_matrix.csv`
- `scoring/output/<sample_id>/consensus/labels_matrix.csv`
- `scoring/input/*_<sample_id>_spot.csv`
- `visium_dir`
- 可选 `truth_file`

规范输出：

- `output/best/<sample_id>/BEST_<sample_id>_spot.csv`
- `output/best/<sample_id>/BEST_<sample_id>_DEGs.csv`
- `output/best/<sample_id>/BEST_<sample_id>_PATHWAY.csv`
- `output/best/<sample_id>/<sample_id>_result.png`
- 可选 `output/best/<sample_id>/ari.json`

下游消费者：

- Stage D `Annotation`
- `/api/data/spatial`

## Stage D: Annotation

入口：

- `annotation/run_annotation_main.py`
- `ensagent_tools/annotation.py`

主要输入：

- `output/best/<sample_id>/BEST_<sample_id>_spot.csv`
- `output/best/<sample_id>/BEST_<sample_id>_DEGs.csv`
- `output/best/<sample_id>/BEST_<sample_id>_PATHWAY.csv`

规范输出：

- `output/best/<sample_id>/annotation_output/domain_annotations.json`

下游消费者：

- `/api/annotation/{sample_id}/{cluster_id}`

## API 读取层

### `/api/data/spatial`

读取：

- `output/best/<sample_id>/BEST_<sample_id>_spot.csv`

返回：

- spot 坐标、cluster/domain 信息

### `/api/data/scores`

读取：

- `scoring/output/<sample_id>/consensus/scores_matrix.csv`
- `scoring/output/<sample_id>/consensus/labels_matrix.csv`

兼容回退：

- 仅为旧结果兼容，才回退到共享 `scoring/output/consensus/`

### `/api/annotation/{sample_id}/{cluster_id}`

读取：

- 首选 `output/best/<sample_id>/annotation_output/domain_annotations.json`

兼容回退：

- 仅为旧结果兼容，才回退到旧共享路径

## 当前约束

- API 和前端不得再把共享输出目录当作默认事实来源。
- 新样本运行结果必须按 `sample_id` 隔离。
- `pipeline_config.yaml` 只描述运行参数，不承担“推断最新结果路径”的职责。
