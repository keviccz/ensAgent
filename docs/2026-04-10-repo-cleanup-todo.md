# 仓库清理与结构梳理 TODO

日期：2026-04-10

## 目的

这份清单用于后续分阶段整理 EnsAgent 仓库，目标是：

- 明确哪些目录和文件仍然属于主流程。
- 标记可删除、可归档、待确认的历史遗留内容。
- 梳理各阶段输入输出，避免继续出现共享输出、隐式依赖和读取错位。
- 为后续重构提供一个可执行的检查表。

说明：

- 本清单不包含 `frontend/` 内部的整理项。
- 本清单以当前代码审查结论为基础，删除前仍应做一次引用确认和运行验证。

## 一、当前应保留的主流程骨架

以下目录和入口目前仍属于主流程，不应直接删除：

- `ensagent_tools/`
- `Tool-runner/`
- `scoring/`
- `ensemble/`
- `annotation/`
- `api/`
- `tests/`
- `envs/`
- `endtoend.py`
- `start.py`
- `pipeline_config.example.yaml`
- `pipeline_config.yaml`
- `ensagent_agent/`

## 二、可直接归档或删除的候选项

这些内容目前更像历史遗留、手工分析脚本或已退出主流程的实现，优先进入清理范围：

- [x] `streamlit_app/`
  说明：当前仓库主 Web 入口已经是 `api/` + `frontend/`，`streamlit_app/` 更像旧版 UI 残留。需要先做最后一次引用确认，然后整体归档或删除。
- [x] `tools/summarize_annotation_results.py`
  说明：仓库内未发现代码引用，更像一次性分析脚本。
- [x] `tools/summarize_domain_rounds.py`
  说明：仓库内未发现代码引用，更像一次性分析脚本。

## 三、待确认后处理的项

这些项不建议直接删，需要先确认是否仍被手工流程、演示流程或外部文档依赖：

- [x] `docs/design/`
  说明：属于设计过程文档集合，不参与主运行流程。需要决定是长期保留在主仓库，还是迁移到 `docs/archive/`。
- [ ] `docs/dev-progress.md`
  说明：如果是阶段性工作日志，应转为归档文档，而不是继续作为仓库长期主文档。
- [x] `docs/superpowers/`
  说明：需要确认这部分是团队协作规范、外部依赖说明，还是临时 AI 协作文档。
- [ ] 根目录下的手工运行脚本与临时文件
  说明：包括后续若出现的 `run_*_test.py`、截图、排障脚本等，应统一迁移到 `scripts/manual/` 或完全移出版本管理。

## 四、目录规划与归属重整任务

当前仓库最大的问题不是单点 bug，而是职责边界不稳定。下面这些任务建议按顺序处理：

- [x] 为仓库建立统一的目录归属表
  目标：把所有目录标记为 `运行时核心`、`开发辅助`、`文档`、`生成物`、`legacy` 五类。
- [x] 为 `tools/` 目录补充边界
  目标：明确哪些脚本是运行时必需，哪些只是运维/排障工具，哪些可以迁移到 `scripts/` 或 `docs/archive/`。
- [x] 为 `docs/` 目录建立分层
  目标：至少拆成 `docs/active/`、`docs/archive/` 或类似结构，避免进行中的文档、历史记录和说明文档混放。
- [x] 明确 `output/`、`scoring/output/` 一类目录是运行结果，不是源码模块
  目标：避免后续再把生成物当作事实来源写死在 API 路由里。
- [x] 建立“单一入口”原则
  目标：每个阶段只保留一个推荐入口，避免 CLI、脚本、包装器和 API 各自绕开同一套配置。

## 五、配置层整理任务

当前配置体系已经分裂为 Web 侧 YAML 字典读写和 pipeline 侧 dataclass 两套实现，需要尽快统一：

- [x] 盘点所有配置读取入口
  范围：`api/deps.py`、`api/routes/config.py`、`ensagent_tools/config_manager.py`、各 stage wrapper、`endtoend.py`。
- [x] 建立唯一配置模型
  目标：以 `ensagent_tools/config_manager.py` 为中心，决定 API 是否也必须经由同一模型读写。
- [x] 清理无效配置项
  重点：确认 `visual_factor` 这类前端可写但主流程不消费的参数，决定删除、接线或改名。
- [x] 标记“用户本地配置”和“仓库默认配置”
  目标：明确 `pipeline_config.example.yaml` 是模板，`pipeline_config.yaml` 是本地运行态，不应混作事实来源。

## 六、输入输出梳理任务

这一部分需要单独做成可追踪文档，建议后续补成一份正式的 I/O map。

### Stage A: Tool-runner

- [x] 明确输入
  包括 `data_path`、`sample_id`、`methods`、`n_clusters`、环境名等。
- [x] 明确输出目录
  当前主输出应归于 `output/tool_runner/<sample_id>/`。
- [x] 列出供 Stage B 消费的交付件
  包括 `spot/`、`DEGs/`、`PATHWAY/` 及其文件命名规则。

### Stage A -> B 中转

- [x] 标准化 staging 行为
  目标：把复制和重命名规则写清楚，避免靠隐式约定把内容塞进 `scoring/input/`。

### Stage B: Scoring

- [x] 明确输入来源
  包括 `scoring/input/`、模型配置、视觉评分缓存。
- [x] 明确输出结构
  包括 `scoring/output/` 和 `scoring/output/consensus/`。
- [x] 标记哪些输出是全局共享，哪些必须按 `sample_id` 隔离
  这是后续修复 API 读取错位的关键。

### Stage C: BEST Builder

- [x] 明确输入契约
  包括 `scores_matrix.csv`、`labels_matrix.csv`、spot template、`visium_dir`、可选 truth file。
- [x] 明确输出目录
  当前应归于 `output/best/<sample_id>/`。
- [x] 统一 BEST 输出文件命名
  避免 API 和 annotation 侧再各自猜测路径。

### Stage D: Annotation

- [x] 修复 annotation 默认输出为共享目录的问题
  当前共享写入 `output/best/annotation_output/domain_annotations.json`，多样本运行会互相覆盖。
- [x] 让 annotation 输出按 `sample_id` 隔离
  目标：使 API 读取路径和运行输出保持一一对应。
- [x] 明确 annotation 的输入三件套
  包括 `BEST_*_spot.csv`、`BEST_*_DEGs.csv`、`BEST_*_PATHWAY.csv`。

### API 读取层

- [x] 为每个 API 路由补一份“读取来源清单”
  重点覆盖 `/api/data/spatial`、`/api/data/scores`、`/api/annotation/{sample}/{cluster}`。
- [x] 禁止 API 继续读取模糊或共享结果路径
  目标：每个 API 响应都能追溯到明确的 sample 级输出。

## 七、已确认需要跟进的后端与流程问题

这些问题不一定属于“删除清理”，但必须进入整改队列，否则仓库即使瘦身后仍然会不稳定：

- [x] 修复 `/api/data/scores` 忽略 `sample_id` 的问题
  当前实现直接读取 `scoring/output/consensus/{scores,labels}_matrix.csv`，会导致样本切换后仍读到旧结果。
- [x] 修复 annotation API 优先读取共享路径的问题
  否则即使 Stage D 支持多样本，也会被 API 层重新读错。
- [x] 重新定义 pipeline/agents demo 状态页的后端职责
  当前 `api/routes/pipeline.py` 和 `api/routes/agents.py` 更接近 demo stub，不应继续伪装成真实运行态。
- [x] 统一 skip / control 类接口契约
  需要统一字段命名、阶段标识和后端允许值。
- [x] 去除 `start.py` 中的机器绑定 Python 路径
  当前实现写死本机解释器路径，不适合作为仓库公共入口。
- [x] 修复环境相关测试的不稳定性
  `tests/test_env_manager_resolver.py` 目前受宿主机环境变量影响，不够 hermetic。
- [x] 更新 `tools/health_check.py`
  该脚本仍偏向 Azure-only 校验，应与当前多 provider 设计保持一致。

## 八、执行顺序建议

建议按下面顺序推进，避免一边删旧文件一边打断主流程：

- [x] 第一步：补“目录归属表”
- [x] 第二步：补“主流程 I/O map”
- [x] 第三步：统一配置模型
- [x] 第四步：修复共享输出和 API 错读
- [x] 第五步：清理 legacy 目录和未使用脚本
- [x] 第六步：补最小验证集，确保整理后主流程仍可运行

## 九、完成标准

当以下条件同时满足时，可以认为本轮仓库整理完成：

- [x] 每个主目录都有明确职责说明
- [x] 每个阶段的输入、输出、消费者都能追溯
- [x] API 不再读取共享或模糊路径
- [x] `pipeline_config` 只有一套权威配置模型
- [x] legacy 目录和未使用脚本已经归档或删除
- [x] 关键测试和健康检查能够稳定通过
