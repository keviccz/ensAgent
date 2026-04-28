# 仓库目录归属表

日期：2026-04-10

## 目的

这份文档定义 EnsAgent 当前仓库的目录职责，避免再次出现“运行时源码、历史遗留、文档、生成物”混放的问题。

## 分类规则

- `运行时核心`：主流程直接依赖的源码或入口。
- `开发辅助`：健康检查、测试、环境定义、启动脚本等。
- `文档`：长期保留的说明、设计、过程记录。
- `生成物`：运行结果、构建产物、缓存，不应作为源码事实来源。
- `legacy`：已退出主流程、仅为迁移兼容或待清理内容。

## 目录归属

| 路径 | 分类 | 当前职责 |
| --- | --- | --- |
| `ensagent_tools/` | 运行时核心 | 统一工具层、配置模型、Stage A-D 包装器、主流程编排。 |
| `Tool-runner/` | 运行时核心 | Stage A 聚类编排、方法脚本、后处理、配置样例。 |
| `scoring/` | 运行时核心 | Stage B 打分、共识矩阵生成、视觉评分入口。 |
| `ensemble/` | 运行时核心 | Stage C BEST 构建。 |
| `annotation/` | 运行时核心 | Stage D 多 agent 注释。 |
| `api/` | 运行时核心 | FastAPI 后端与运行态 API。 |
| `frontend/` | 运行时核心 | Next.js 前端。 |
| `ensagent_agent/` | 运行时核心 | CLI 聊天代理入口。 |
| `tests/` | 开发辅助 | 单元测试与运行时契约回归。 |
| `tools/` | 开发辅助 | 仓库级健康检查和运维脚本；只保留被主流程或验证命令使用的脚本。 |
| `envs/` | 开发辅助 | Conda/Mamba 环境定义。 |
| `docs/` | 文档 | 当前清理 TODO、目录归属、I/O map 等长期说明。 |
| `output/` | 生成物 | Stage A/C/D 运行输出；只能作为结果目录，不是源码模块。 |
| `scoring/output/` | 生成物 | Stage B 运行输出；规范路径为 `scoring/output/<sample_id>/...`。 |
| `frontend/.next/` | 生成物 | Next.js 构建产物；应忽略，不入库。 |
| `streamlit_app/` | legacy | 旧 UI，已退出当前支持栈，应持续从主仓库清理。 |

## 单一入口原则

- 全流程推荐入口：`endtoend.py`
- Web 本地入口：`start.py`
- CLI 代理入口：`python -m ensagent_agent.chat`
- 配置唯一权威模型：`ensagent_tools/config_manager.py`

## 维护规则

- 新增运行时代码优先放在 `ensagent_tools/`、对应 stage 目录、`api/` 或 `frontend/`，不要放进 `tools/`。
- `tools/` 下的脚本必须满足至少一项：
  - 被 README / AGENTS / 测试命令显式引用
  - 属于仓库级健康检查或排障
- 新增文档优先放在 `docs/`，不要散落在根目录。
- `output/`、`scoring/output/`、上传数据、截图、临时脚本默认视为运行态或手工产物，不作为事实来源提交。
