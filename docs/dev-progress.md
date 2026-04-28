# EnsAgent 开发进度记录

> 最后更新：2026-04-05

---

## 已完成工作

### T1 — 补全 `completion_text_stream`
**文件：** `scoring/provider_runtime.py`

原 `provider_runtime.py` 只有同步的 `completion_text`，缺少 chat 后端依赖的异步流式生成器。

新增 `async def completion_text_stream(config, messages, ...)` 函数：
- 在 `run_in_executor` 中调用 litellm 同步流（sync → async 桥接）
- 逐 chunk yield delta 字符串
- 修复 `resolve_provider_config(**str_cfg)` 调用方式（kwarg-only 接口）

---

### T2 — 前端 UI 修复
**文件：** `frontend/components/analysis/ScoresMatrix.tsx`，`frontend/app/chat/page.tsx`

| 问题 | 修复 |
|------|------|
| ScoresMatrix 方法名多了前缀"D"（如 BASS→DBASS） | 改为仅对纯数字 key 加 `D` 前缀：`isNaN(Number(d)) ? d : \`D${d}\`` |
| Chat 空白页文字过淡（opacity: 0.5） | 改为 opacity: 0.85 |

---

### T3 — Function Calling 工具调用循环
**文件：** `api/routes/chat.py`，`frontend/lib/{types,store,api}.ts`，`frontend/app/chat/page.tsx`，`frontend/components/chat/ChatMessages.tsx`

#### 后端（`api/routes/chat.py`）
- `_sync_completion(messages, cfg, stream)` — litellm 调用，传入 `TOOL_SCHEMAS` + `tool_choice="auto"`
- `_agent_stream` — 最多 `MAX_TOOL_ROUNDS=5` 轮循环：
  1. 检测 `tool_calls` → 发送 `{type:"tool_call", ...}` SSE 事件
  2. 调用 `execute_tool(name, args, cfg)` → 发送 `{type:"tool_result", ...}` SSE 事件
  3. 将结果追加到 messages 继续循环
  4. 无工具调用时流式输出文本 delta

#### 前端
- `ToolCall` 类型新增 `result?: string` 字段
- Store 新增 `streamingToolCalls`、`addStreamingToolCall`、`updateStreamingToolCallResult` 
- `createChatStream` 新增 `onToolCall` / `onToolResult` 回调
- `ChatMessages` 组件在流式阶段渲染 `<ToolCallBlock>` 组件

---

### T4 — Scores API 数据格式修复
**文件：** `api/routes/data.py`

原始 CSV 为 spot×method 格式，前端需要 method×domain 格式。

重写 `GET /api/data/scores`：
1. 读取 `scoring/output/consensus/scores_matrix.csv`（spot×method 分数）
2. 联表 `scoring/output/consensus/labels_matrix.csv`（spot×method domain 标签）
3. 按 method+domain 聚合取均值
4. 输出：`[{method: "BASS", scores: {"1": 0.82, "2": 0.68, ...}}, ...]`

另修复 `GET /api/data/spatial`：BEST 输出列名为 `spatial_domain`，原代码只查找 `domain`/`label`，已补充。

---

### T5 — 目录清理
**文件：** `.gitignore`，根目录

- 删除根目录散落的调试 PNG（约 12 张）和临时脚本
- 设计截图统一移至 `docs/design/`
- `.gitignore` 新增 `!frontend/lib/` 例外规则（防止被 `lib/` 规则误忽略）

---

### T6 — 打分流水线冒烟测试（Stage B + C）

使用 `ens_dev` conda 环境，`vlm_off=True`（跳过视觉评分，因 `ens_dev` 缺少 `flask`/`litellm` 等 pic_analyze 依赖）：

```
Stage B (Scoring): OK
  - 8 个方法全部评分完成
  - 输出: scoring/output/consensus/scores_matrix.csv
  - 输出: scoring/output/consensus/labels_matrix.csv
  - 选用方法: SEDR, STAGATE, IRIS（平均分 0.889）

Stage C (BEST Builder): OK
  - 输出: output/best/151507/BEST_151507_spot.csv
  - 输出: output/best/151507/BEST_151507_DEGs.csv
  - 输出: output/best/151507/BEST_151507_PATHWAY.csv
  - 输出: output/best/151507/151507_result.png
```

---

## 已知问题 / 待办

| 项目 | 状态 | 说明 |
|------|------|------|
| `ens_dev` 缺少 pic_analyze 依赖 | 待处理 | 需在 ens_dev 中 `pip install flask litellm` 等，才能启用视觉评分 |
| Stage D（Annotation）未测试 | 待测试 | 多智能体标注流程尚未在当前环境验证 |
| 前端 Analysis 页面未完整验证 | 待验证 | ScoresMatrix 和 SpatialView 需在浏览器中确认数据显示正确 |
| 前端 Chat 工具调用显示 | 待验证 | 需实际发送触发工具的消息，验证 ToolCallBlock 渲染 |

---

## 环境说明

| 环境 | 用途 |
|------|------|
| `ens_dev` | 前后端开发运行（FastAPI + Next.js） |
| `ensagent` | 完整生产环境（含 Streamlit） |
| `ensagent_R` | R 聚类方法（IRIS, BASS, DR-SC, BayesSpace） |
| `ensagent_PY` | PyTorch 聚类方法（SEDR, GraphST, STAGATE） |
| `ensagent_PY2` | TF 聚类方法（stLearn） |

### 启动命令

```bash
# 后端（ens_dev 环境，项目根目录）
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 前端（frontend/ 目录）
npm run dev
```

---

## 提交记录（本轮开发）

| Hash | 说明 |
|------|------|
| `60af487` | feat(chat): implement tool-calling agent loop with SSE events |
| `0ad4a3c` | chore: move design screenshots to docs/design, remove tmp dirs |
| `815b2dd` | fix(T1/T2/T4): chat stream, UI bugs, scores API |
| `1e000d5` | feat(frontend+api): complete Next.js frontend and FastAPI backend |
