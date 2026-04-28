# EnsAgent Next.js Frontend — Plan Status

> 状态文件生成时间：2026-04-03  
> 用途：context clear 后继续工作的完整参考

---

## 项目目标

将 EnsAgent 的 Streamlit 前端替换为 **Next.js + Tailwind CSS** 前端，后端用 **FastAPI** 本地服务，用户运行一个启动脚本即可在 `localhost:3000` 使用。

---

## 已确认的技术栈 & 设计决策

| 方面 | 决策 |
|------|------|
| 框架 | Next.js 14 App Router + TypeScript |
| 样式 | Tailwind CSS |
| 图标 | Lucide Icons (`lucide-react`) |
| 图表 | Recharts |
| 状态管理 | Zustand |
| 后端 | FastAPI (`api/main.py`)，本地 `:8000` |
| 启动 | `start.py` 同时拉起 FastAPI + Next.js dev |
| 字体 | Sora (UI) + JetBrains Mono (代码/数据)，Google Fonts CDN 加载 |

---

## 设计系统（已用户确认）

```
背景:       #FFFFFF (主区) / #F7F7F7 (侧边栏)
表面层:     #F3F4F6
Hover:      #EBEBEB
主色/按钮:  #2F2F2F (深灰，GPT风格)
活跃导航:   #CCCCCC (浅灰背景 + 黑色文字)
用户气泡:   #CCCCCC
天蓝/数据:  #0EA5E9
成功:       #10B981
警告:       #F59E0B
边框:       rgba(0,0,0,0.08)
```

---

## 页面结构（已确认）

### 布局
- 左侧固定侧边栏 (230px)：品牌名 + 导航 + 对话历史
- 右侧主区：顶栏 + 页面内容

### 4 个页面

#### 1. Chat（已设计完毕）
- 用户消息：右对齐 #CCCCCC 圆角气泡，无头像
- AI 回复：左对齐纯文字，无头像
- Reasoning 块：完整展开，灰色左竖线，编号步骤
- Tool Call 块（折叠/展开）：
  - 折叠：工具名 + 参数 tag 行
  - 展开：JSON / Python 两 tab，语法高亮
- Pipeline 进度卡：4 阶段进度条，实时更新
- 输入框：圆角，深灰发送按钮

#### 2. Analysis（已设计完毕）
- KPI 条：Total Spots / Domains / Avg Expression / Coverage
- **Domain Annotation 分栏（左右）**：
  - 左：散点图，点击 cluster → 右侧显示 annotation
  - 右：Annotation + Confidence Score + Marker Genes + Interpretability
  - API: `GET /api/annotation/{sample_id}/{cluster_id}`
- Spatial Expression：Expression scatter + Scores Matrix
  - **Scores Matrix 每行显示对应的 method 标签**（如 BASS、IRIS、GraphST 等）

#### 3. Agents（已设计完毕）
- 6 个 Agent 卡片 (2列)：DP / TR / SA / BB / AA / CR
- 状态：IDLE / ACTIVE (进度条 + 跳动蓝点) / DONE
- **SKIP 按钮**：当已有对应输出文件时显示（DP、BB 示例）
- **点击卡片 → Activity Log 自动筛选该 Agent 日志**
- 顶部 Filter Bar：All + 6 个 Agent 按钮随时切换
- Activity Log：时间戳 + 颜色区分（blue=info, green=success）

#### 4. Settings（已设计完毕）
- 双列布局
- 左：API 配置（Provider 12+种 / Key / Model / Endpoint）+ Model Parameters
  - **3 个滑块**：Temperature / Top-p / Visual Factor（0-1，控制视觉评分权重）
  - 滑动实时更新数值显示
- 右：Pipeline 配置（data_path / sample_id / n_clusters / methods / Skip Stages 复选框）

---

## 目录结构（规划）

```
EnsAgent/
├── frontend/                    # Next.js 项目
│   ├── app/
│   │   ├── layout.tsx           # 根布局：侧边栏 + 主区
│   │   ├── page.tsx             # redirect → /chat
│   │   ├── chat/page.tsx
│   │   ├── analysis/page.tsx
│   │   ├── agents/page.tsx
│   │   └── settings/page.tsx
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Sidebar.tsx
│   │   │   └── Topbar.tsx
│   │   ├── chat/
│   │   │   ├── ChatMessages.tsx
│   │   │   ├── MessageBubble.tsx
│   │   │   ├── ReasoningBlock.tsx
│   │   │   ├── ToolCallBlock.tsx      # 折叠/展开，JSON/Python tab
│   │   │   ├── PipelineProgress.tsx
│   │   │   └── ChatInput.tsx
│   │   ├── analysis/
│   │   │   ├── KpiStrip.tsx
│   │   │   ├── DomainScatter.tsx      # 可点击
│   │   │   ├── AnnotationPanel.tsx    # 右侧信息面板
│   │   │   ├── ExpressionPlot.tsx
│   │   │   └── ScoresMatrix.tsx       # 含 method 标签
│   │   ├── agents/
│   │   │   ├── AgentCard.tsx          # 可点击，含 SKIP
│   │   │   ├── AgentGrid.tsx
│   │   │   ├── ActivityLog.tsx        # 可筛选
│   │   │   └── FilterBar.tsx
│   │   └── settings/
│   │       ├── ApiConfig.tsx
│   │       ├── ModelParams.tsx        # 3 sliders
│   │       └── PipelineConfig.tsx
│   ├── lib/
│   │   ├── api.ts                     # FastAPI client
│   │   ├── store.ts                   # Zustand store
│   │   └── types.ts
│   ├── tailwind.config.ts
│   └── package.json
│
├── api/                               # FastAPI 后端
│   ├── main.py
│   ├── routes/
│   │   ├── chat.py                    # LLM 对话 + SSE 流式
│   │   ├── pipeline.py                # 触发各阶段
│   │   ├── config.py                  # 读写 pipeline_config.yaml
│   │   ├── data.py                    # 读取结果文件
│   │   └── annotation.py             # annotation 查询
│   └── deps.py                        # 复用 ensagent_tools
│
├── start.py                           # 一键启动
```

---

## FastAPI 端点设计

| 端点 | 方法 | 用途 |
|------|------|------|
| `/api/chat` | POST + SSE | LLM 对话 + 流式输出 |
| `/api/pipeline/run` | POST | 触发完整 pipeline |
| `/api/pipeline/stage/{name}` | POST | 触发单个阶段 |
| `/api/pipeline/skip` | POST | 跳过某阶段 |
| `/api/pipeline/status` | GET | 获取各阶段状态 + 进度 |
| `/api/config/load` | GET | 读取 pipeline_config.yaml |
| `/api/config/save` | POST | 保存配置 |
| `/api/config/test_connection` | POST | 测试 LLM 连接 |
| `/api/annotation/{sample_id}/{cluster_id}` | GET | 获取 cluster annotation |
| `/api/data/scores` | GET | 读取 scores_matrix.csv |
| `/api/data/labels` | GET | 历史设计接口，当前运行时已移除 |
| `/api/data/spatial` | GET | 读取空间坐标数据 |
| `/api/agents/status` | GET | 获取所有 agent 状态 |
| `/api/agents/logs` | GET | 获取 activity log |

---

## 待修复 BUG（当前阻塞）

### BUG 1：Chat 页面出现在其他页面中（高优先级）

**问题**：`#page-chat` 的 HTML 里有内联 `style="flex-direction:column;height:100%;overflow:hidden;"` 导致该元素的 `display` 不被 CSS 的 `.page { display:none }` 覆盖（内联样式优先级高于外部 CSS），所以 Chat 页面永远可见。

**文件**：`E:\A project\EnsAgent Files\EnsAgent\frontend\_mockup\pages_v2.html`

**修复方案**：
1. 去掉 `#page-chat` 的所有内联 `style`，改为在 CSS 中添加：
   ```css
   #page-chat { flex-direction: column; height: 100%; overflow: hidden; }
   ```
2. 或者将所有页面的 `.page.active` 分别处理：
   ```css
   .page.active { display: flex; }
   #page-analysis.active, #page-agents.active, #page-settings.active { display: block; }
   ```

---

## 当前进度（Brainstorming 流程）

| # | 任务 | 状态 |
|---|------|------|
| 1 | 探索项目上下文 | ✅ 完成 |
| 2 | 提供可视化辅助工具 | ✅ 完成 |
| 3 | 澄清问题 | ✅ 完成 |
| 4 | 提出 2-3 个架构方案 | ✅ 完成（选方案 A）|
| 5 | 展示设计各节并获批 | ✅ 完成（所有页面已展示）|
| 6 | 写设计文档到 `docs/archive/superpowers/specs/` | ✅ 完成 |
| 7 | Spec 自审 + 用户审阅 | ✅ 完成 |
| 8 | 调用 writing-plans skill | ✅ 完成 |

---

## 实现计划已生成

- **Spec**: `docs/archive/superpowers/specs/2026-04-03-nextjs-frontend-design.md`
- **Plan**: `docs/archive/superpowers/plans/2026-04-03-nextjs-frontend.md`

## 下一步

执行实现计划：调用 `superpowers:executing-plans` 或 `superpowers:subagent-driven-development`

---

## 参考文件位置

| 文件 | 说明 |
|------|------|
| `frontend/_mockup/pages_v2.html` | 当前最新交互式设计预览（含所有4页） |
| `frontend/_mockup/design_preview_v3.html` | Chat 页面设计（已定稿）|
| legacy Streamlit UI | 已退役并从仓库移除 |
| `ensagent_tools/` | 工具层，FastAPI 将直接复用 |
