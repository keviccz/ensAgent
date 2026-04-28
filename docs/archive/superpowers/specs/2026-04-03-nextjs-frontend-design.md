# EnsAgent Next.js Frontend — Design Spec

**Status:** Confirmed  
**Date:** 2026-04-03  
**Goal:** Replace the Streamlit UI with a Next.js 14 + FastAPI frontend served at `localhost:3000`.

---

## 1. Technology Stack

| Concern | Choice |
|---------|--------|
| Framework | Next.js 14 App Router + TypeScript |
| Styling | Tailwind CSS |
| Icons | Lucide Icons (`lucide-react`) |
| Charts | Recharts |
| State | Zustand |
| Backend | FastAPI (`api/main.py`) at `localhost:8000` |
| Launch | `start.py` (starts both FastAPI + Next.js dev server) |
| Fonts | Sora (UI) + JetBrains Mono (code/data) via Google Fonts CDN |

---

## 2. Design Tokens

```
--bg-main:       #FFFFFF
--bg-sidebar:    #F7F7F7
--bg-surface:    #F3F4F6
--bg-hover:      #EBEBEB
--dark-gray:     #2F2F2F   (primary button / text)
--mid-gray:      #CCCCCC   (active nav bg / user bubble)
--data:          #0EA5E9   (sky-blue accent)
--success:       #10B981
--warning:       #F59E0B
--border:        rgba(0,0,0,0.08)
--text-primary:  #111111
--text-muted:    #6B7280
```

---

## 3. Application Layout

```
┌─────────────────────────────────────────────────┐
│  Sidebar (230px fixed)  │  Main area (flex:1)    │
│  ─────────────────────  │  ─────────────────────  │
│  Brand: EnsAgent        │  Topbar (48px)          │
│  ─────────────────────  │  Page content           │
│  Nav items:             │                         │
│    Chat / Analysis /    │                         │
│    Agents / Settings    │                         │
│  ─────────────────────  │                         │
│  Conversation history   │                         │
└─────────────────────────────────────────────────┘
```

Sidebar background: `#F7F7F7`; Main background: `#FFFFFF`.

---

## 4. Page Specifications

### 4.1 Chat Page (`/chat`)

**Layout:** Flex column, full height, no scroll on outer container.

**Message list (flex:1, overflow-y:auto):**
- **User messages:** right-aligned, `#CCCCCC` rounded bubble, no avatar
- **AI replies:** left-aligned plain text, no avatar
- **Reasoning block:** fully expanded, gray left border (`#E5E7EB`), numbered steps
- **Tool Call block (collapsible):**
  - Collapsed: tool name + argument tags on one line
  - Expanded: JSON tab / Python tab with syntax highlighting
- **Pipeline Progress card:** 4-stage horizontal bar (Tool-Runner → Scoring → BEST → Annotation), each stage shows status (idle/running/done/error) with a thin progress bar; updates via SSE

**Input area (fixed bottom):**
- Rounded textarea (auto-grow up to 5 lines)
- Dark-gray send button (`#2F2F2F`)

**Conversation history sidebar section:**
- Shows 10 most recent conversations
- Click to switch; right-chevron context menu: Export JSON / Delete

### 4.2 Analysis Page (`/analysis`)

**KPI Strip (4 cards, grid):**
- Total Spots / Domains / Avg Expression / Coverage

**Domain Annotation section (50/50 split):**
- Left: Recharts scatter plot, one dot per spot colored by cluster; clicking a dot triggers `GET /api/annotation/{sample_id}/{cluster_id}`
- Right: `AnnotationPanel` — Cluster label, Confidence Score bar, Marker Genes chips, Interpretability text

**Spatial Expression section:**
- Expression scatter (color = expression level)
- Scores Matrix table: rows = methods (IRIS/BASS/DR-SC/BayesSpace/SEDR/GraphST/STAGATE/stLearn), columns = domains; each row prefixed with method label

### 4.3 Agents Page (`/agents`)

**Agent cards (2-column grid, 6 cards):**
- Labels: DP (Data Prep) / TR (Tool-Runner) / SA (Scoring/Analysis) / BB (BEST Builder) / AA (Annotation Agent) / CR (Critic/Review)
- States:
  - **IDLE:** gray badge
  - **ACTIVE:** sky-blue badge + animated pulsing dot + progress bar
  - **DONE:** green badge
- **SKIP button:** visible when corresponding output files already exist (DP, BB by default)
- Click on card → Activity Log filters to that agent's entries

**Filter bar (top of log section):** All + one button per agent; clicking highlights that agent's log rows.

**Activity Log:**
- Columns: timestamp | agent | message
- Color coding: blue = info, green = success, yellow = warning, red = error
- Auto-scroll to bottom; max 500 entries (virtualized if needed)

### 4.4 Settings Page (`/settings`)

**Two-column layout:**

Left column — API Configuration:
- Provider dropdown (12+ options: openai, azure, anthropic, gemini, openrouter, …)
- API Key (password input)
- Model name
- Endpoint URL (shown only when needed)
- API Version (shown only when needed)
- **Test Connection** button → calls `POST /api/config/test_connection`

Left column — Model Parameters (3 sliders):
- Temperature: 0–2, step 0.01, default 0.7
- Top-p: 0–1, step 0.01, default 0.95
- Visual Factor: 0–1, step 0.01, default 0.5 (visual scoring weight)
- Each slider shows live numeric readout

Right column — Pipeline Configuration:
- `data_path` text input
- `sample_id` text input
- `n_clusters` number input (default 7)
- `methods` multi-select checkboxes (8 methods)
- Skip Stages: Skip Tool-Runner / Skip Scoring checkboxes

**Save button** at bottom → calls `POST /api/config/save`

---

## 5. FastAPI Backend Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/chat` | POST + SSE | LLM streaming via `text/event-stream` |
| `/api/pipeline/run` | POST | Trigger full A→B→C→D pipeline |
| `/api/pipeline/stage/{name}` | POST | Trigger single stage |
| `/api/pipeline/skip` | POST | Mark stage as skipped |
| `/api/pipeline/status` | GET | Per-stage status + progress 0–100 |
| `/api/config/load` | GET | Read `pipeline_config.yaml` |
| `/api/config/save` | POST | Write `pipeline_config.yaml` |
| `/api/config/test_connection` | POST | Test LLM connectivity |
| `/api/annotation/{sample_id}/{cluster_id}` | GET | Cluster annotation result |
| `/api/data/scores` | GET | `scores_matrix.csv` as JSON |
| `/api/data/labels` | GET | `labels_matrix.csv` as JSON |
| `/api/data/spatial` | GET | Spot coordinates + cluster labels |
| `/api/agents/status` | GET | All agent states |
| `/api/agents/logs` | GET (SSE) | Activity log stream |

---

## 6. File Structure

```
EnsAgent/
├── frontend/                        # Next.js project root
│   ├── app/
│   │   ├── layout.tsx               # Root layout: sidebar + main
│   │   ├── page.tsx                 # redirect → /chat
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
│   │   │   ├── ToolCallBlock.tsx
│   │   │   ├── PipelineProgress.tsx
│   │   │   └── ChatInput.tsx
│   │   ├── analysis/
│   │   │   ├── KpiStrip.tsx
│   │   │   ├── DomainScatter.tsx
│   │   │   ├── AnnotationPanel.tsx
│   │   │   ├── ExpressionPlot.tsx
│   │   │   └── ScoresMatrix.tsx
│   │   ├── agents/
│   │   │   ├── AgentCard.tsx
│   │   │   ├── AgentGrid.tsx
│   │   │   ├── ActivityLog.tsx
│   │   │   └── FilterBar.tsx
│   │   └── settings/
│   │       ├── ApiConfig.tsx
│   │       ├── ModelParams.tsx
│   │       └── PipelineConfig.tsx
│   ├── lib/
│   │   ├── api.ts                   # FastAPI HTTP/SSE client
│   │   ├── store.ts                 # Zustand store
│   │   └── types.ts                 # Shared TypeScript types
│   ├── tailwind.config.ts
│   ├── next.config.ts
│   └── package.json
│
├── api/                             # FastAPI backend
│   ├── main.py                      # App factory + CORS
│   ├── deps.py                      # Shared dependencies (config loader)
│   └── routes/
│       ├── chat.py                  # SSE chat endpoint
│       ├── pipeline.py              # Pipeline control
│       ├── config.py                # Config read/write
│       ├── data.py                  # Data file readers
│       ├── annotation.py            # Annotation queries
│       └── agents.py                # Agent status + log stream
│
└── start.py                         # Launch FastAPI + Next.js
```

---

## 7. Data Flow

```
User action
    │
    ▼
Zustand store (optimistic update)
    │
    ▼
lib/api.ts (fetch / EventSource)
    │
    ▼
FastAPI route
    │
    ├─► ensagent_tools.execute_tool()   (pipeline stages)
    ├─► litellm / provider_runtime.py  (LLM calls)
    └─► filesystem reads               (scores_matrix, labels_matrix, annotations)
```

---

## 8. Non-Goals

- No authentication / user accounts
- No Docker (local dev only)
- No production build optimization (development server is sufficient)
- Streamlit app remains untouched and functional
