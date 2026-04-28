import { create } from 'zustand'
import type {
  Conversation, ChatMessage, PipelineState, StageState,
  AgentState, LogEntry, AppConfig, ToolCall, ChartData,
} from './types'

const DEFAULT_STAGES: StageState[] = [
  { name: 'tool_runner', label: 'Tool-Runner', status: 'idle', progress: 0 },
  { name: 'scoring',     label: 'Scoring',     status: 'idle', progress: 0 },
  { name: 'best',        label: 'BEST',        status: 'idle', progress: 0 },
  { name: 'annotation',  label: 'Annotation',  status: 'idle', progress: 0 },
]

const DEFAULT_AGENTS: AgentState[] = [
  { id: 'TR', label: 'TR', fullName: 'Tool-Runner',      status: 'IDLE', progress: 0, canSkip: false },
  { id: 'SA', label: 'SA', fullName: 'Scoring/Analysis', status: 'IDLE', progress: 0, canSkip: false },
  { id: 'BB', label: 'BB', fullName: 'BEST Builder',     status: 'IDLE', progress: 0, canSkip: false },
  { id: 'AA', label: 'AA', fullName: 'Annotation Agent', status: 'IDLE', progress: 0, canSkip: false },
]

// Pre-seeded demo run log (DLPFC_151507) — shown even before the API responds
const DEMO_LOGS: LogEntry[] = (
  [
    { ts: '08:31:02', a: 'TR', lv: 'info',    m: 'Tool-Runner started. Sample: DLPFC_151507, n_clusters=7, methods=[IRIS,BASS,DR-SC,BayesSpace,SEDR,GraphST,STAGATE,stLearn].' },
    { ts: '08:31:05', a: 'TR', lv: 'info',    m: 'Running IRIS (R env)…' },
    { ts: '08:43:18', a: 'TR', lv: 'info',    m: 'IRIS finished. Output: output/tool_runner/DLPFC_151507/IRIS/' },
    { ts: '08:43:20', a: 'TR', lv: 'info',    m: 'Running BASS (R env)…' },
    { ts: '08:57:44', a: 'TR', lv: 'info',    m: 'BASS finished. Output: output/tool_runner/DLPFC_151507/BASS/' },
    { ts: '08:57:46', a: 'TR', lv: 'info',    m: 'Running DR-SC (R env)…' },
    { ts: '09:09:31', a: 'TR', lv: 'info',    m: 'DR-SC finished. Output: output/tool_runner/DLPFC_151507/DR-SC/' },
    { ts: '09:09:33', a: 'TR', lv: 'info',    m: 'Running BayesSpace (R env)…' },
    { ts: '09:22:07', a: 'TR', lv: 'info',    m: 'BayesSpace finished. Output: output/tool_runner/DLPFC_151507/BayesSpace/' },
    { ts: '09:22:09', a: 'TR', lv: 'info',    m: 'Running SEDR (PY env, CUDA)…' },
    { ts: '09:36:55', a: 'TR', lv: 'info',    m: 'SEDR finished. Output: output/tool_runner/DLPFC_151507/SEDR/' },
    { ts: '09:36:57', a: 'TR', lv: 'info',    m: 'Running GraphST (PY env, CUDA)…' },
    { ts: '09:51:12', a: 'TR', lv: 'info',    m: 'GraphST finished. Output: output/tool_runner/DLPFC_151507/GraphST/' },
    { ts: '09:51:14', a: 'TR', lv: 'info',    m: 'Running STAGATE (PY env, CUDA)…' },
    { ts: '10:06:39', a: 'TR', lv: 'info',    m: 'STAGATE finished. Output: output/tool_runner/DLPFC_151507/STAGATE/' },
    { ts: '10:06:41', a: 'TR', lv: 'info',    m: 'Running stLearn (PY2 env)…' },
    { ts: '10:19:03', a: 'TR', lv: 'success', m: 'Tool-Runner completed. 8/8 methods succeeded.' },
    { ts: '10:19:05', a: 'SA', lv: 'info',    m: 'Scoring/Analysis started. Staging tool-runner outputs for DLPFC_151507…' },
    { ts: '10:19:18', a: 'SA', lv: 'info',    m: 'Evaluating Domain 1 (n=726 spots) — calling LLM scorer…' },
    { ts: '10:21:44', a: 'SA', lv: 'info',    m: 'Domain 1 scored. Best method: STAGATE (0.81). Running pathway analysis…' },
    { ts: '10:23:02', a: 'SA', lv: 'info',    m: 'Evaluating Domain 2 (n=1,060 spots) — calling LLM scorer…' },
    { ts: '10:25:31', a: 'SA', lv: 'info',    m: 'Domain 2 scored. Best method: GraphST (0.78).' },
    { ts: '10:25:33', a: 'SA', lv: 'info',    m: 'Evaluating Domain 3 (n=891 spots) — calling LLM scorer…' },
    { ts: '10:27:58', a: 'SA', lv: 'info',    m: 'Domain 3 scored. Best method: STAGATE (0.76).' },
    { ts: '10:27:59', a: 'SA', lv: 'info',    m: 'Evaluating Domain 4 (n=654 spots) — calling LLM scorer…' },
    { ts: '10:30:22', a: 'SA', lv: 'info',    m: 'Domain 4 scored. Best method: BayesSpace (0.74).' },
    { ts: '10:30:24', a: 'SA', lv: 'info',    m: 'Evaluating Domain 5–7 (n=2,418 spots) — calling LLM scorer…' },
    { ts: '10:34:11', a: 'SA', lv: 'info',    m: 'Domains 5–7 scored. Building scores_matrix.csv…' },
    { ts: '10:34:29', a: 'SA', lv: 'success', m: 'Scoring/Analysis completed. scores_matrix.csv written to output/scoring/DLPFC_151507/.' },
    { ts: '10:34:31', a: 'BB', lv: 'info',    m: 'BEST Builder started. Reading scores_matrix.csv…' },
    { ts: '10:34:33', a: 'BB', lv: 'info',    m: 'Computing ensemble consensus — Borda count + weighted voting…' },
    { ts: '10:34:41', a: 'BB', lv: 'info',    m: 'Applying kNN smoothing (k=15, sigma=0.8)…' },
    { ts: '10:34:48', a: 'BB', lv: 'success', m: 'BEST Builder completed. BEST_DLPFC_151507.csv written to output/best/DLPFC_151507/.' },
    { ts: '10:34:50', a: 'AA', lv: 'info',    m: 'Annotation Agent started. Loading BEST labels for DLPFC_151507 (7 domains)…' },
    { ts: '10:35:02', a: 'AA', lv: 'info',    m: 'Domain 1 — Round 1: Marker expert scoring (MOBP, PLP1, MBP markers detected)…' },
    { ts: '10:36:44', a: 'AA', lv: 'info',    m: 'Domain 1 — Round 1: Critic score 0.87 ≥ threshold 0.65. Gate passed.' },
    { ts: '10:36:46', a: 'AA', lv: 'info',    m: 'Domain 2 — Round 1: Pathway expert scoring (myelination, axon ensheathment)…' },
    { ts: '10:38:29', a: 'AA', lv: 'info',    m: 'Domain 2 — Round 1: Critic score 0.79 ≥ threshold 0.65. Gate passed.' },
    { ts: '10:38:31', a: 'AA', lv: 'info',    m: 'Domain 3 — Round 1: Spatial expert scoring (Layer V pyramidal morphology)…' },
    { ts: '10:40:14', a: 'AA', lv: 'warning', m: 'Domain 3 — Round 1: Critic score 0.61 < threshold 0.65. Requesting rerun.' },
    { ts: '10:40:17', a: 'AA', lv: 'info',    m: 'Domain 3 — Round 2: Rerunning Spatial + VLM experts (low score on morphological evidence)…' },
    { ts: '10:42:08', a: 'AA', lv: 'info',    m: 'Domain 3 — Round 2: Critic score 0.72 ≥ threshold 0.65. Gate passed.' },
    { ts: '10:42:10', a: 'AA', lv: 'info',    m: 'Domain 4–7 — Running experts in parallel…' },
    { ts: '10:47:55', a: 'AA', lv: 'info',    m: 'Domains 4–7 annotated. All critic gates passed.' },
    { ts: '10:48:02', a: 'AA', lv: 'info',    m: 'Writing domain_annotations.json to output/best/DLPFC_151507/annotation_output/…' },
    { ts: '10:48:07', a: 'AA', lv: 'success', m: 'Annotation Agent completed. 7 domains annotated (6 in round 1, 1 in round 2).' },
    { ts: '10:48:08', a: 'AA', lv: 'success', m: 'D1 → Layer 1 (conf=87%): GFAP+ astrocyte-rich superficial zone. Key markers: GFAP (FC=41.3), AQP4 (FC=18.7).' },
    { ts: '10:48:08', a: 'AA', lv: 'success', m: 'D2 → Layer 2 (conf=82%): Excitatory neuron-rich upper cortex. Key markers: CAMK2N1 (FC=11.38), ENC1 (FC=10.98), CUX2 (FC=8.21).' },
    { ts: '10:48:09', a: 'AA', lv: 'success', m: 'D3 → Layer 3 (conf=85%): Deep excitatory band. Key markers: NRGN (FC=13.5), MT-CO2 (FC=50.08). Required 2 rounds.' },
    { ts: '10:48:09', a: 'AA', lv: 'success', m: 'D4 → Layer 4 (conf=79%): Granular input layer. Key markers: NEFL (FC=5.86), TUBA1B (FC=9.6), RORB (FC=6.43).' },
    { ts: '10:48:10', a: 'AA', lv: 'success', m: 'D5 → Layer 5 (conf=81%): Deep corticothalamic neurons. Key markers: TMSB10 (FC=22.04), MBP (FC=7.63).' },
    { ts: '10:48:10', a: 'AA', lv: 'success', m: 'D6 → Layer 6/White Matter (conf=93%): Myelinated deep layer. Key markers: MBP (FC=68.29), PLP1 (FC=40.1), MAG (FC=32.56).' },
    { ts: '10:48:11', a: 'AA', lv: 'success', m: 'D7 → White Matter (conf=96%): Dense oligodendrocyte core. Key markers: MBP (FC=179.8), PLP1 (FC=106.87), MOBP (FC=19.06).' },
    { ts: '10:48:14', a: 'AA', lv: 'success', m: 'Pipeline complete. DLPFC_151507: 4,220 spots · 7 domains · ARI=0.512 vs ground truth.' },
  ] as { ts: string; a: string; lv: string; m: string }[]
).map((e, i) => ({
  id: `demo-${i}`,
  timestamp: e.ts,
  agentId: e.a,
  level: e.lv,
  message: e.m,
  progress: 0,
} as LogEntry))

interface AppStore {
  // Chat
  conversations: Conversation[]
  activeConversationId: string | null
  streamingMessage: string
  streamingToolCalls: ToolCall[]
  streamingImages: string[]
  streamingCharts: ChartData[]
  isWaiting: boolean
  setWaiting: (v: boolean) => void
  addStreamingToolCall: (tc: ToolCall) => void
  updateStreamingToolCallResult: (id: string, result: string) => void
  addStreamingImage: (b64: string) => void
  addStreamingChart: (chart: ChartData) => void
  setActiveConversation: (id: string) => void
  addMessage: (convId: string, msg: ChatMessage) => void
  appendStreamChunk: (chunk: string) => void
  finalizeStream: (convId: string) => void
  newConversation: () => string
  deleteConversation: (id: string) => void

  // Pipeline
  pipeline: PipelineState
  updateStage: (name: string, patch: Partial<StageState>) => void
  setPipeline: (pipeline: PipelineState) => void

  // Agents
  agents: AgentState[]
  logs: LogEntry[]
  activeLogFilter: string
  updateAgent: (id: string, patch: Partial<AgentState>) => void
  setAgents: (agents: AgentState[]) => void
  addLog: (entry: LogEntry) => void
  setLogs: (entries: LogEntry[]) => void
  setLogFilter: (id: string) => void

  // Config
  config: Partial<AppConfig>
  setConfig: (patch: Partial<AppConfig>) => void
}

export const useStore = create<AppStore>((set, get) => ({
  conversations: [],
  activeConversationId: null,
  streamingMessage: '',
  streamingToolCalls: [],
  streamingImages: [],
  streamingCharts: [],
  isWaiting: false,

  setWaiting: (v) => set({ isWaiting: v }),
  addStreamingImage: (b64) =>
    set((s) => ({ streamingImages: [...s.streamingImages, b64] })),
  addStreamingChart: (chart) =>
    set((s) => ({ streamingCharts: [...s.streamingCharts, chart] })),
  setActiveConversation: (id) => set({ activeConversationId: id }),

  addMessage: (convId, msg) =>
    set((s) => ({
      conversations: s.conversations.map((c) =>
        c.id === convId
          ? {
              ...c,
              messages: [...c.messages, msg],
              // Auto-title from first user message
              title: c.messages.length === 0 && msg.role === 'user'
                ? msg.text.slice(0, 48) + (msg.text.length > 48 ? '…' : '')
                : c.title,
            }
          : c,
      ),
    })),

  appendStreamChunk: (chunk) =>
    set((s) => ({ streamingMessage: s.streamingMessage + chunk })),

  addStreamingToolCall: (tc) =>
    set((s) => ({ streamingToolCalls: [...s.streamingToolCalls, tc] })),

  updateStreamingToolCallResult: (id, result) =>
    set((s) => ({
      streamingToolCalls: s.streamingToolCalls.map((tc) =>
        tc.id === id ? { ...tc, result } : tc
      ),
    })),

  finalizeStream: (convId) => {
    const { streamingMessage, streamingToolCalls, streamingImages, streamingCharts } = get()
    if (!streamingMessage && streamingToolCalls.length === 0 && streamingImages.length === 0 && streamingCharts.length === 0) return
    const msg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'assistant',
      text: streamingMessage,
      timestamp: Date.now(),
      toolCalls: streamingToolCalls.length > 0 ? [...streamingToolCalls] : undefined,
      images: streamingImages.length > 0 ? [...streamingImages] : undefined,
      charts: streamingCharts.length > 0 ? [...streamingCharts] : undefined,
    }
    set((s) => ({
      streamingMessage: '',
      streamingToolCalls: [],
      streamingImages: [],
      streamingCharts: [],
      isWaiting: false,
      conversations: s.conversations.map((c) =>
        c.id === convId ? { ...c, messages: [...c.messages, msg] } : c,
      ),
    }))
  },

  newConversation: () => {
    const id = crypto.randomUUID()
    set((s) => ({
      conversations: [
        { id, title: 'New conversation', messages: [], createdAt: Date.now() },
        ...s.conversations,
      ],
      activeConversationId: id,
    }))
    return id
  },

  deleteConversation: (id) =>
    set((s) => {
      const filtered = s.conversations.filter((c) => c.id !== id)
      return {
        conversations: filtered,
        activeConversationId:
          s.activeConversationId === id
            ? (filtered[0]?.id ?? null)
            : s.activeConversationId,
      }
    }),

  pipeline: { stages: DEFAULT_STAGES, running: false },

  updateStage: (name, patch) =>
    set((s) => ({
      pipeline: {
        ...s.pipeline,
        stages: s.pipeline.stages.map((st) =>
          st.name === name ? { ...st, ...patch } : st,
        ),
      },
    })),

  setPipeline: (pipeline) => set({ pipeline }),

  agents: DEFAULT_AGENTS,
  logs: DEMO_LOGS,
  activeLogFilter: 'all',

  updateAgent: (id, patch) =>
    set((s) => ({
      agents: s.agents.map((a) => (a.id === id ? { ...a, ...patch } : a)),
    })),

  setAgents: (agents) => set({ agents }),

  addLog: (entry) =>
    set((s) => ({ logs: [...s.logs.slice(-499), entry] })),

  setLogs: (entries) => set({ logs: entries.slice(-500) }),

  setLogFilter: (id) => set({ activeLogFilter: id }),

  config: {
    temperature: 0.7,
    topP: 0.95,
    visualFactor: 0.5,
    sampleId: 'DLPFC_151507',
    nClusters: 7,
    methods: ['IRIS', 'BASS', 'DR-SC', 'BayesSpace', 'SEDR', 'GraphST', 'STAGATE', 'stLearn'],
    skipToolRunner: false,
    skipScoring: false,
  },
  setConfig: (patch) => set((s) => ({ config: { ...s.config, ...patch } })),
}))
