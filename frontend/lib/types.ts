// ── Chat ──────────────────────────────────────────────────────────────────────
export type MessageRole = 'user' | 'assistant'

export interface ReasoningStep {
  index: number
  text: string
}

export interface ToolCallArg {
  key: string
  value: string
}

export interface ToolCall {
  id: string
  toolName: string
  args: ToolCallArg[]
  jsonPayload: string
  pythonSnippet: string
  expanded: boolean
  result?: string
}

// ── Chat Charts (rendered from tool results) ──────────────────────────────────
export type ChartType = 'scatter' | 'scores_matrix' | 'distributions'

export interface ChartData {
  type: ChartType
  sampleId?: string
}

export interface ChatMessage {
  id: string
  role: MessageRole
  text: string
  reasoning?: ReasoningStep[]
  toolCalls?: ToolCall[]
  images?: string[]   // base64-encoded PNGs from tool results
  charts?: ChartData[] // interactive chart tokens from tool results
  timestamp: number
}

export interface Conversation {
  id: string
  title: string
  messages: ChatMessage[]
  createdAt: number
}

// ── Pipeline ──────────────────────────────────────────────────────────────────
export type StageStatus = 'idle' | 'running' | 'done' | 'error' | 'skipped'

export interface StageState {
  name: string
  label: string
  status: StageStatus
  progress: number
}

export interface PipelineState {
  stages: StageState[]
  running: boolean
}

// ── Agents ────────────────────────────────────────────────────────────────────
export type AgentStatus = 'IDLE' | 'ACTIVE' | 'DONE' | 'ERROR' | 'SKIPPED'

export interface AgentState {
  id: string
  label: string
  fullName: string
  status: AgentStatus
  progress: number
  canSkip: boolean
}

export type LogLevel = 'info' | 'success' | 'warning' | 'error'

export interface LogEntry {
  id: string
  timestamp: string
  agentId: string
  level: LogLevel
  message: string
}

// ── Analysis ──────────────────────────────────────────────────────────────────
export interface SpotData {
  spotId: string
  x: number
  y: number
  cluster: number
  expression?: number
}

export interface AnnotationResult {
  clusterId: number
  label: string
  confidence: number
  markerGenes: string[]
  interpretation: string
}

export interface ScoresRow {
  method: string
  scores: Record<string, number>
}

// ── Annotation Dialogue ───────────────────────────────────────────────────────
export interface ExpertOutput {
  agent: string
  agent_score: number
  label_support?: Record<string, number>
  celltype_support?: Record<string, number>
  reasoning?: string
  evidence?: { id: string; type: string; items: string[]; direction?: string; strength?: number; note?: string }[]
  quality_flags?: string[]
}

export interface AnnotationDraft {
  domain_id: number
  biological_identity: string
  biological_identity_conf: number
  margin?: number
  alternatives?: { label: string; conf: number }[]
  primary_cell_types?: string[]
  key_evidence?: string[]
  reasoning?: string
  function?: string
}

export interface CriticIssue {
  type: string
  severity: number
  blocker?: boolean
  rerun_agent?: string
  fix_hint?: string
}

export interface CriticOutput {
  critic_score: number
  reasoning?: string
  feedback_by_agent?: Record<string, string[]>
  issues?: CriticIssue[]
  rerun_request?: { rerun: boolean; agents?: string[]; reason?: string }
}

export interface RoundLog {
  round: number
  run_dir?: string
  experts?: Record<string, ExpertOutput>
  annotation?: AnnotationDraft
  critic?: CriticOutput
  gate?: { passed: boolean; final_score?: number; fail_reasons?: string[] }
}

export interface DialogueData {
  sample_id: string
  domains: number[]
  rounds: Record<number, RoundLog[]>
}

// ── Config ────────────────────────────────────────────────────────────────────
export interface ApiConfig {
  apiProvider: string
  apiKey: string
  apiModel: string
  apiEndpoint: string
  apiVersion: string
  temperature: number
  topP: number
  visualFactor: number
}

export interface PipelineConfig {
  dataPath: string
  sampleId: string
  nClusters: number
  methods: string[]
  skipToolRunner: boolean
  skipScoring: boolean
}

export interface AnnotationExpertConfig {
  annotationWMarker: number
  annotationWPathway: number
  annotationWSpatial: number
  annotationWVlm: number
  annotationMaxRounds: number
  annotationStandardScore: number
  annotationVlmRequired: boolean
  annotationVlmMinScore: number
}

export interface AppConfig extends ApiConfig, PipelineConfig, AnnotationExpertConfig {}
