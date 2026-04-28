import type { AgentState, PipelineState } from './types'

const BASE = '/api'

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json', ...init?.headers },
      ...init,
    })
  } catch (err) {
    throw new Error(`Cannot connect to API server. Make sure you started with "python start.py" and the FastAPI backend is running on localhost:8000.`)
  }
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    if (res.status === 500 && (!body || body === 'Internal Server Error')) {
      throw new Error(`API server returned 500. The FastAPI backend may not be running — start with "python start.py".`)
    }
    throw new Error(`API ${res.status}: ${body || res.statusText}`)
  }
  return res.json() as Promise<T>
}

// ── Config ────────────────────────────────────────────────────────────────────
export const loadConfig = () =>
  apiFetch<Record<string, unknown>>('/config/load')

export const saveConfig = (cfg: Record<string, unknown>) =>
  apiFetch<{ ok: boolean }>('/config/save', { method: 'POST', body: JSON.stringify(cfg) })

export const testConnection = (cfg: Record<string, unknown>) =>
  apiFetch<{ ok: boolean; message: string }>('/config/test_connection', {
    method: 'POST',
    body: JSON.stringify(cfg),
  })

// ── Pipeline ──────────────────────────────────────────────────────────────────
export const runPipeline = () =>
  apiFetch<{ ok: boolean }>('/pipeline/run', { method: 'POST' })

export const runStage = (name: string) =>
  apiFetch<{ ok: boolean }>(`/pipeline/stage/${name}`, { method: 'POST' })

export const skipStage = (name: string) =>
  apiFetch<{ ok: boolean }>('/pipeline/skip', {
    method: 'POST',
    body: JSON.stringify({ name }),
  })

export const resetStage = (name: string) =>
  apiFetch<{ ok: boolean }>('/pipeline/reset', {
    method: 'POST',
    body: JSON.stringify({ name }),
  })

export const getPipelineStatus = () =>
  apiFetch<PipelineState>('/pipeline/status')

// ── Data ──────────────────────────────────────────────────────────────────────
export const getSpatialData = (sampleId: string) =>
  apiFetch<{ spots: unknown[] }>(`/data/spatial?sample_id=${encodeURIComponent(sampleId)}`)

export const getScores = (sampleId: string) =>
  apiFetch<{ rows: unknown[] }>(`/data/scores?sample_id=${encodeURIComponent(sampleId)}`)

export const getAnnotation = (sampleId: string, clusterId: number) =>
  apiFetch<unknown>(`/annotation/${encodeURIComponent(sampleId)}/${clusterId}`)

export const getAnnotationDialogue = (sampleId: string) =>
  apiFetch<import('./types').DialogueData>(`/annotation/dialogue/${encodeURIComponent(sampleId)}`)

export const getAvailableMethods = () =>
  apiFetch<{ methods: string[] }>('/config/methods')

export const getRecentSamples = () =>
  apiFetch<{ samples: string[] }>('/config/recent_samples')

// ── Agents ────────────────────────────────────────────────────────────────────
export const getAgentStatus = () =>
  apiFetch<{ agents: AgentState[]; live: boolean; message: string }>('/agents/status')

export const getAgentHistory = () =>
  apiFetch<{ entries: unknown[] }>('/agents/history')

// ── SSE: chat streaming ───────────────────────────────────────────────────────
export function createChatStream(
  message: string,
  history: { role: string; content: string }[],
  onChunk: (chunk: string) => void,
  onToolCall: (tc: import('./types').ToolCall) => void,
  onToolResult: (id: string, result: string) => void,
  onDone: () => void,
  onError: (err: Error) => void,
  onImage?: (b64: string) => void,
  onChart?: (chart: import('./types').ChartData) => void,
): () => void {
  const controller = new AbortController()

  fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history }),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) throw new Error(`Chat error ${res.status}`)
      const reader = res.body!.getReader()
      const decoder = new TextDecoder()

      // Buffer incomplete SSE lines across network chunks.
      // A single SSE event (e.g. tool_result with image_b64) can be several
      // hundred KB, which the browser splits into many read() chunks.
      let lineBuf = ''

      const processLine = (line: string) => {
        if (!line.startsWith('data: ')) return false
        const raw = line.slice(6).trim()
        if (raw === '[DONE]') { onDone(); return true }
        try {
          const ev = JSON.parse(raw) as {
            type?: string; delta?: string
            id?: string; toolName?: string
            args?: { key: string; value: string }[]
            jsonPayload?: string; pythonSnippet?: string
            result?: string
          }
          const t = ev.type
          if (!t || t === 'delta') {
            onChunk(ev.delta ?? '')
          } else if (t === 'tool_call') {
            onToolCall({
              id: ev.id!,
              toolName: ev.toolName!,
              args: ev.args ?? [],
              jsonPayload: ev.jsonPayload ?? '{}',
              pythonSnippet: ev.pythonSnippet ?? '',
              expanded: false,
            })
          } else if (t === 'tool_result') {
            const resultStr = ev.result ?? ''
            onToolResult(ev.id!, resultStr)
            try {
              const parsed = JSON.parse(resultStr) as Record<string, unknown>
              if (onImage && typeof parsed.image_b64 === 'string' && parsed.image_b64) {
                onImage(parsed.image_b64)
              }
              if (onChart && parsed._chart && typeof parsed._chart === 'object') {
                onChart(parsed._chart as import('./types').ChartData)
              }
            } catch { /* not JSON */ }
          }
        } catch { /* skip malformed JSON */ }
        return false
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        // stream: true keeps the internal state of multi-byte chars correct
        lineBuf += decoder.decode(value, { stream: true })

        // Split on newlines, keep the last (possibly incomplete) segment buffered
        const parts = lineBuf.split('\n')
        lineBuf = parts.pop() ?? ''

        for (const line of parts) {
          if (processLine(line)) return  // [DONE] received
        }
      }

      // Flush any remaining buffered line (edge case: stream ends without trailing \n)
      if (lineBuf) processLine(lineBuf)

      onDone()
    })
    .catch((err: Error) => { if (err.name !== 'AbortError') onError(err) })

  return () => controller.abort()
}

// ── SSE: agent log stream ─────────────────────────────────────────────────────
export function createLogStream(
  onEntry: (entry: unknown) => void,
  onError: (err: Error) => void,
): () => void {
  const es = new EventSource(`${BASE}/agents/logs`)
  es.onmessage = (e) => { try { onEntry(JSON.parse(e.data as string)) } catch { /* skip */ } }
  es.onerror = () => { onError(new Error('Log stream disconnected')); es.close() }
  return () => es.close()
}
