'use client'
import { useEffect, useRef, useState } from 'react'
import { Search, ChevronDown } from 'lucide-react'
import { Topbar } from '@/components/layout/Topbar'
import { AgentGrid } from '@/components/agents/AgentGrid'
import { FilterBar } from '@/components/agents/FilterBar'
import { ActivityLog } from '@/components/agents/ActivityLog'
import { AnnotationDialogue } from '@/components/agents/AnnotationDialogue'
import { useStore } from '@/lib/store'
import { getAgentStatus, getAgentHistory, createLogStream, getAnnotationDialogue, getRecentSamples } from '@/lib/api'
import type { AgentState, LogEntry, DialogueData } from '@/lib/types'

export default function AgentsPage() {
  const { agents, logs, activeLogFilter, setAgents, addLog, setLogs, setLogFilter, config } = useStore()
  const [fetchError, setFetchError] = useState<string | null>(null)

  // Annotation Dialogue state
  const [sampleInput, setSampleInput] = useState('')
  const [recentSamples, setRecentSamples] = useState<string[]>([])
  const [showDropdown, setShowDropdown] = useState(false)
  const [dialogueData, setDialogueData] = useState<DialogueData | null>(null)
  const [dialogueLoading, setDialogueLoading] = useState(false)
  const [dialogueError, setDialogueError] = useState<string | null>(null)

  const historyLoaded = useRef(false)

  useEffect(() => {
    let cancelled = false

    // Load history once on mount — overwrite demo logs only if API returns more
    if (!historyLoaded.current) {
      historyLoaded.current = true
      getAgentHistory()
        .then((res) => {
          if (!cancelled && Array.isArray(res.entries) && res.entries.length > 10) {
            const entries = (res.entries as object[]).map((e, i) => ({
              ...(e as object),
              id: `hist-${i}`,
            })) as LogEntry[]
            setLogs(entries)
          }
        })
        .catch(() => { /* ignore — demo logs already shown */ })
    }

    // Load recent samples
    getRecentSamples()
      .then((res) => {
        if (!cancelled) {
          setRecentSamples(res.samples ?? [])
          // Auto-populate from config sample_id
          const configSample = config.sampleId
          const best = configSample || res.samples?.[0] || ''
          if (best && !sampleInput) setSampleInput(best)
        }
      })
      .catch(() => { /* ignore */ })

    const refreshAgents = () => {
      getAgentStatus()
        .then((d) => {
          if (!cancelled) {
            setAgents((d as { agents: AgentState[] }).agents ?? [])
            setFetchError(null)
          }
        })
        .catch((err) => {
          if (!cancelled) setFetchError(`Cannot reach API: ${err.message}`)
        })
    }

    refreshAgents()
    const timer = window.setInterval(refreshAgents, 4000)

    const stop = createLogStream(
      (entry) => addLog({ ...(entry as object), id: crypto.randomUUID() } as LogEntry),
      () => { /* silent on disconnect */ },
    )
    return () => {
      cancelled = true
      window.clearInterval(timer)
      stop()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setAgents, addLog, setLogs])

  const handleLoadDialogue = async (sid?: string) => {
    const id = (sid ?? sampleInput).trim()
    if (!id) return
    if (sid) setSampleInput(sid)
    setShowDropdown(false)
    setDialogueLoading(true)
    setDialogueError(null)
    try {
      const data = await getAnnotationDialogue(id)
      if (data.domains.length === 0) {
        setDialogueError('No annotation round logs found for this sample. Run Stage D first.')
        setDialogueData(null)
      } else {
        setDialogueData(data)
      }
    } catch (err: unknown) {
      setDialogueError(err instanceof Error ? err.message : 'Failed to load dialogue data')
      setDialogueData(null)
    } finally {
      setDialogueLoading(false)
    }
  }

  const activeCount = agents.filter((a) => a.status === 'ACTIVE').length
  const doneCount   = agents.filter((a) => a.status === 'DONE' || a.status === 'SKIPPED').length

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      <Topbar
        title="Agents"
        subtitle={`${activeCount} active · ${doneCount} done`}
        showDot={activeCount > 0}
      />
      <div style={{ flex: 1, overflowY: 'auto', padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {fetchError && (
          <div style={{
            background: 'rgba(239,68,68,0.06)',
            border: '1px solid rgba(239,68,68,0.2)',
            borderRadius: '8px',
            padding: '10px 14px',
            fontFamily: 'var(--font-mono)',
            fontSize: '11px',
            color: 'var(--err)',
            lineHeight: 1.5,
          }}>
            {fetchError}
          </div>
        )}

        <AgentGrid
          agents={agents}
          selectedId={activeLogFilter}
          onSelect={(id) => setLogFilter(id === activeLogFilter ? 'all' : id)}
        />

        {/* Activity log */}
        <div>
          <FilterBar
            agents={agents.map((a) => ({ id: a.id, label: a.label }))}
            active={activeLogFilter}
            onFilter={setLogFilter}
          />
          <ActivityLog entries={logs} filter={activeLogFilter} />
        </div>

        {/* Annotation Dialogue */}
        <div>
          <div className="section-head" style={{ marginBottom: '10px' }}>Annotation Dialogue</div>

          {/* Sample input with recent samples dropdown */}
          <div style={{ display: 'flex', gap: '8px', marginBottom: '12px', position: 'relative' }}>
            <div style={{ position: 'relative', flex: 1, maxWidth: '320px' }}>
              <input
                type="text"
                className="input-base"
                value={sampleInput}
                onChange={(e) => setSampleInput(e.target.value)}
                placeholder="Sample ID (e.g. DLPFC_151507)"
                onKeyDown={(e) => e.key === 'Enter' && handleLoadDialogue()}
                style={{ paddingRight: '30px', width: '100%' }}
              />
              {recentSamples.length > 0 && (
                <button
                  type="button"
                  onClick={() => setShowDropdown((s) => !s)}
                  style={{
                    position: 'absolute', right: '8px', top: '50%', transform: 'translateY(-50%)',
                    background: 'none', border: 'none', cursor: 'pointer',
                    color: 'var(--ink-ghost)', display: 'flex', alignItems: 'center', padding: '2px',
                  }}
                >
                  <ChevronDown size={13} />
                </button>
              )}
              {/* Dropdown */}
              {showDropdown && recentSamples.length > 0 && (
                <div style={{
                  position: 'absolute', top: '100%', left: 0, right: 0, zIndex: 50,
                  background: 'var(--s1)', border: '1px solid var(--rule-mid)',
                  borderRadius: '8px', marginTop: '4px',
                  boxShadow: '0 8px 24px rgba(0,0,0,0.1)', overflow: 'hidden',
                }}>
                  {recentSamples.map((s) => (
                    <button
                      key={s}
                      onClick={() => handleLoadDialogue(s)}
                      style={{
                        width: '100%', textAlign: 'left', padding: '8px 12px',
                        background: s === sampleInput ? 'var(--data-dim)' : 'transparent',
                        border: 'none', cursor: 'pointer',
                        fontFamily: 'var(--font-mono)', fontSize: '11px',
                        color: s === sampleInput ? 'var(--data)' : 'var(--ink-mid)',
                        borderBottom: '1px solid var(--rule)',
                        transition: 'background 0.1s',
                      }}
                      onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'var(--s2)' }}
                      onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = s === sampleInput ? 'var(--data-dim)' : 'transparent' }}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              )}
            </div>

            <button
              className="btn-primary"
              onClick={() => handleLoadDialogue()}
              disabled={dialogueLoading || !sampleInput.trim()}
              style={{ fontSize: '12px', gap: '5px', flexShrink: 0 }}
            >
              <Search size={12} />
              {dialogueLoading ? 'Loading…' : 'Load'}
            </button>
          </div>

          {dialogueError && (
            <div style={{
              fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--warn)',
              background: 'rgba(234,179,8,0.06)', border: '1px solid rgba(234,179,8,0.2)',
              borderRadius: '8px', padding: '10px 14px', marginBottom: '10px',
            }}>
              {dialogueError}
            </div>
          )}

          {dialogueData && <AnnotationDialogue data={dialogueData} />}

          {!dialogueData && !dialogueError && !dialogueLoading && (
            <div style={{
              fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-ghost)',
              textAlign: 'center', padding: '20px',
              border: '1px dashed var(--rule)', borderRadius: '8px',
            }}>
              Select a sample to view Proposer / Critic annotation dialogue
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
