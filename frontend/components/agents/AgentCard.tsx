'use client'
import type { AgentState } from '@/lib/types'
import { skipStage, resetStage } from '@/lib/api'

const STATUS_STYLES: Record<AgentState['status'], { bg: string; color: string; border: string }> = {
  IDLE:  { bg: 'var(--s2)',              color: 'var(--ink-ghost)',  border: 'var(--rule)' },
  ACTIVE:{ bg: 'var(--data-dim)',         color: 'var(--data)',       border: 'rgba(14,165,233,0.2)' },
  DONE:  { bg: 'rgba(34,197,94,0.08)',   color: 'var(--ok)',         border: 'rgba(34,197,94,0.2)' },
  SKIPPED:{ bg: 'var(--s2)',             color: 'var(--ink-muted)',  border: 'var(--rule-mid)' },
  ERROR: { bg: 'rgba(239,68,68,0.08)',   color: 'var(--err)',        border: 'rgba(239,68,68,0.2)' },
}

interface Props {
  agent: AgentState
  selected: boolean
  onSelect: (id: string) => void
}

const AGENT_STAGE_MAP: Record<string, string> = {
  TR: 'tool_runner',
  SA: 'scoring',
  BB: 'best',
  AA: 'annotation',
}

export function AgentCard({ agent, selected, onSelect }: Props) {
  const st = STATUS_STYLES[agent.status]

  return (
    <div
      onClick={() => onSelect(agent.id)}
      style={{
        border: `1px solid ${selected ? 'rgba(14,165,233,0.3)' : 'var(--rule)'}`,
        borderRadius: '12px',
        padding: '14px',
        background: selected ? 'rgba(14,165,233,0.04)' : 'var(--s1)',
        cursor: 'pointer',
        transition: 'all 0.15s',
        animation: 'fade-up 0.25s ease-out both',
      }}
      onMouseEnter={(e) => {
        if (!selected) (e.currentTarget as HTMLElement).style.borderColor = 'rgba(0,0,0,0.15)'
      }}
      onMouseLeave={(e) => {
        if (!selected) (e.currentTarget as HTMLElement).style.borderColor = 'var(--rule)'
      }}
    >
      {/* Top row: badge + activity dot */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '10px',
          fontWeight: 600,
          letterSpacing: '0.04em',
          padding: '2px 8px',
          borderRadius: '6px',
          background: st.bg,
          color: st.color,
          border: `1px solid ${st.border}`,
        }}>
          {agent.label}
        </span>

        {agent.status === 'ACTIVE' && (
          <span style={{
            width: '7px',
            height: '7px',
            borderRadius: '50%',
            background: 'var(--data)',
            display: 'block',
            animation: 'pulse-dot 1.5s ease-in-out infinite',
          }} />
        )}
        {agent.status === 'DONE' && (
          <svg viewBox="0 0 14 14" fill="none" width="13" height="13">
            <circle cx="7" cy="7" r="6.5" stroke="var(--ok)" strokeWidth="1" />
            <path d="M4 7.5 L6 9.5 L10 5.5" stroke="var(--ok)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        )}
        {agent.status === 'SKIPPED' && (
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '9px',
            color: 'var(--ink-ghost)',
            letterSpacing: '0.05em',
          }}>
            skip
          </span>
        )}
      </div>

      {/* Name */}
      <div style={{ fontFamily: 'var(--font-display)', fontSize: '13px', fontWeight: 500, color: 'var(--ink-heavy)', marginBottom: '3px' }}>
        {agent.fullName}
      </div>

      {/* Status label */}
      <div style={{
        fontFamily: 'var(--font-mono)',
        fontSize: '10px',
        letterSpacing: '0.05em',
        textTransform: 'uppercase',
        color:
          agent.status === 'ACTIVE'
            ? 'var(--data)'
            : agent.status === 'DONE'
              ? 'var(--ok)'
              : agent.status === 'ERROR'
                ? 'var(--err)'
                : 'var(--ink-ghost)',
        marginBottom: '10px',
      }}>
        {agent.status}
      </div>

      {/* Progress */}
      {agent.status === 'ACTIVE' && (
        <div className="progress-track" style={{ marginBottom: '8px' }}>
          <div
            className="progress-fill"
            style={{ width: `${agent.progress}%`, background: 'var(--data)' }}
          />
        </div>
      )}

      {/* Skip button */}
      {agent.canSkip && agent.status === 'IDLE' && (
        <button
          onClick={(e) => {
            e.stopPropagation()
            const stageName = AGENT_STAGE_MAP[agent.id]
            if (stageName) void skipStage(stageName)
          }}
          className="btn-ghost"
          style={{ fontSize: '10.5px', padding: '3px 10px', marginTop: '2px' }}
        >
          Skip
        </button>
      )}

      {/* Reset button for skipped */}
      {agent.status === 'SKIPPED' && (
        <button
          onClick={(e) => {
            e.stopPropagation()
            const stageName = AGENT_STAGE_MAP[agent.id]
            if (stageName) void resetStage(stageName)
          }}
          className="btn-ghost"
          style={{ fontSize: '10.5px', padding: '3px 10px', marginTop: '2px' }}
        >
          Reset
        </button>
      )}
    </div>
  )
}
