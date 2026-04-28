'use client'
import { useEffect, useRef } from 'react'
import type { LogEntry } from '@/lib/types'

const LEVEL_STYLES: Record<string, { color: string; bg: string }> = {
  info:    { color: 'var(--data)',    bg: 'transparent' },
  success: { color: 'var(--ok)',     bg: 'transparent' },
  warning: { color: 'var(--warn)',   bg: 'transparent' },
  error:   { color: 'var(--err)',    bg: 'rgba(239,68,68,0.04)' },
}

interface Props { entries: LogEntry[]; filter: string }

export function ActivityLog({ entries, filter }: Props) {
  const visible = filter === 'all' ? entries : entries.filter((e) => e.agentId === filter)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [entries.length])

  return (
    <div style={{ background: 'var(--s1)', border: '1px solid var(--rule)', borderRadius: '12px', overflow: 'hidden' }}>
      {/* Header */}
      <div style={{ padding: '10px 14px', borderBottom: '1px solid var(--rule)', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span style={{ fontFamily: 'var(--font-display)', fontSize: '12px', fontWeight: 600, color: 'var(--ink-heavy)' }}>
          Activity Log
        </span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9.5px', color: 'var(--ink-ghost)' }}>
          {visible.length} {filter !== 'all' ? `· ${filter}` : ''}
        </span>
      </div>

      {/* Log rows */}
      <div style={{ overflowY: 'auto', maxHeight: '300px' }}>
        {visible.length === 0 ? (
          <div style={{ padding: '24px', textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: '10.5px', color: 'var(--ink-ghost)' }}>
            No log entries
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <tbody>
              {visible.map((entry, i) => {
                const ls = LEVEL_STYLES[entry.level] ?? { color: 'var(--ink-mid)', bg: 'transparent' }
                return (
                  <tr
                    key={entry.id}
                    style={{
                      borderBottom: '1px solid var(--rule)',
                      background: ls.bg,
                      animation: i === visible.length - 1 ? 'fade-up 0.15s ease-out' : 'none',
                    }}
                  >
                    <td style={{
                      padding: '6px 12px',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '9.5px',
                      color: 'var(--ink-ghost)',
                      whiteSpace: 'nowrap',
                      verticalAlign: 'top',
                    }}>
                      {entry.timestamp}
                    </td>
                    <td style={{ padding: '6px 8px', verticalAlign: 'top' }}>
                      <span style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: '9.5px',
                        background: 'var(--s3)',
                        color: 'var(--ink-mid)',
                        padding: '1px 6px',
                        borderRadius: '4px',
                      }}>
                        {entry.agentId}
                      </span>
                    </td>
                    <td style={{
                      padding: '6px 12px 6px 4px',
                      fontFamily: 'var(--font-display)',
                      fontSize: '11.5px',
                      color: ls.color,
                      verticalAlign: 'top',
                      lineHeight: 1.4,
                    }}>
                      {entry.message}
                    </td>
                  </tr>
                )
              })}
              <tr><td ref={bottomRef as unknown as React.Ref<HTMLTableCellElement>} colSpan={3} /></tr>
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
