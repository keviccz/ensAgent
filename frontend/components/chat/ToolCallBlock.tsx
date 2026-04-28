'use client'
import { useState } from 'react'
import { ChevronRight, ChevronDown } from 'lucide-react'
import type { ToolCall } from '@/lib/types'

function syntaxHighlight(json: string): string {
  return json
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, (match) => {
      let cls = 'code-num'
      if (/^"/.test(match)) {
        cls = /:$/.test(match) ? 'code-key' : 'code-str'
      } else if (/true|false/.test(match)) {
        cls = 'code-bool'
      } else if (/null/.test(match)) {
        cls = 'code-null'
      }
      return `<span class="${cls}">${match}</span>`
    })
}

export function ToolCallBlock({ call }: { call: ToolCall }) {
  const [expanded, setExpanded] = useState(call.expanded)
  const [tab, setTab] = useState<'json' | 'python'>('json')

  const formatted = (() => {
    try { return JSON.stringify(JSON.parse(call.jsonPayload), null, 2) } catch { return call.jsonPayload }
  })()

  return (
    <div
      className="my-1.5 overflow-hidden"
      style={{ border: '1px solid var(--rule-mid)', borderRadius: '10px' }}
    >
      {/* Header row */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 w-full text-left transition-colors duration-100"
        style={{
          padding: '7px 10px',
          background: expanded ? 'var(--s1)' : 'var(--s2)',
          border: 'none',
          cursor: 'pointer',
          borderBottom: expanded ? '1px solid var(--rule)' : 'none',
        }}
        onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'var(--s3)' }}
        onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = expanded ? 'var(--s1)' : 'var(--s2)' }}
      >
        {expanded
          ? <ChevronDown size={11} strokeWidth={2} style={{ color: 'var(--ink-ghost)', flexShrink: 0 }} />
          : <ChevronRight size={11} strokeWidth={2} style={{ color: 'var(--ink-ghost)', flexShrink: 0 }} />
        }

        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '11px',
          fontWeight: 500,
          color: 'var(--ink-mid)',
          flexShrink: 0,
        }}>
          {call.toolName}
        </span>

        <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap', marginLeft: '4px' }}>
          {call.args.slice(0, 4).map((arg) => (
            <span
              key={arg.key}
              style={{
                background: 'var(--data-dim)',
                color: 'var(--data)',
                fontFamily: 'var(--font-mono)',
                fontSize: '9.5px',
                padding: '1px 6px',
                borderRadius: '4px',
              }}
            >
              {arg.key}=<span style={{ opacity: 0.8 }}>{arg.value.slice(0, 24)}{arg.value.length > 24 ? '…' : ''}</span>
            </span>
          ))}
        </div>
      </button>

      {/* Expanded body */}
      {expanded && (
        <div>
          {/* Tab row */}
          <div style={{ display: 'flex', borderBottom: '1px solid var(--rule)', background: 'var(--s1)' }}>
            {(['json', 'python'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                style={{
                  padding: '5px 12px',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '10px',
                  letterSpacing: '0.03em',
                  textTransform: 'lowercase',
                  color: tab === t ? 'var(--ink-heavy)' : 'var(--ink-ghost)',
                  borderBottom: tab === t ? '2px solid var(--ink-heavy)' : '2px solid transparent',
                  background: 'transparent',
                  border: 'none',
                  cursor: 'pointer',
                  transition: 'color 0.15s',
                }}
              >
                {t}
              </button>
            ))}
          </div>

          {/* Code */}
          <div
            className="code-block"
            style={{ maxHeight: '140px', overflowY: 'auto' }}
            dangerouslySetInnerHTML={
              tab === 'json'
                ? { __html: syntaxHighlight(formatted) }
                : { __html: call.pythonSnippet }
            }
          />
        </div>
      )}
    </div>
  )
}
