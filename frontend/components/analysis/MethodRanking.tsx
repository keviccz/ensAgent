'use client'
import { useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import type { ScoresRow } from '@/lib/types'

interface Props {
  rows: ScoresRow[]
}

interface MethodTickProps {
  x?: number
  y?: number
  payload?: {
    value?: string | number
  }
}

const METHOD_TICK_FONT_SIZE = 10
const BAYES_SPACE_TICK_FONT_SIZE = 9

function renderMethodTick({ x, y, payload }: MethodTickProps) {
  const method = String(payload?.value ?? '')
  const fontSize = method === 'BayesSpace' ? BAYES_SPACE_TICK_FONT_SIZE : METHOD_TICK_FONT_SIZE

  return (
    <text
      x={x}
      y={y}
      dy={4}
      textAnchor="end"
      fill="var(--ink-muted)"
      style={{ fontFamily: 'var(--font-mono)', fontSize }}
    >
      {method}
    </text>
  )
}

export function MethodRanking({ rows }: Props) {
  const domains = rows.length > 0 ? Object.keys(rows[0].scores).sort() : []
  const [selectedDomain, setSelectedDomain] = useState<string | null>(null)

  if (rows.length === 0) {
    return (
      <div style={{ background: 'var(--s1)', border: '1px solid var(--rule)', borderRadius: '12px', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10.5px', color: 'var(--ink-ghost)' }}>
          Scores data not available
        </span>
      </div>
    )
  }

  // Build per-method data: average across all domains, or score for selected domain
  const data = rows.map((row) => {
    const vals = selectedDomain
      ? [row.scores[selectedDomain] ?? 0]
      : Object.values(row.scores)
    const avg = vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0
    return { method: row.method, score: parseFloat(avg.toFixed(4)) }
  }).sort((a, b) => b.score - a.score)

  const maxScore = Math.max(...data.map(d => d.score), 0.01)
  const chartHeight = Math.max(240, data.length * 34 + 40)

  return (
    <div style={{
      background: 'var(--s1)',
      border: '1px solid var(--rule)',
      borderRadius: '12px',
      overflow: 'hidden',
      display: 'flex',
      flexDirection: 'column',
    }}>
      <div style={{
        padding: '10px 14px',
        borderBottom: '1px solid var(--rule)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexShrink: 0,
      }}>
        <span style={{ fontFamily: 'var(--font-display)', fontSize: '12px', fontWeight: 600, color: 'var(--ink-heavy)' }}>
          Method Ranking
        </span>
        <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
          <button
            onClick={() => setSelectedDomain(null)}
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '9px',
              padding: '2px 6px',
              borderRadius: '4px',
              border: `1px solid ${selectedDomain === null ? 'var(--data)' : 'var(--rule)'}`,
              background: selectedDomain === null ? 'var(--data-dim)' : 'transparent',
              color: selectedDomain === null ? 'var(--data)' : 'var(--ink-ghost)',
              cursor: 'pointer',
            }}
          >
            ALL
          </button>
          {domains.map((d) => (
            <button
              key={d}
              onClick={() => setSelectedDomain(d)}
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '9px',
                padding: '2px 6px',
                borderRadius: '4px',
                border: `1px solid ${selectedDomain === d ? 'var(--data)' : 'var(--rule)'}`,
                background: selectedDomain === d ? 'var(--data-dim)' : 'transparent',
                color: selectedDomain === d ? 'var(--data)' : 'var(--ink-ghost)',
                cursor: 'pointer',
              }}
            >
              {isNaN(Number(d)) ? d : `D${d}`}
            </button>
          ))}
        </div>
      </div>
      <div style={{ height: `${chartHeight}px`, padding: '12px 8px 8px 0' }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, bottom: 4, left: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--rule)" horizontal={false} />
            <XAxis
              type="number"
              domain={[0, Math.ceil(maxScore * 10) / 10]}
              tick={{ fontFamily: 'var(--font-mono)', fontSize: 9, fill: 'var(--ink-ghost)' }}
              axisLine={{ stroke: 'var(--rule)' }}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="method"
              tick={renderMethodTick}
              axisLine={false}
              tickLine={false}
              interval={0}
              width={72}
            />
            <Tooltip
              contentStyle={{
                fontFamily: 'var(--font-mono)',
                fontSize: '11px',
                background: 'var(--s0)',
                border: '1px solid var(--rule)',
                borderRadius: '8px',
                boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
              }}
              formatter={(value: number) => [value.toFixed(4), selectedDomain ? `D${selectedDomain} Score` : 'Avg Score']}
              cursor={{ fill: 'var(--data-dim)' }}
            />
            <Bar dataKey="score" fill="var(--data)" fillOpacity={0.75} radius={[0, 4, 4, 0]} maxBarSize={24} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
