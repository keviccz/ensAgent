'use client'
import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import type { ChartData, SpotData, ScoresRow } from '@/lib/types'
import { getSpatialData, getScores } from '@/lib/api'
import { ChatScatter } from './ChatScatter'
import { ScoresMatrix } from '@/components/analysis/ScoresMatrix'

const DOMAIN_COLORS: Record<number, string> = {
  1: '#4472C4', 2: '#ED7D31', 3: '#A9241F',
  4: '#7B5B45', 5: '#D98CB3', 6: '#70AD47', 7: '#4ECDC4',
}
const FALLBACK_COLOR = '#9CA3AF'

interface Props { chart: ChartData }

export function ChatChart({ chart }: Props) {
  if (chart.type === 'scatter') {
    return (
      <div style={{ marginTop: '8px' }}>
        <ChatScatter sampleId={chart.sampleId ?? 'DLPFC_151507'} />
      </div>
    )
  }
  if (chart.type === 'scores_matrix') {
    return <ScoresMatrixChat sampleId={chart.sampleId ?? 'DLPFC_151507'} />
  }
  if (chart.type === 'distributions') {
    return <DistributionsChat sampleId={chart.sampleId ?? 'DLPFC_151507'} />
  }
  return null
}

// ── Scores Matrix ────────────────────────────────────────────────────────────
function ScoresMatrixChat({ sampleId }: { sampleId: string }) {
  const [rows, setRows] = useState<ScoresRow[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getScores(sampleId)
      .then((d) => setRows((d.rows ?? []) as ScoresRow[]))
      .catch(() => { /* ignore */ })
      .finally(() => setLoading(false))
  }, [sampleId])

  if (loading) return <ChartShell title="Scores Matrix" sampleId={sampleId} loading />

  return (
    <div style={{ marginTop: '8px', maxWidth: '520px' }}>
      <ChartLabel title="Scores Matrix" sampleId={sampleId} />
      <ScoresMatrix rows={rows} />
    </div>
  )
}

// ── Domain Distribution + Method Ranking (inline, fixed heights) ─────────────
function DistributionsChat({ sampleId }: { sampleId: string }) {
  const [spots, setSpots] = useState<SpotData[]>([])
  const [rows, setRows] = useState<ScoresRow[]>([])
  const [loading, setLoading] = useState(true)
  const [selDomain, setSelDomain] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([
      getSpatialData(sampleId).then(d => (d.spots ?? []) as SpotData[]),
      getScores(sampleId).then(d => (d.rows ?? []) as ScoresRow[]),
    ])
      .then(([s, r]) => { setSpots(s); setRows(r) })
      .catch(() => { /* ignore */ })
      .finally(() => setLoading(false))
  }, [sampleId])

  if (loading) return <ChartShell title="Domain Distribution · Method Ranking" sampleId={sampleId} loading />

  // Domain Distribution data
  const counts: Record<number, number> = {}
  for (const s of spots) counts[s.cluster] = (counts[s.cluster] || 0) + 1
  const distData = Object.entries(counts)
    .map(([d, count]) => ({ domain: `D${d}`, domainNum: parseInt(d), count }))
    .sort((a, b) => a.domainNum - b.domainNum)
  const total = spots.length

  // Method Ranking data
  const domains = rows.length > 0 ? Object.keys(rows[0].scores).sort() : []
  const rankData = rows.map((row) => {
    const vals = selDomain ? [row.scores[selDomain] ?? 0] : Object.values(row.scores)
    const avg = vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0
    return { method: row.method, score: parseFloat(avg.toFixed(4)) }
  }).sort((a, b) => b.score - a.score)
  const maxScore = Math.max(...rankData.map(d => d.score), 0.01)

  const cardStyle: React.CSSProperties = {
    background: 'var(--s1)',
    border: '1px solid var(--rule)',
    borderRadius: '10px',
    overflow: 'hidden',
  }
  const headerStyle: React.CSSProperties = {
    padding: '7px 12px',
    borderBottom: '1px solid var(--rule)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexShrink: 0,
  }
  const titleStyle: React.CSSProperties = {
    fontFamily: 'var(--font-display)',
    fontSize: '11px',
    fontWeight: 600,
    color: 'var(--ink-heavy)',
  }

  return (
    <div style={{ marginTop: '8px', display: 'flex', flexDirection: 'column', gap: '8px', maxWidth: '520px' }}>
      <ChartLabel title="Domain Distribution · Method Ranking" sampleId={sampleId} />

      {/* Domain Distribution */}
      <div style={cardStyle}>
        <div style={headerStyle}>
          <span style={titleStyle}>Domain Distribution</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'var(--ink-ghost)' }}>
            {distData.length} domains
          </span>
        </div>
        {spots.length === 0 ? (
          <div style={{ height: '130px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--ink-ghost)' }}>No data</span>
          </div>
        ) : (
          <div style={{ height: '130px', padding: '8px 6px 4px 0' }}>
            <ResponsiveContainer width="100%" height={130}>
              <BarChart data={distData} margin={{ top: 4, right: 10, bottom: 2, left: 6 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--rule)" vertical={false} />
                <XAxis dataKey="domain" tick={{ fontFamily: 'var(--font-mono)', fontSize: 9, fill: 'var(--ink-muted)' }} axisLine={{ stroke: 'var(--rule)' }} tickLine={false} />
                <YAxis tick={{ fontFamily: 'var(--font-mono)', fontSize: 8, fill: 'var(--ink-ghost)' }} axisLine={false} tickLine={false} width={36} />
                <Tooltip
                  contentStyle={{ fontFamily: 'var(--font-mono)', fontSize: '10px', background: 'var(--s0)', border: '1px solid var(--rule)', borderRadius: '6px' }}
                  formatter={(v: number) => [`${v} (${((v / total) * 100).toFixed(1)}%)`, 'Count']}
                  cursor={{ fill: 'var(--data-dim)' }}
                />
                <Bar dataKey="count" radius={[3, 3, 0, 0]} maxBarSize={32}>
                  {distData.map((entry) => (
                    <Cell key={entry.domain} fill={DOMAIN_COLORS[entry.domainNum] ?? FALLBACK_COLOR} fillOpacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Method Ranking */}
      <div style={cardStyle}>
        <div style={headerStyle}>
          <span style={titleStyle}>Method Ranking</span>
          <div style={{ display: 'flex', gap: '3px', flexWrap: 'wrap' }}>
            {(['all', ...domains] as string[]).map((d) => {
              const isAll = d === 'all'
              const active = isAll ? selDomain === null : selDomain === d
              return (
                <button
                  key={d}
                  onClick={() => setSelDomain(isAll ? null : d)}
                  style={{
                    fontFamily: 'var(--font-mono)', fontSize: '8px',
                    padding: '1px 5px', borderRadius: '3px',
                    border: `1px solid ${active ? 'var(--data)' : 'var(--rule)'}`,
                    background: active ? 'var(--data-dim)' : 'transparent',
                    color: active ? 'var(--data)' : 'var(--ink-ghost)',
                    cursor: 'pointer',
                  }}
                >
                  {isAll ? 'ALL' : (isNaN(Number(d)) ? d : `D${d}`)}
                </button>
              )
            })}
          </div>
        </div>
        {rows.length === 0 ? (
          <div style={{ height: '160px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--ink-ghost)' }}>No data</span>
          </div>
        ) : (
          <div style={{ padding: '8px 6px 4px 0' }}>
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={rankData} layout="vertical" margin={{ top: 2, right: 14, bottom: 2, left: 2 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--rule)" horizontal={false} />
                <XAxis type="number" domain={[0, Math.ceil(maxScore * 10) / 10]} tick={{ fontFamily: 'var(--font-mono)', fontSize: 8, fill: 'var(--ink-ghost)' }} axisLine={{ stroke: 'var(--rule)' }} tickLine={false} />
                <YAxis type="category" dataKey="method" tick={{ fontFamily: 'var(--font-mono)', fontSize: 9, fill: 'var(--ink-muted)' }} axisLine={false} tickLine={false} width={68} />
                <Tooltip
                  contentStyle={{ fontFamily: 'var(--font-mono)', fontSize: '10px', background: 'var(--s0)', border: '1px solid var(--rule)', borderRadius: '6px' }}
                  formatter={(v: number) => [v.toFixed(4), selDomain ? `D${selDomain} Score` : 'Avg Score']}
                  cursor={{ fill: 'var(--data-dim)' }}
                />
                <Bar dataKey="score" fill="var(--data)" fillOpacity={0.75} radius={[0, 3, 3, 0]} maxBarSize={20} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function ChartLabel({ title, sampleId }: { title: string; sampleId: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'var(--ink-ghost)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
        {title}
      </span>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', background: 'var(--data-dim)', color: 'var(--data)', padding: '1px 6px', borderRadius: '4px' }}>
        {sampleId}
      </span>
    </div>
  )
}

function ChartShell({ title, sampleId, loading }: { title: string; sampleId: string; loading?: boolean }) {
  return (
    <div style={{ marginTop: '8px', maxWidth: '520px' }}>
      <ChartLabel title={title} sampleId={sampleId} />
      <div style={{ border: '1px solid var(--rule)', borderRadius: '10px', padding: '24px', textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--ink-ghost)' }}>
        {loading ? 'loading…' : 'No data'}
      </div>
    </div>
  )
}
