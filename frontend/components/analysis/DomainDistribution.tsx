'use client'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import type { SpotData } from '@/lib/types'

const DOMAIN_COLORS: Record<number, string> = {
  1: '#4472C4',
  2: '#ED7D31',
  3: '#A9241F',
  4: '#7B5B45',
  5: '#D98CB3',
  6: '#70AD47',
  7: '#4ECDC4',
}
const FALLBACK = '#9CA3AF'

interface Props {
  spots: SpotData[]
}

export function DomainDistribution({ spots }: Props) {
  if (spots.length === 0) {
    return (
      <div style={{ background: 'var(--s1)', border: '1px solid var(--rule)', borderRadius: '12px', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10.5px', color: 'var(--ink-ghost)' }}>
          No spatial data
        </span>
      </div>
    )
  }

  const counts: Record<number, number> = {}
  for (const s of spots) {
    counts[s.cluster] = (counts[s.cluster] || 0) + 1
  }
  const data = Object.entries(counts)
    .map(([d, count]) => ({ domain: `D${d}`, domainNum: parseInt(d), count }))
    .sort((a, b) => a.domainNum - b.domainNum)

  const total = spots.length

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
          Domain Distribution
        </span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9.5px', color: 'var(--ink-ghost)' }}>
          {data.length} domains
        </span>
      </div>
      <div style={{ flex: 1, padding: '12px 8px 8px 0', minHeight: '200px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 4, right: 12, bottom: 4, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--rule)" vertical={false} />
            <XAxis
              dataKey="domain"
              tick={{ fontFamily: 'var(--font-mono)', fontSize: 10, fill: 'var(--ink-muted)' }}
              axisLine={{ stroke: 'var(--rule)' }}
              tickLine={false}
            />
            <YAxis
              tick={{ fontFamily: 'var(--font-mono)', fontSize: 9, fill: 'var(--ink-ghost)' }}
              axisLine={false}
              tickLine={false}
              width={40}
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
              formatter={(value: number) => [`${value} spots (${((value / total) * 100).toFixed(1)}%)`, 'Count']}
              cursor={{ fill: 'var(--data-dim)' }}
            />
            <Bar dataKey="count" radius={[4, 4, 0, 0]} maxBarSize={36}>
              {data.map((entry) => (
                <Cell key={entry.domain} fill={DOMAIN_COLORS[entry.domainNum] ?? FALLBACK} fillOpacity={0.85} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
