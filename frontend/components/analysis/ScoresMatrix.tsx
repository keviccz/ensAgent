import type { ScoresRow } from '@/lib/types'

// Canonical method labels in display order
const METHOD_LABELS: Record<string, string> = {
  IRIS: 'IRIS', BASS: 'BASS', 'DR-SC': 'DR-SC', BayesSpace: 'BayesSpace',
  SEDR: 'SEDR', GraphST: 'GraphST', STAGATE: 'STAGATE', stLearn: 'stLearn',
}

function cellBg(value: number): string {
  const t = Math.min(Math.max(value, 0), 1)
  // White → sky-blue tint
  return `rgba(14,165,233,${t * 0.55})`
}

function cellText(value: number): string {
  return value > 0.65 ? 'var(--s0)' : 'var(--ink-mid)'
}

export function ScoresMatrix({ rows }: { rows: ScoresRow[] }) {
  if (rows.length === 0) {
    return (
      <div style={{ background: 'var(--s1)', border: '1px solid var(--rule)', borderRadius: '12px', overflow: 'hidden', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10.5px', color: 'var(--ink-ghost)' }}>
          Scores matrix not available
        </span>
      </div>
    )
  }

  const domains = Object.keys(rows[0]?.scores ?? {})
  const rowHeight = `${100 / rows.length}%`

  return (
    <div style={{ background: 'var(--s1)', border: '1px solid var(--rule)', borderRadius: '12px', overflow: 'hidden', height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '10px 14px', borderBottom: '1px solid var(--rule)' }}>
        <span style={{ fontFamily: 'var(--font-display)', fontSize: '12px', fontWeight: 600, color: 'var(--ink-heavy)' }}>
          Scores Matrix
        </span>
      </div>
      <div style={{ overflowX: 'auto', flex: 1, display: 'flex' }}>
        <table style={{ width: '100%', height: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={{
                padding: '7px 12px',
                fontFamily: 'var(--font-mono)',
                fontSize: '9.5px',
                fontWeight: 500,
                letterSpacing: '0.05em',
                textTransform: 'uppercase',
                color: 'var(--ink-ghost)',
                textAlign: 'left',
                borderBottom: '1px solid var(--rule)',
                background: 'var(--s2)',
              }}>
                Method
              </th>
              {domains.map((d) => (
                <th key={d} style={{
                  padding: '5px 6px',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '9.5px',
                  fontWeight: 500,
                  letterSpacing: '0.04em',
                  color: 'var(--ink-ghost)',
                  textAlign: 'center',
                  borderBottom: '1px solid var(--rule)',
                  background: 'var(--s2)',
                }}>
                  {isNaN(Number(d)) ? d : `D${d}`}
                </th>
              ))}
            </tr>
          </thead>
          <tbody style={{ height: '100%' }}>
            {rows.map((row) => (
              <tr key={row.method} style={{ borderBottom: '1px solid var(--rule)', height: rowHeight }}>
                <td style={{
                  padding: '4px 12px',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '10.5px',
                  fontWeight: 500,
                  color: 'var(--ink-mid)',
                  whiteSpace: 'nowrap',
                  background: 'var(--s2)',
                }}>
                  {METHOD_LABELS[row.method] ?? row.method}
                </td>
                {domains.map((d) => {
                  const v = row.scores[d] ?? 0
                  return (
                    <td key={d} style={{
                      padding: '4px 6px',
                      textAlign: 'center',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '10.5px',
                      background: cellBg(v),
                      color: cellText(v),
                      fontVariantNumeric: 'tabular-nums',
                      transition: 'background 0.2s',
                    }}>
                      {v.toFixed(2)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
