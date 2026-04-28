'use client'
import { useEffect, useRef, useState } from 'react'
import { buildKpiItems } from '@/lib/analysisMetrics'
import type { ScoresRow, SpotData } from '@/lib/types'

// Animated counter — eases from 0 to target value
function AnimatedValue({ raw }: { raw: string }) {
  const [display, setDisplay] = useState('—')
  const rafRef = useRef<number | null>(null)
  const prevRef = useRef<string | null>(null)

  useEffect(() => {
    if (raw === prevRef.current) return
    prevRef.current = raw

    // Parse numeric value (strip commas)
    const cleaned = raw.replace(/,/g, '')
    const num = parseFloat(cleaned)
    if (isNaN(num) || raw === '--') { setDisplay(raw); return }

    const isDecimal = raw.includes('.')
    const decimals = isDecimal ? (raw.split('.')[1]?.length ?? 2) : 0
    const duration = 700
    const start = performance.now()

    if (rafRef.current) cancelAnimationFrame(rafRef.current)

    const tick = (now: number) => {
      const t = Math.min((now - start) / duration, 1)
      // Ease-out cubic
      const eased = 1 - Math.pow(1 - t, 3)
      const current = num * eased
      setDisplay(decimals > 0
        ? current.toFixed(decimals)
        : Math.round(current).toLocaleString()
      )
      if (t < 1) rafRef.current = requestAnimationFrame(tick)
      else setDisplay(raw)
    }

    rafRef.current = requestAnimationFrame(tick)
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current) }
  }, [raw])

  return <>{display}</>
}

interface Props {
  spots: SpotData[]
  scoreRows: ScoresRow[]
}

export function KpiStrip({ spots, scoreRows }: Props) {
  const items = buildKpiItems(spots, scoreRows)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    if (spots.length > 0 || scoreRows.length > 0) setMounted(true)
  }, [spots, scoreRows])

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px', marginBottom: '20px' }}>
      {items.map((item, i) => (
        <div
          key={item.label}
          style={{
            background: 'var(--s1)',
            border: '1px solid var(--rule)',
            borderRadius: '12px',
            padding: '14px 16px',
            opacity: mounted ? 1 : 0,
            transform: mounted ? 'translateY(0)' : 'translateY(8px)',
            transition: `opacity 0.4s ease-out ${i * 0.07}s, transform 0.4s ease-out ${i * 0.07}s`,
          }}
        >
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '9.5px',
            letterSpacing: '0.07em',
            textTransform: 'uppercase',
            color: 'var(--ink-ghost)',
            marginBottom: '6px',
          }}>
            {item.label}
          </div>
          <div style={{
            fontFamily: 'var(--font-display)',
            fontSize: '26px',
            fontWeight: 600,
            letterSpacing: '-0.025em',
            color: 'var(--ink-heavy)',
            lineHeight: 1.1,
          }}>
            <AnimatedValue raw={item.value} />
          </div>
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '10px',
            color: 'var(--ink-ghost)',
            marginTop: '4px',
          }}>
            {item.sub}
          </div>
        </div>
      ))}
    </div>
  )
}
